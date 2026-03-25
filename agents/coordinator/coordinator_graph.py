"""
Coordinator Agent — LangGraph StateGraph orchestration.

Graph topology
--------------
                        classify_intent
                              |
                        route_intent
                       /      |      \\
          handle_process  handle_query  (modification/approve/undo branches)
              |                |
          run_pipeline   build_response
              |
          check_hitl
              |
        build_response

Full routing:
  process         -> handle_process -> run_pipeline -> check_hitl -> build_response
  query           -> handle_query -> build_response
  modify_strategy -> handle_modification -> build_response
  ignore          -> handle_modification -> build_response
  strengthen      -> handle_modification -> build_response
  approve         -> handle_approve -> build_response
  undo            -> handle_undo -> build_response
  reject          -> handle_undo -> build_response   (same as undo)

All write nodes call Safety Kernel tools and record to the audit trail.
VLM calls (explanation / summary) are deferred to build_response.

Key ML integrations:
  - intent_classifier.hybrid_classify()    -- classify every user message
  - hitl_confidence.compute_hitl_decision() -- gate after each pipeline run
  - reexecution_graph.compute_reexecution_plan() -- selective re-execution
  - coordinator_prompts.build_explain_prompt()  -- explain_decision queries

Bug fixes in this version:
  1. route_from_intent uses current_intent["action"] directly (no _route key)
  2. _apply_pending_modifications patches strategy_result in-place before runs
  3. Uses compute_reexecution_plan().must_rerun for stage determination
  4. 120s timeout per stage via concurrent.futures.ThreadPoolExecutor
  5. Reads HITL mode from deps (NodeContext) not hardcoded HYBRID
  6. Real VLM intent classification via vlm_call_fn from ctx
  7. Graph invocation is synchronous (no async)

Multi-agent enhancements:
  1. Disagreement detection after risk and strategy stages
  2. Grouped HITL presentation with confidence, groups, suggestions
  3. Snapshot before re-execution for undo support
  4. Wire adaptive_learning: PreferenceManager integration
  5. Dynamic agent skipping: consent if 0 faces, SAM if all text
"""
from __future__ import annotations

import copy
import logging
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# LangGraph is an optional dependency -- degrade gracefully so imports work
# even in environments without it (e.g., during unit tests).
try:
    from langgraph.graph import StateGraph, END  # type: ignore[import]
    _LANGGRAPH_AVAILABLE = True
    try:
        from langgraph.types import interrupt  # type: ignore[import]
        _INTERRUPT_AVAILABLE = True
    except ImportError:
        _INTERRUPT_AVAILABLE = False
        interrupt = None
    try:
        from langgraph.errors import GraphInterrupt  # type: ignore[import]
    except ImportError:
        GraphInterrupt = None  # type: ignore[assignment,misc]
except ImportError:  # pragma: no cover
    _LANGGRAPH_AVAILABLE = False
    _INTERRUPT_AVAILABLE = False
    StateGraph = None  # type: ignore[assignment,misc]
    END = "__end__"
    interrupt = None
    GraphInterrupt = None  # type: ignore[assignment,misc]

from agents.coordinator.state import (
    CoordinatorState,
    InnerPipelineState,
    DisagreementEvent,
    PipelineSnapshot,
    STAGE_ORDER,
)
from agents.coordinator.nodes import NodeContext

# ML components
from agents.coordinator.intent_classifier import (
    IntentAction,
    ParsedIntent,
    hybrid_classify,
    needs_clarification,
    generate_clarification_prompt,
)
from agents.coordinator.hitl_confidence import (
    HITLMode,
    CheckpointType,
    compute_hitl_decision,
)
from agents.coordinator.reexecution_graph import (
    ModificationType,
    ReExecutionPlan,
    PipelineStage,
    compute_reexecution_plan,
    intent_to_modification_type,
)
from agents.coordinator.coordinator_prompts import (
    build_explain_prompt,
    build_summary_prompt,
    parse_summary_response,
)

# Stage timeout in seconds (per-stage ceiling)
_STAGE_TIMEOUT_S = 120

# Maximum snapshots to keep
_MAX_SNAPSHOTS = 10

# PipelineStage enum -> canonical stage name mapping
_STAGE_ENUM_TO_NAME: Dict[PipelineStage, str] = {
    PipelineStage.DETECT:    "detection",
    PipelineStage.RISK:      "risk",
    PipelineStage.CONSENT:   "consent",
    PipelineStage.STRATEGY:  "strategy",
    PipelineStage.SAM:       "sam",
    PipelineStage.EXECUTION: "execution",
    PipelineStage.EXPORT:    "export",
}
# Helpers

def _now_ms() -> float:
    return time.time() * 1000


def _get_hitl_mode(ctx: NodeContext) -> HITLMode:
    """Read HITL mode from NodeContext or default to HYBRID."""
    if hasattr(ctx, "hitl_mode") and ctx.hitl_mode is not None:
        try:
            return HITLMode(ctx.hitl_mode)
        except ValueError:
            pass
    return HITLMode.HYBRID


def _extract_assessments_list(risk_result: Any) -> List[Dict]:
    """Safely extract assessments as a list of dicts from risk_result."""
    if risk_result is None:
        return []
    if isinstance(risk_result, dict):
        items = risk_result.get("assessments", [])
    elif hasattr(risk_result, "assessments"):
        items = list(risk_result.assessments)
    else:
        return []
    return [
        a if isinstance(a, dict) else (a.dict() if hasattr(a, "dict") else {})
        for a in items
    ]


def _extract_strategies_list(strategy_result: Any) -> List[Dict]:
    """Safely extract strategies as a list of dicts from strategy_result."""
    if strategy_result is None:
        return []
    if isinstance(strategy_result, dict):
        items = strategy_result.get("strategies", [])
    elif hasattr(strategy_result, "strategies"):
        items = list(strategy_result.strategies)
    else:
        return []
    return [
        s if isinstance(s, dict) else (s.dict() if hasattr(s, "dict") else {})
        for s in items
    ]


def _make_vlm_call_fn(ctx: NodeContext):
    """
    Build a VLM call function from the NodeContext if a VLM wrapper is available.
    Returns None if no VLM is configured (regex-only fallback).
    """
    if ctx.fallback_only:
        return None
    # Check if ctx has a vlm_wrapper or similar
    vlm = getattr(ctx, "vlm_wrapper", None) or getattr(ctx, "vlm", None)
    if vlm is None:
        return None

    def _call(system_prompt: str, user_msg: str) -> str:
        try:
            return vlm.call(system_prompt=system_prompt, user_message=user_msg)
        except Exception as exc:
            logger.warning("VLM intent call failed: %s", exc)
            return ""

    return _call


def _run_node_with_timeout(node_fn, pipeline_state, ctx, timeout_s=_STAGE_TIMEOUT_S):
    """Run a pipeline node function with a timeout. Returns updated state."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(node_fn, pipeline_state, ctx)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            stage_name = getattr(node_fn, "__name__", "unknown")
            logger.error("Stage %s timed out after %ds", stage_name, timeout_s)
            errors = dict(pipeline_state.get("errors") or {})
            errors[stage_name] = f"Timed out after {timeout_s}s"
            return {**pipeline_state, "errors": errors}
# Snapshot helpers

def _take_snapshot(state: CoordinatorState) -> PipelineSnapshot:
    """Create a PipelineSnapshot from the current coordinator state."""
    pipeline_state = state.get("pipeline_state") or {}

    # Deep-copy serialisable parts only (skip _cached_image)
    def _safe_copy(obj):
        if obj is None:
            return None
        try:
            if isinstance(obj, dict):
                return copy.deepcopy(obj)
            elif hasattr(obj, "dict"):
                return copy.deepcopy(obj.dict())
            return copy.deepcopy(obj)
        except Exception:
            return None

    return {  # type: ignore[return-value]
        "snapshot_id": uuid.uuid4().hex[:12],
        "timestamp": time.time(),
        "entry_stage": pipeline_state.get("entry_stage", "detection"),
        "detections": _safe_copy(pipeline_state.get("detections")),
        "risk_result": _safe_copy(pipeline_state.get("risk_result")),
        "strategy_result": _safe_copy(pipeline_state.get("strategy_result")),
        "execution_report": _safe_copy(pipeline_state.get("execution_report")),
        "protected_image_path": pipeline_state.get("protected_image_path"),
    }


def _restore_snapshot(state: CoordinatorState, snapshot: Dict) -> CoordinatorState:
    """Restore a PipelineSnapshot into the coordinator state."""
    pipeline_state = dict(state.get("pipeline_state") or {})

    for key in ("detections", "risk_result", "strategy_result",
                "execution_report", "protected_image_path"):
        if key in snapshot and snapshot[key] is not None:
            pipeline_state[key] = snapshot[key]

    pipeline_state["entry_stage"] = snapshot.get("entry_stage", "detection")
    pipeline_state["pending_modifications"] = []

    return {  # type: ignore[return-value]
        **state,
        "pipeline_state": pipeline_state,
    }
# Disagreement detection

def _detect_disagreements_risk(
    pipeline_state: Dict,
) -> List[Dict[str, Any]]:
    """
    Compare Phase 1 vs Phase 2 risk results and return DisagreementEvents.

    Phase 2 results are identified by the vlm_phase2_ran flag on assessments.
    If vlm_phase2_ran is True and the severity changed, that is a disagreement.
    """
    disagreements: List[Dict[str, Any]] = []
    assessments = _extract_assessments_list(pipeline_state.get("risk_result"))

    for a in assessments:
        if not a.get("vlm_phase2_ran", False):
            continue
        # Phase 1 severity is stored in original_severity if VLM changed it
        phase1_sev = a.get("original_severity") or a.get("phase1_severity", "")
        phase2_sev = a.get("severity", "")
        if phase1_sev and phase2_sev and phase1_sev.lower() != phase2_sev.lower():
            disagreements.append({
                "stage": "risk",
                "detection_id": a.get("detection_id", "unknown"),
                "field": "severity",
                "phase1_value": phase1_sev.lower(),
                "phase2_value": phase2_sev.lower(),
                "reasoning": a.get("reasoning", "VLM reclassified severity"),
            })

    return disagreements


def _detect_disagreements_strategy(
    pipeline_state: Dict,
) -> List[Dict[str, Any]]:
    """
    Compare Phase 1 vs Phase 2 strategy results and return DisagreementEvents.
    """
    disagreements: List[Dict[str, Any]] = []
    strategies = _extract_strategies_list(pipeline_state.get("strategy_result"))

    for s in strategies:
        phase1_method = s.get("original_method") or s.get("phase1_method", "")
        phase2_method = s.get("method", "")
        if phase1_method and phase2_method and phase1_method != phase2_method:
            disagreements.append({
                "stage": "strategy",
                "detection_id": s.get("detection_id", "unknown"),
                "field": "method",
                "phase1_value": phase1_method,
                "phase2_value": phase2_method,
                "reasoning": s.get("reasoning", "VLM modified strategy method"),
            })

    return disagreements
# Grouped HITL presentation

def _build_hitl_presentation(
    state: CoordinatorState,
    confidence_score: float,
    checkpoint_type: str,
) -> Dict[str, Any]:
    """
    Build grouped HITL presentation data for the UI.

    Groups elements by type and summarises methods and counts.
    """
    pipeline_state = state.get("pipeline_state") or {}
    strategies = _extract_strategies_list(pipeline_state.get("strategy_result"))
    assessments = _extract_assessments_list(pipeline_state.get("risk_result"))
    disagreements = list(pipeline_state.get("phase_disagreements") or [])

    # Group by element_type
    type_groups: Dict[str, List[Dict]] = defaultdict(list)
    for s in strategies:
        etype = s.get("element_type", "unknown")
        type_groups[etype].append(s)

    groups = []
    for etype, items in sorted(type_groups.items()):
        methods_used = list(set(
            s.get("method", "none") for s in items
            if s.get("method") not in (None, "none")
        ))
        protected = [s for s in items if s.get("method") not in (None, "none")]
        count = len(protected)
        if count == 0:
            continue

        method_str = ", ".join(sorted(methods_used)) if methods_used else "none"
        summary = f"{count} {etype}(s) -> {method_str}"
        groups.append({
            "type": etype,
            "count": count,
            "methods": methods_used,
            "summary": summary,
        })

    # Suggestions
    suggestions = []
    if not disagreements:
        suggestions.append("Approve all")
    else:
        suggestions.append("Review disagreements individually")

    has_faces = any(g["type"] == "face" for g in groups)
    if has_faces:
        suggestions.append("Review faces individually")

    return {
        "checkpoint": checkpoint_type,
        "confidence": round(confidence_score, 3),
        "groups": groups,
        "disagreements": disagreements,
        "suggestions": suggestions,
    }
# Apply pending modifications to strategy_result in-place

def _apply_pending_modifications(state: CoordinatorState) -> CoordinatorState:
    """
    Patch strategy_result in-place based on pending_modifications
    before re-running the pipeline.

    This ensures that user-requested changes (method overrides, ignores,
    strengthens) are reflected in the strategy before execution.
    """
    pipeline_state = dict(state.get("pipeline_state") or {})
    pending_mods = pipeline_state.get("pending_modifications") or []

    if not pending_mods:
        return state

    strategy_result = pipeline_state.get("strategy_result")
    if strategy_result is None:
        return state

    # Get mutable strategies list
    if isinstance(strategy_result, dict):
        strategies = list(strategy_result.get("strategies", []))
    elif hasattr(strategy_result, "strategies"):
        strategies = [
            s if isinstance(s, dict) else (s.dict() if hasattr(s, "dict") else {})
            for s in strategy_result.strategies
        ]
    else:
        return state

    # Build lookup
    strat_by_id = {}
    for i, s in enumerate(strategies):
        det_id = s.get("detection_id") if isinstance(s, dict) else getattr(s, "detection_id", None)
        if det_id:
            strat_by_id[det_id] = i

    for mod in pending_mods:
        det_id = mod.get("detection_id")
        if det_id not in strat_by_id:
            continue
        idx = strat_by_id[det_id]
        s = strategies[idx] if isinstance(strategies[idx], dict) else dict(strategies[idx])

        action = mod.get("type", "")
        value = mod.get("value", "")

        if action in ("ignore", IntentAction.IGNORE.value):
            s["method"] = "none"
        elif action in ("modify_strategy", IntentAction.MODIFY_STRATEGY.value):
            if value:
                s["method"] = value
        elif action in ("strengthen", IntentAction.STRENGTHEN.value):
            if value and value != "none":
                s["method"] = value
            expand_px = mod.get("expand_px", 0)
            if expand_px:
                params = dict(s.get("parameters") or {})
                params["expand_px"] = expand_px
                s["parameters"] = params

        strategies[idx] = s

    # Write back
    if isinstance(strategy_result, dict):
        strategy_result = {**strategy_result, "strategies": strategies}
    else:
        strategy_result = {"strategies": strategies}

    pipeline_state["strategy_result"] = strategy_result
    return {**state, "pipeline_state": pipeline_state}  # type: ignore[return-value]
# Dynamic agent skipping

def _compute_skip_stages(pipeline_state: Dict) -> Set[str]:
    """
    Determine which stages can be skipped based on pipeline state.

    - Skip consent if 0 faces detected.
    - Skip SAM if all protected elements are text (text uses bbox overlay).
    """
    skip: Set[str] = set()

    detections = pipeline_state.get("detections")
    if detections is not None:
        # Count faces
        if isinstance(detections, dict):
            items = detections.get("detections", [])
        elif hasattr(detections, "detections"):
            items = list(detections.detections)
        else:
            items = []

        face_count = sum(
            1 for d in items
            if (d.get("element_type") if isinstance(d, dict)
                else getattr(d, "element_type", "")) == "face"
        )
        if face_count == 0:
            skip.add("consent")

    strategy_result = pipeline_state.get("strategy_result")
    if strategy_result is not None:
        strategies = _extract_strategies_list(strategy_result)
        protected = [
            s for s in strategies
            if s.get("method") not in (None, "none")
        ]
        if protected and all(s.get("element_type") == "text" for s in protected):
            skip.add("sam")

    return skip
# Node implementations

def node_classify_intent(state: CoordinatorState, ctx: NodeContext) -> CoordinatorState:
    """
    Classify the last user message into a ParsedIntent.

    Uses hybrid_classify (regex fast-path + optional VLM Level 2).
    Stores the serialised ParsedIntent in state["current_intent"].
    """
    history = state.get("conversation_history") or []
    if not history:
        logger.warning("classify_intent called with empty conversation history")
        return {  # type: ignore[return-value]
            **state,
            "current_intent": {
                "action": IntentAction.QUERY.value,
                "target_stage": "",
                "confidence": 0.3,
                "natural_language": "",
            },
        }

    # Get last user message
    last_user_msg = ""
    for msg in reversed(history):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    pipeline_state = state.get("pipeline_state") or {}
    assessments = _extract_assessments_list(pipeline_state.get("risk_result"))

    context = {
        "current_stage": pipeline_state.get("entry_stage", "detection"),
        "element_count": len(assessments),
        "has_critical": any(
            a.get("severity", "").lower() == "critical" for a in assessments
        ),
        "pending_hitl": state.get("pending_user_decision", False),
        "recent_actions": [
            e.get("action") for e in (state.get("audit_trail") or [])[-3:]
            if isinstance(e, dict)
        ],
    }
    vlm_call_fn = _make_vlm_call_fn(ctx)

    intent = hybrid_classify(
        query=last_user_msg,
        vlm_call_fn=vlm_call_fn,
        context=context,
    )

    return {  # type: ignore[return-value]
        **state,
        "current_intent": {
            "action": intent.action.value,
            "target_stage": intent.target_stage,
            "target_elements": intent.target_elements,
            "target_element_types": intent.target_element_types,
            "confidence": intent.confidence,
            "method_specified": intent.method_specified,
            "strength_parameter": intent.strength_parameter,
            "natural_language": intent.natural_language,
            "extracted_constraints": intent.extracted_constraints,
            "requires_safety_check": intent.requires_safety_check,
            "requires_checkpoint": intent.requires_checkpoint,
        },
    }


def node_handle_process(state: CoordinatorState, ctx: NodeContext) -> CoordinatorState:
    """
    Handle a PROCESS intent: prepare the InnerPipelineState for a full pipeline run.

    Initialises a fresh InnerPipelineState (or resets an existing one if the user
    is re-processing the same image).  The actual pipeline execution happens in
    node_run_pipeline.
    """
    pipeline_state = state.get("pipeline_state") or {}
    image_path = pipeline_state.get("image_path", "")
    session_id = state.get("session_id", "")
    fallback_only = ctx.fallback_only

    fresh_inner: InnerPipelineState = {  # type: ignore[assignment]
        "session_id": session_id,
        "image_path": image_path,
        "detections": None,
        "risk_result": None,
        "identity_assessments": None,
        "strategy_result": None,
        "seg_results": None,
        "execution_report": None,
        "protected_image_path": None,
        "risk_map_path": None,
        "strategy_json_path": None,
        "entry_stage": "detection",
        "fallback_only": fallback_only,
        "pending_modifications": [],
        "stage_timings": {},
        "errors": {},
        "phase_disagreements": [],
        "_cached_image": None,
    }

    return {  # type: ignore[return-value]
        **state,
        "pipeline_state": fresh_inner,
        "pipeline_action_taken": "pipeline_started",
    }


def node_handle_modification(
    state: CoordinatorState, ctx: NodeContext
) -> CoordinatorState:
    """
    Handle MODIFY_STRATEGY, IGNORE, and STRENGTHEN intents.

    1. Extract target element(s) from the intent.
    2. Call the appropriate write tool (apply_strategy_change / apply_ignore /
       apply_strengthen) -- each is Safety Kernel gated.
    3. Compute the ReExecutionPlan so the graph knows which stages to re-run.
    4. Update pending_modifications and pipeline_action_taken.
    5. Wire adaptive_learning: record method overrides for preference learning.

    Errors (BLOCK / not found) are reported in response_text.
    """
    from agents.coordinator.tools import (  # noqa: PLC0415
        apply_strategy_change, apply_ignore, apply_strengthen,
    )

    intent = state.get("current_intent") or {}
    action = intent.get("action", "")
    detection_ids: Optional[List[str]] = intent.get("target_elements")
    new_method: Optional[str] = intent.get("method_specified")
    session_id = state.get("session_id", "")
    pipeline_state = state.get("pipeline_state") or {}
    safety_kernel = ctx.safety_kernel

    audit_trail = list(state.get("audit_trail") or [])
    pending_mods = list(pipeline_state.get("pending_modifications") or [])
    response_parts: List[str] = []
    actions_taken: List[str] = []
    errors_encountered: List[str] = []

    if safety_kernel is None:
        return {  # type: ignore[return-value]
            **state,
            "response_text": (
                "Safety Kernel is not available. Cannot apply modifications. "
                "Please check service status."
            ),
            "pipeline_action_taken": "modification_failed:no_safety_kernel",
        }

    # If no specific elements were targeted, use all applicable ones
    if not detection_ids:
        from agents.coordinator.tools import _all_strategies  # noqa: PLC0415
        strategies = _all_strategies(pipeline_state)
        element_types = intent.get("target_element_types") or []
        if element_types:
            detection_ids = [
                s.get("detection_id") for s in strategies
                if s.get("element_type") in element_types
                and s.get("detection_id")
            ]
        else:
            detection_ids = [
                s.get("detection_id") for s in strategies if s.get("detection_id")
            ]

    if not detection_ids:
        return {  # type: ignore[return-value]
            **state,
            "response_text": (
                "No elements found to modify. Run the pipeline first or "
                "specify which element you want to change."
            ),
            "pipeline_action_taken": "modification_skipped:no_targets",
        }

    # Get preference manager from ctx if available (enhancement #4)
    preference_manager = getattr(ctx, "preference_manager", None)

    for det_id in (detection_ids or []):
        try:
            if action == IntentAction.IGNORE.value:
                result = apply_ignore(
                    safety_kernel=safety_kernel,
                    pipeline_state=pipeline_state,
                    session_id=session_id,
                    detection_id=det_id,
                )
                mod_type = ModificationType.IGNORE_ELEMENT

            elif action == IntentAction.STRENGTHEN.value:
                result = apply_strengthen(
                    safety_kernel=safety_kernel,
                    pipeline_state=pipeline_state,
                    session_id=session_id,
                    detection_id=det_id,
                    new_method=new_method,
                )
                mod_type = ModificationType.STRENGTHEN_ONLY

            else:  # MODIFY_STRATEGY
                if not new_method:
                    errors_encountered.append(
                        f"{det_id}: No method specified for modification."
                    )
                    continue
                result = apply_strategy_change(
                    safety_kernel=safety_kernel,
                    pipeline_state=pipeline_state,
                    session_id=session_id,
                    detection_id=det_id,
                    new_method=new_method,
                )
                mod_type = ModificationType.METHOD_ONLY_CHANGE

            safety_action = result.get("safety_action", "allow")
            approved_value = result.get("approved_value", "")

            if safety_action == "challenge":
                # Surface the challenge to the user; don't apply yet
                response_parts.append(
                    f"Warning for {det_id}: {result.get('challenge_message', '')} "
                    "Reply 'yes' to confirm."
                )
                continue

            # Record in pending_modifications
            pending_mods.append({
                "detection_id": det_id,
                "type": action,
                "value": approved_value,
                "modification_type": mod_type.value,
                "timestamp": time.time(),
            })
            actions_taken.append(f"{det_id}:{action}={approved_value}")
            if preference_manager is not None and action == IntentAction.MODIFY_STRATEGY.value:
                from agents.coordinator.tools import _find_strategy  # noqa: PLC0415
                old_strat = _find_strategy(pipeline_state, det_id) or {}
                old_method = old_strat.get("method", "none")
                if old_method != approved_value:
                    try:
                        # Find element type
                        from agents.coordinator.tools import _find_assessment  # noqa: PLC0415
                        assessment = _find_assessment(pipeline_state, det_id) or {}
                        etype = assessment.get("element_type", "unknown")
                        preference_manager.record_method_override(
                            etype, old_method, approved_value, session_id
                        )
                    except Exception as pref_exc:
                        logger.debug("Preference recording failed: %s", pref_exc)

            # Audit trail entry
            audit_trail.append({
                "action": action,
                "detection_id": det_id,
                "value": approved_value,
                "rule_id": result.get("rule_id"),
                "timestamp": time.time(),
            })

        except ValueError as exc:
            errors_encountered.append(f"{det_id}: {exc}")

    # Update pipeline_state with new pending_modifications
    updated_pipeline_state = {
        **pipeline_state,
        "pending_modifications": pending_mods,
    }

    # Build response text
    if actions_taken:
        response_parts.append(
            f"Applied {len(actions_taken)} modification(s): {', '.join(actions_taken)}. "
            "The pipeline will re-run from the appropriate stage."
        )
    if errors_encountered:
        response_parts.append(
            f"The following could not be applied: {'; '.join(errors_encountered)}"
        )

    pipeline_action = (
        f"modifications_applied:{len(actions_taken)}"
        if actions_taken
        else "modifications_blocked"
    )

    return {  # type: ignore[return-value]
        **state,
        "pipeline_state": updated_pipeline_state,
        "audit_trail": audit_trail,
        "pipeline_action_taken": pipeline_action,
        "response_text": " ".join(response_parts) or "No changes were applied.",
    }


def node_handle_query(state: CoordinatorState, ctx: NodeContext) -> CoordinatorState:
    """
    Handle QUERY intents: return explanation for a specific element or general state.

    For element-specific queries: calls explain_decision() to build the VLM prompt.
    For general queries: returns a structured summary from query_* read tools.
    """
    from agents.coordinator.tools import (  # noqa: PLC0415
        query_risk_assessments, query_strategies, explain_decision,
    )

    intent = state.get("current_intent") or {}
    detection_ids: Optional[List[str]] = intent.get("target_elements")
    pipeline_state = state.get("pipeline_state") or {}
    user_question = intent.get("natural_language", "")

    # Element-specific explanation
    if detection_ids and len(detection_ids) == 1:
        try:
            system_prompt, user_msg = explain_decision(
                pipeline_state=pipeline_state,
                detection_id=detection_ids[0],
                user_question=user_question,
            )
            # For now return a structured answer without calling VLM
            # (VLM call would happen in production via build_response)
            risks = query_risk_assessments(pipeline_state)
            assessment = next(
                (a for a in risks.get("assessments", [])
                 if a.get("detection_id") == detection_ids[0]),
                {}
            )
            response = (
                f"Element {detection_ids[0]} ({assessment.get('element_type', 'unknown')}): "
                f"severity={assessment.get('severity', 'unknown')}, "
                f"risk_type={assessment.get('risk_type', 'unknown')}. "
                f"{assessment.get('reasoning', 'No reasoning available.')}"
            )
        except ValueError as exc:
            response = str(exc)

        return {  # type: ignore[return-value]
            **state,
            "response_text": response,
            "pipeline_action_taken": "query_answered",
        }

    # General state summary
    risks = query_risk_assessments(pipeline_state)
    strats = query_strategies(pipeline_state)

    if risks.get("error"):
        response = (
            "The pipeline has not run yet. Say 'process' to analyse your image."
        )
    else:
        response = (
            f"Found {risks.get('total', 0)} element(s): "
            f"{sum(1 for a in risks.get('assessments', []) if a.get('severity') == 'critical')} critical, "
            f"{sum(1 for a in risks.get('assessments', []) if a.get('severity') == 'high')} high. "
            f"Protecting {strats.get('protected_count', 0)} element(s), "
            f"skipping {strats.get('skipped_count', 0)}."
        )

    return {  # type: ignore[return-value]
        **state,
        "response_text": response,
        "pipeline_action_taken": "query_answered",
    }


def node_handle_approve(state: CoordinatorState, ctx: NodeContext) -> CoordinatorState:
    """
    Handle APPROVE intent: advance past the current HITL checkpoint.

    Clears pending_user_decision and records the approval in the audit trail.
    If pipeline_service is available, signals the threading.Event to resume.
    """
    audit_trail = list(state.get("audit_trail") or [])
    audit_trail.append({
        "action": "approved",
        "checkpoint": state.get("hitl_checkpoint"),
        "timestamp": time.time(),
    })

    # Signal pipeline to resume (if available)
    if ctx.pipeline_service is not None:
        session_id = state.get("session_id", "")
        if ctx.session_manager is not None:
            session_record = ctx.session_manager.get_by_id(session_id)
            if session_record is not None:
                try:
                    ctx.pipeline_service.approve_checkpoint(
                        session=session_record,
                        checkpoint=state.get("hitl_checkpoint") or "strategy_review",
                    )
                except Exception as exc:
                    logger.warning("approve_checkpoint failed: %s", exc)

    return {  # type: ignore[return-value]
        **state,
        "pending_user_decision": False,
        "audit_trail": audit_trail,
        "pipeline_action_taken": "checkpoint_approved",
        "response_text": "Approved. The pipeline will continue.",
    }


def node_handle_undo(state: CoordinatorState, ctx: NodeContext) -> CoordinatorState:
    """
    Handle UNDO / REJECT intents: restore the most recent snapshot if available,
    otherwise pop the last pending modification.

    Enhancement #3: Uses PipelineSnapshot for real undo capability.
    """
    snapshots = list(state.get("snapshots") or [])
    pipeline_state = state.get("pipeline_state") or {}
    pending_mods = list(pipeline_state.get("pending_modifications") or [])
    audit_trail = list(state.get("audit_trail") or [])

    # Try snapshot-based undo first
    if snapshots:
        snapshot = snapshots.pop()
        restored_state = _restore_snapshot(state, snapshot)
        audit_trail.append({
            "action": "undo_snapshot",
            "snapshot_id": snapshot.get("snapshot_id", ""),
            "timestamp": time.time(),
        })
        return {  # type: ignore[return-value]
            **restored_state,
            "snapshots": snapshots,
            "audit_trail": audit_trail,
            "pipeline_action_taken": "undo_snapshot_restored",
            "response_text": (
                f"Restored to snapshot {snapshot.get('snapshot_id', '?')} "
                f"(entry: {snapshot.get('entry_stage', 'unknown')})."
            ),
        }

    # Fallback: pop last pending modification
    if pending_mods:
        reverted = pending_mods.pop()
        audit_trail.append({
            "action": "undo",
            "reverted": reverted,
            "timestamp": time.time(),
        })
        response = (
            f"Reverted: {reverted.get('type', 'modification')} on "
            f"{reverted.get('detection_id', 'element')}."
        )
    else:
        response = "Nothing to undo -- no recent modifications or snapshots found."

    updated_pipeline_state = {**pipeline_state, "pending_modifications": pending_mods}

    return {  # type: ignore[return-value]
        **state,
        "pipeline_state": updated_pipeline_state,
        "audit_trail": audit_trail,
        "pipeline_action_taken": "undo_applied",
        "response_text": response,
    }


def node_run_pipeline(state: CoordinatorState, ctx: NodeContext) -> CoordinatorState:
    """
    Execute inner pipeline stages based on the re-execution plan.

    Determines entry_stage from pending_modifications (selective re-execution)
    or uses full pipeline for first run.  Calls node functions from nodes.py
    in topological order starting at entry_stage.

    Bug fixes:
      #2: Apply pending modifications to strategy_result before execution
      #3: Use compute_reexecution_plan().must_rerun for stage set
      #4: 120s timeout per stage via ThreadPoolExecutor

    Enhancements:
      #1: Disagreement detection after risk and strategy stages
      #3: Snapshot before re-execution
      #5: Dynamic agent skipping
    """
    from agents.coordinator.nodes import (  # noqa: PLC0415
        node_detection, node_risk, node_consent, node_strategy,
        node_sam, node_execution, node_export,
    )
    snapshots = list(state.get("snapshots") or [])
    existing_pipeline = state.get("pipeline_state") or {}
    if existing_pipeline.get("detections") is not None:
        # Only snapshot if there is existing state worth preserving
        snapshot = _take_snapshot(state)
        snapshots.append(snapshot)
        if len(snapshots) > _MAX_SNAPSHOTS:
            snapshots = snapshots[-_MAX_SNAPSHOTS:]
    state = _apply_pending_modifications(state)

    pipeline_state: InnerPipelineState = dict(state.get("pipeline_state") or {})  # type: ignore[assignment]
    pending_mods = pipeline_state.get("pending_modifications") or []
    session_id = state.get("session_id", "")
    reexec_plan: Optional[ReExecutionPlan] = None
    if pending_mods:
        # Find the most impactful modification type
        mod_types = []
        for m in pending_mods:
            raw = m.get("modification_type", "full_pipeline")
            if raw in ModificationType._value2member_map_:
                mod_types.append(ModificationType(raw))
            else:
                mod_types.append(ModificationType.FULL_PIPELINE)

        # Priority ordering: pick the most impactful
        _MOD_PRIORITY = {
            ModificationType.FULL_PIPELINE: 10,
            ModificationType.DETECTION_CHANGE: 9,
            ModificationType.SEVERITY_CHANGE: 7,
            ModificationType.CONSENT_CHANGE: 6,
            ModificationType.IGNORE_ELEMENT: 5,
            ModificationType.METHOD_WITH_MASK_CHANGE: 4,
            ModificationType.METHOD_ONLY_CHANGE: 3,
            ModificationType.STRENGTHEN_ONLY: 2,
            ModificationType.ADD_REGION: 1,
        }
        most_impactful = max(mod_types, key=lambda mt: _MOD_PRIORITY.get(mt, 0))
        reexec_plan = compute_reexecution_plan(most_impactful)
        entry_stage = _STAGE_ENUM_TO_NAME.get(reexec_plan.entry_stage, "detection")
    else:
        entry_stage = pipeline_state.get("entry_stage", "detection")

    pipeline_state["entry_stage"] = entry_stage

    # Clear pending modifications once we start running
    pipeline_state["pending_modifications"] = []

    # Initialise phase_disagreements if not present
    if "phase_disagreements" not in pipeline_state:
        pipeline_state["phase_disagreements"] = []  # type: ignore[typeddict-unknown-key]

    # Node dispatch table
    _NODES = {
        "detection": node_detection,
        "risk":      node_risk,
        "consent":   node_consent,
        "strategy":  node_strategy,
        "sam":       node_sam,
        "execution": node_execution,
        "export":    node_export,
    }

    # Build set of stages to run
    start_idx = STAGE_ORDER.index(entry_stage) if entry_stage in STAGE_ORDER else 0
    candidate_stages = STAGE_ORDER[start_idx:]

    # If we have a reexec_plan, use must_rerun to filter
    if reexec_plan is not None:
        must_rerun_names = set(
            _STAGE_ENUM_TO_NAME.get(s, "") for s in reexec_plan.must_rerun
        )
        stages_to_run = [s for s in candidate_stages if s in must_rerun_names]
    else:
        stages_to_run = list(candidate_stages)
    skip_stages = _compute_skip_stages(pipeline_state)
    stages_to_run = [s for s in stages_to_run if s not in skip_stages]

    logger.info(
        "run_pipeline session=%s entry=%s stages=%s skipped=%s",
        session_id, entry_stage, stages_to_run, skip_stages,
    )

    for stage in stages_to_run:
        node_fn = _NODES.get(stage)
        if node_fn is None:
            continue
        try:
            pipeline_state = _run_node_with_timeout(  # type: ignore[assignment]
                node_fn, pipeline_state, ctx, _STAGE_TIMEOUT_S
            )
        except Exception as exc:
            logger.error("Stage %s crashed: %s", stage, exc)
            errors = dict(pipeline_state.get("errors") or {})
            errors[stage] = str(exc)
            pipeline_state = {**pipeline_state, "errors": errors}  # type: ignore[assignment]
            # Stop pipeline on hard failure; export whatever we have
            break
        if stage == "risk":
            disagreements = _detect_disagreements_risk(pipeline_state)
            existing = list(pipeline_state.get("phase_disagreements") or [])
            existing.extend(disagreements)
            pipeline_state["phase_disagreements"] = existing  # type: ignore[typeddict-unknown-key]
        elif stage == "strategy":
            disagreements = _detect_disagreements_strategy(pipeline_state)
            existing = list(pipeline_state.get("phase_disagreements") or [])
            existing.extend(disagreements)
            pipeline_state["phase_disagreements"] = existing  # type: ignore[typeddict-unknown-key]

    return {  # type: ignore[return-value]
        **state,
        "pipeline_state": pipeline_state,
        "pipeline_action_taken": f"pipeline_ran:entry={entry_stage}",
        "snapshots": snapshots,
    }


def node_check_hitl(state: CoordinatorState, ctx: NodeContext) -> CoordinatorState:
    """
    Post-pipeline HITL confidence gate.

    Calls compute_hitl_decision() on the completed risk assessments and
    strategies to determine whether to pause for user review.

    Bug fix #5: Reads HITL mode from ctx (not hardcoded HYBRID).
    Enhancement #1: Disagreements reduce confidence by 0.1 each.
    Enhancement #2: Builds grouped HITL presentation.
    Enhancement #4: Applies adaptive_learning preferences before presenting.
    """
    pipeline_state = state.get("pipeline_state") or {}
    risk_result = pipeline_state.get("risk_result")
    strategy_result = pipeline_state.get("strategy_result")

    if risk_result is None:
        # Pipeline didn't reach risk stage -- no HITL needed
        return {  # type: ignore[return-value]
            **state,
            "hitl_checkpoint": None,
            "pending_user_decision": False,
            "hitl_confidence": None,
            "hitl_presentation": None,
        }

    assessments = _extract_assessments_list(risk_result)
    strategies = _extract_strategies_list(strategy_result)
    preference_manager = getattr(ctx, "preference_manager", None)
    session_id = state.get("session_id", "")
    if preference_manager is not None:
        adapted_strategies = []
        for s in strategies:
            adapted = preference_manager.adapt_strategy(dict(s), session_id)
            adapted_strategies.append(adapted)
        strategies = adapted_strategies
    hitl_mode = _get_hitl_mode(ctx)

    report = compute_hitl_decision(
        risk_assessments=assessments,
        strategies=strategies,
        mode=hitl_mode,
    )
    disagreements = list(pipeline_state.get("phase_disagreements") or [])
    adjusted_score = report.session_score
    if disagreements:
        penalty = len(disagreements) * 0.1
        adjusted_score = max(0.0, adjusted_score - penalty)
        logger.info(
            "HITL confidence adjusted by %d disagreements: %.2f -> %.2f",
            len(disagreements), report.session_score, adjusted_score,
        )

    checkpoint_type = report.checkpoint_type

    # Re-evaluate checkpoint with adjusted score if it changed thresholds
    from agents.coordinator.hitl_confidence import (
        AUTO_ADVANCE_THRESHOLD, STRATEGY_REVIEW_THRESHOLD,
    )
    if hitl_mode == HITLMode.HYBRID:
        if (adjusted_score >= AUTO_ADVANCE_THRESHOLD
                and not report.has_critical_elements
                and not report.has_consent_conflicts):
            checkpoint_type = CheckpointType.AUTO_ADVANCE_SUMMARY
        elif adjusted_score >= STRATEGY_REVIEW_THRESHOLD or report.has_critical_elements:
            checkpoint_type = CheckpointType.STRATEGY_REVIEW
        else:
            checkpoint_type = CheckpointType.RISK_REVIEW

    pending = checkpoint_type != CheckpointType.AUTO_ADVANCE_SUMMARY

    report_dict = {
        "session_score": adjusted_score,
        "original_score": report.session_score,
        "checkpoint_type": checkpoint_type.value,
        "has_critical_elements": report.has_critical_elements,
        "has_consent_conflicts": report.has_consent_conflicts,
        "has_unverified_screens": report.has_unverified_screens,
        "low_confidence_elements": report.low_confidence_elements,
        "rationale": report.rationale,
        "auto_advance_possible": (checkpoint_type == CheckpointType.AUTO_ADVANCE_SUMMARY),
        "disagreement_count": len(disagreements),
    }
    presentation = _build_hitl_presentation(
        state, adjusted_score, checkpoint_type.value
    )

    # Use LangGraph interrupt if available and we need to pause.
    # GraphInterrupt (BaseException subclass) MUST propagate so the
    # checkpointer can suspend the graph.  Only catch plain Exception
    # (e.g. when running outside a real LangGraph context).
    if pending and _INTERRUPT_AVAILABLE and interrupt is not None:
        try:
            interrupt(presentation)
        except (GraphInterrupt, BaseException) as exc:
            # Let GraphInterrupt propagate — it's how the checkpointer pauses.
            if GraphInterrupt is not None and isinstance(exc, GraphInterrupt):
                raise
            # Other BaseException subclasses (KeyboardInterrupt, etc.) — re-raise
            if not isinstance(exc, Exception):
                raise
            # Plain Exception: interrupt not supported in this context — continue
            pass

    return {  # type: ignore[return-value]
        **state,
        "hitl_checkpoint": checkpoint_type.value,
        "pending_user_decision": pending,
        "hitl_confidence": report_dict,
        "hitl_presentation": presentation,
    }


def node_build_response(state: CoordinatorState, ctx: NodeContext) -> CoordinatorState:
    """
    Build the final response_text and suggestions for the current turn.

    For PROCESS intents: generates a pipeline summary.
    For other intents: response_text was set by the handler node.
    Appends the assistant response to conversation_history.
    """
    pipeline_state = state.get("pipeline_state") or {}
    intent = state.get("current_intent") or {}
    action = intent.get("action", "")
    current_response = state.get("response_text", "")
    suggestions: List[str] = list(state.get("suggestions") or [])

    # Augment response for HITL pause
    if state.get("pending_user_decision") and not current_response:
        checkpoint = state.get("hitl_checkpoint", "")
        report = state.get("hitl_confidence") or {}
        presentation = state.get("hitl_presentation") or {}

        # Use grouped presentation if available
        groups_text = ""
        for g in presentation.get("groups", []):
            groups_text += f"\n  - {g.get('summary', '')}"

        current_response = (
            f"Pipeline paused at {checkpoint}. "
            f"Session confidence: {report.get('session_score', 0):.2f}."
        )
        if report.get("disagreement_count", 0) > 0:
            current_response += (
                f" ({report['disagreement_count']} Phase 1/2 disagreement(s) detected.)"
            )
        current_response += f"\n{report.get('rationale', '')}"
        if groups_text:
            current_response += f"\nProtection summary:{groups_text}"
        current_response += "\nReply 'yes' to approve and continue."

        suggestions = list(presentation.get("suggestions", []))
        if not suggestions:
            suggestions = [
                "Say 'yes' to approve and continue.",
                "Say 'no' or 'undo' to revert.",
                "Ask 'why' followed by an element ID to get an explanation.",
            ]

    # For PROCESS action: generate summary when pipeline completed
    elif action == IntentAction.PROCESS.value and not current_response:
        risk_result = pipeline_state.get("risk_result")
        strategy_result = pipeline_state.get("strategy_result")
        timings = pipeline_state.get("stage_timings") or {}
        total_time_ms = sum(timings.values())

        if risk_result is None:
            current_response = (
                "Pipeline encountered an error during processing. "
                "Check the errors field for details."
            )
        else:
            assessments = _extract_assessments_list(risk_result)
            strats = _extract_strategies_list(strategy_result)

            n_protected = sum(
                1 for s in strats if s.get("method") not in {None, "none"}
            )
            n_critical = sum(
                1 for a in assessments if a.get("severity", "").lower() == "critical"
            )

            current_response = (
                f"Pipeline complete. Found {len(assessments)} element(s). "
                f"Protecting {n_protected} element(s)"
                + (f", including {n_critical} critical item(s)" if n_critical else "")
                + f". Total time: {total_time_ms:.0f}ms."
            )

            if pipeline_state.get("protected_image_path"):
                suggestions.append(
                    "The protected image is ready for download."
                )

    # Append to conversation history
    history = list(state.get("conversation_history") or [])
    if current_response:
        history.append({"role": "assistant", "content": current_response})

    return {  # type: ignore[return-value]
        **state,
        "response_text": current_response,
        "suggestions": suggestions,
        "conversation_history": history,
    }
# Edge routing function

def route_from_intent(state: CoordinatorState) -> str:
    """
    LangGraph conditional edge function.

    Bug fix #1: Reads current_intent["action"] directly instead of
    relying on a separate _route key set by a routing node.
    """
    intent = state.get("current_intent") or {}
    action = intent.get("action", IntentAction.QUERY.value)

    # Check for low confidence (needs clarification)
    confidence = float(intent.get("confidence", 0.0))
    if confidence > 0:
        try:
            _pi = ParsedIntent(
                action=IntentAction(action) if action in IntentAction._value2member_map_ else IntentAction.QUERY,
                target_stage=intent.get("target_stage", ""),
                target_elements=intent.get("target_elements"),
                target_element_types=intent.get("target_element_types"),
                confidence=confidence,
                method_specified=intent.get("method_specified"),
                strength_parameter=intent.get("strength_parameter"),
                natural_language=intent.get("natural_language", ""),
                extracted_constraints=intent.get("extracted_constraints", {}),
                requires_safety_check=intent.get("requires_safety_check", False),
                requires_checkpoint=intent.get("requires_checkpoint", False),
            )
            if needs_clarification(_pi):
                return "build_response"
        except Exception:
            pass

    routing_map = {
        "process":          "handle_process",
        "modify_strategy":  "handle_modification",
        "ignore":           "handle_modification",
        "strengthen":       "handle_modification",
        "query":            "handle_query",
        "approve":          "handle_approve",
        "reject":           "handle_undo",
        "undo":             "handle_undo",
    }
    return routing_map.get(action, "handle_query")


def route_after_pipeline(state: CoordinatorState) -> str:
    """
    After run_pipeline: check if pipeline actually ran or was skipped.
    Always go to check_hitl.
    """
    return "check_hitl"


def route_after_check_hitl(state: CoordinatorState) -> str:
    """
    After check_hitl: always go to build_response.
    The pending_user_decision flag is what the frontend checks.
    """
    return "build_response"
# Graph factory

def build_coordinator_graph(ctx: NodeContext):
    """
    Build and compile the LangGraph StateGraph for the Coordinator Agent.

    Args:
        ctx: NodeContext carrying all shared services.  This is captured
             by node-function closures so the graph remains stateless.

    Returns:
        A compiled LangGraph app (callable with state dict) if LangGraph is
        available, otherwise a _FallbackCoordinator that runs nodes sequentially.
    """
    if not _LANGGRAPH_AVAILABLE:
        logger.warning(
            "LangGraph not installed -- using sequential fallback coordinator."
        )
        return _FallbackCoordinator(ctx)

    # Wrap node functions to inject ctx
    def _wrap(fn):
        def _wrapped(state):
            return fn(state, ctx)
        _wrapped.__name__ = fn.__name__
        return _wrapped

    graph = StateGraph(CoordinatorState)  # type: ignore[arg-type]

    # Add nodes (no separate route_intent node -- routing done via conditional edge)
    graph.add_node("classify_intent",     _wrap(node_classify_intent))
    graph.add_node("handle_process",      _wrap(node_handle_process))
    graph.add_node("handle_modification", _wrap(node_handle_modification))
    graph.add_node("handle_query",        _wrap(node_handle_query))
    graph.add_node("handle_approve",      _wrap(node_handle_approve))
    graph.add_node("handle_undo",         _wrap(node_handle_undo))
    graph.add_node("run_pipeline",        _wrap(node_run_pipeline))
    graph.add_node("check_hitl",          _wrap(node_check_hitl))
    graph.add_node("build_response",      _wrap(node_build_response))

    # Entry point
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_from_intent,
        {
            "handle_process":      "handle_process",
            "handle_modification": "handle_modification",
            "handle_query":        "handle_query",
            "handle_approve":      "handle_approve",
            "handle_undo":         "handle_undo",
            "build_response":      "build_response",
        },
    )

    # handle_process -> run_pipeline
    graph.add_edge("handle_process", "run_pipeline")

    # run_pipeline -> check_hitl
    graph.add_conditional_edges(
        "run_pipeline",
        route_after_pipeline,
        {"check_hitl": "check_hitl"},
    )

    # check_hitl -> build_response
    graph.add_conditional_edges(
        "check_hitl",
        route_after_check_hitl,
        {"build_response": "build_response"},
    )

    # All handlers -> build_response (except handle_process which goes to run_pipeline)
    graph.add_edge("handle_modification", "build_response")
    graph.add_edge("handle_query",        "build_response")
    graph.add_edge("handle_approve",      "build_response")
    graph.add_edge("handle_undo",         "build_response")

    # Terminal
    graph.add_edge("build_response", END)

    try:
        from langgraph.checkpoint.memory import MemorySaver  # type: ignore[import]
        return graph.compile(checkpointer=MemorySaver())
    except ImportError:
        return graph.compile()
# Fallback coordinator (sequential execution when LangGraph not installed)

class _FallbackCoordinator:
    """
    Minimal sequential coordinator used when LangGraph is not installed.

    Runs nodes in linear order sufficient for basic functionality:
    classify -> route -> [handler] -> [run_pipeline -> check_hitl] -> build_response.

    Bug fix #1: Uses route_from_intent() which reads current_intent["action"]
    directly instead of a _route key.
    """

    def __init__(self, ctx: NodeContext) -> None:
        self._ctx = ctx

    def invoke(self, state: CoordinatorState) -> CoordinatorState:
        ctx = self._ctx
        state = node_classify_intent(state, ctx)

        # Route using action directly
        route = route_from_intent(state)

        if route == "handle_process":
            state = node_handle_process(state, ctx)
            state = node_run_pipeline(state, ctx)
            state = node_check_hitl(state, ctx)
        elif route == "handle_modification":
            state = node_handle_modification(state, ctx)
        elif route == "handle_approve":
            state = node_handle_approve(state, ctx)
        elif route == "handle_undo":
            state = node_handle_undo(state, ctx)
        elif route == "handle_query":
            state = node_handle_query(state, ctx)
        else:
            # "build_response" (clarification) or fallback
            pass

        state = node_build_response(state, ctx)
        return state

    # Make it behave like a compiled LangGraph app
    def __call__(self, state: CoordinatorState) -> CoordinatorState:
        return self.invoke(state)
