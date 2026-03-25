"""
LangGraph node functions for the Coordinator Agent pipeline.

Each node corresponds to one pipeline stage (Phases A-F from CLAUDE.md).
Nodes are thin wrappers that:
  1. Emit a WebSocket stage_start event (if ws_manager is available).
  2. Call the underlying pipeline agent.
  3. Store the result in InnerPipelineState.
  4. Emit a WebSocket stage_complete event with timing.

NodeContext carries all shared services so node functions remain
pure-function-like (they only receive state + context, never globals).

Threading note: these nodes execute inside the LangGraph execution context.
WebSocket events are emitted via ws_manager.broadcast_from_thread() which
uses asyncio.run_coroutine_threadsafe() to cross the thread boundary safely.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agents.coordinator.state import InnerPipelineState

logger = logging.getLogger(__name__)
# NodeContext — shared services injected into every node

@dataclass
class NodeContext:
    """
    Carries all shared services and agents needed by coordinator nodes.

    Passed as the second argument to every node function.  This keeps node
    signatures stable even as new services are added.
    """
    # Pipeline agents (may be None in fallback-only mode or during tests)
    detection_agent: Any = None
    risk_agent: Any = None
    consent_agent: Any = None
    strategy_agent: Any = None
    execution_agent: Any = None

    # Infrastructure services
    ws_manager: Any = None                   # WebSocketManager or None
    session_manager: Any = None              # SessionManager or None
    safety_kernel: Any = None                # SafetyKernel or None
    pipeline_service: Any = None             # PipelineService or None

    # Configuration
    output_dir: str = "data/full_pipeline_results"
    fallback_only: bool = False
# WS event helpers

def _emit_stage_start(
    ctx: NodeContext,
    session_id: str,
    stage: str,
    display_name: str,
    description: str = "",
) -> None:
    """Broadcast a stage_start WebSocket event (best-effort, never raises)."""
    if ctx.ws_manager is None:
        return
    try:
        event = {
            "type": "stage_start",
            "session_id": session_id,
            "stage": stage,
            "display_name": display_name,
            "description": description,
        }
        ctx.ws_manager.broadcast_from_thread(session_id, event)
    except Exception as exc:
        logger.debug("stage_start WS emit failed (%s): %s", stage, exc)


def _emit_stage_complete(
    ctx: NodeContext,
    session_id: str,
    stage: str,
    elapsed_ms: float,
    extra: Optional[Dict] = None,
) -> None:
    """Broadcast a stage_complete WebSocket event (best-effort, never raises)."""
    if ctx.ws_manager is None:
        return
    try:
        event: Dict[str, Any] = {
            "type": "stage_complete",
            "session_id": session_id,
            "stage": stage,
            "elapsed_ms": round(elapsed_ms, 1),
        }
        if extra:
            event.update(extra)
        ctx.ws_manager.broadcast_from_thread(session_id, event)
    except Exception as exc:
        logger.debug("stage_complete WS emit failed (%s): %s", stage, exc)


def _emit_error(
    ctx: NodeContext,
    session_id: str,
    stage: str,
    error_message: str,
) -> None:
    """Broadcast a stage_error WebSocket event (best-effort, never raises)."""
    if ctx.ws_manager is None:
        return
    try:
        event = {
            "type": "stage_error",
            "session_id": session_id,
            "stage": stage,
            "error": error_message,
        }
        ctx.ws_manager.broadcast_from_thread(session_id, event)
    except Exception as exc:
        logger.debug("stage_error WS emit failed (%s): %s", stage, exc)
# Node 1: Detection

def node_detection(state: InnerPipelineState, ctx: NodeContext) -> InnerPipelineState:
    """
    Phase A — Agent 1: Face / text / object detection.

    Calls DetectionAgent.run(image_path) and stores the result in
    state["detections"].  Elapsed time is recorded in stage_timings.
    """
    session_id = state.get("session_id", "unknown")
    image_path = state.get("image_path", "")

    _emit_stage_start(
        ctx, session_id, "detection", "Detection",
        "Detecting faces, text, and objects (MTCNN + EasyOCR + YOLOv8)",
    )

    t0 = time.time()
    errors = dict(state.get("errors") or {})
    timings = dict(state.get("stage_timings") or {})

    try:
        if ctx.detection_agent is None:
            raise RuntimeError("DetectionAgent not initialised in NodeContext.")

        from PIL import Image as PILImage  # noqa: PLC0415
        image = PILImage.open(image_path).convert("RGB")
        detections = ctx.detection_agent.run(image, image_path)

        elapsed_ms = (time.time() - t0) * 1000
        timings["detection"] = round(elapsed_ms, 1)

        n_elements = 0
        if hasattr(detections, "detections"):
            n_elements = len(detections.detections)
        elif isinstance(detections, dict):
            n_elements = len(detections.get("detections", []))

        _emit_stage_complete(
            ctx, session_id, "detection", elapsed_ms,
            {"elements_detected": n_elements},
        )

        return {  # type: ignore[return-value]
            **state,
            "detections": detections,
            "_cached_image": image,
            "stage_timings": timings,
            "errors": errors,
        }

    except Exception as exc:
        elapsed_ms = (time.time() - t0) * 1000
        timings["detection"] = round(elapsed_ms, 1)
        errors["detection"] = str(exc)
        logger.error("Detection node failed for session %s: %s", session_id, exc)
        _emit_error(ctx, session_id, "detection", str(exc))
        return {  # type: ignore[return-value]
            **state,
            "stage_timings": timings,
            "errors": errors,
        }
# Node 2: Risk Assessment

def node_risk(state: InnerPipelineState, ctx: NodeContext) -> InnerPipelineState:
    """
    Phase B — Agent 2: Two-phase risk assessment.

    Calls RiskAssessmentAgent.run(detections, image, fallback_only).
    Stores result in state["risk_result"].
    """
    session_id = state.get("session_id", "unknown")
    image_path = state.get("image_path", "")
    fallback_only = state.get("fallback_only", ctx.fallback_only)

    _emit_stage_start(
        ctx, session_id, "risk", "Risk Assessment",
        "Assessing privacy risk severity for each detected element",
    )

    t0 = time.time()
    errors = dict(state.get("errors") or {})
    timings = dict(state.get("stage_timings") or {})

    try:
        if ctx.risk_agent is None:
            raise RuntimeError("RiskAssessmentAgent not initialised in NodeContext.")

        detections = state.get("detections")
        if detections is None:
            raise ValueError("Detection results not available — run detection first.")

        from PIL import Image as PILImage  # noqa: PLC0415
        image = state.get("_cached_image") or PILImage.open(image_path).convert("RGB")

        risk_result = ctx.risk_agent.run(
            detections=detections,
            image=image,
            fallback_only=fallback_only,
        )

        elapsed_ms = (time.time() - t0) * 1000
        timings["risk"] = round(elapsed_ms, 1)

        _emit_stage_complete(ctx, session_id, "risk", elapsed_ms)

        return {  # type: ignore[return-value]
            **state,
            "risk_result": risk_result,
            "stage_timings": timings,
            "errors": errors,
        }

    except Exception as exc:
        elapsed_ms = (time.time() - t0) * 1000
        timings["risk"] = round(elapsed_ms, 1)
        errors["risk"] = str(exc)
        logger.error("Risk node failed for session %s: %s", session_id, exc)
        _emit_error(ctx, session_id, "risk", str(exc))
        return {  # type: ignore[return-value]
            **state,
            "stage_timings": timings,
            "errors": errors,
        }
# Node 3: Consent Identity

def node_consent(state: InnerPipelineState, ctx: NodeContext) -> InnerPipelineState:
    """
    Phase C — Agent 2.5: Face matching and consent resolution.

    Calls ConsentIdentityAgent.run(detections, risk_result).
    Stores enriched assessments in state["identity_assessments"].
    No-ops gracefully if consent_agent is None (MongoDB unavailable).
    """
    session_id = state.get("session_id", "unknown")

    _emit_stage_start(
        ctx, session_id, "consent", "Consent Identity",
        "Matching faces against consent database",
    )

    t0 = time.time()
    errors = dict(state.get("errors") or {})
    timings = dict(state.get("stage_timings") or {})

    try:
        if ctx.consent_agent is None:
            logger.info(
                "ConsentIdentityAgent not available — skipping consent stage "
                "(session=%s)", session_id,
            )
            elapsed_ms = (time.time() - t0) * 1000
            timings["consent"] = round(elapsed_ms, 1)
            _emit_stage_complete(ctx, session_id, "consent", elapsed_ms, {"skipped": True})
            return {  # type: ignore[return-value]
                **state,
                "stage_timings": timings,
                "errors": errors,
            }

        detections = state.get("detections")
        risk_result = state.get("risk_result")

        if detections is None or risk_result is None:
            raise ValueError(
                "Both detections and risk_result must be available before consent stage."
            )

        identity_result = ctx.consent_agent.run(
            detections=detections,
            risk_result=risk_result,
        )

        elapsed_ms = (time.time() - t0) * 1000
        timings["consent"] = round(elapsed_ms, 1)

        # identity_result may be the updated risk_result or a separate object
        # Accept both patterns used in the codebase
        if isinstance(identity_result, dict) and "assessments" in identity_result:
            updated_risk = identity_result
        elif hasattr(identity_result, "assessments"):
            updated_risk = identity_result
        else:
            updated_risk = risk_result  # No change

        _emit_stage_complete(ctx, session_id, "consent", elapsed_ms)

        return {  # type: ignore[return-value]
            **state,
            "risk_result": updated_risk,
            "stage_timings": timings,
            "errors": errors,
        }

    except Exception as exc:
        elapsed_ms = (time.time() - t0) * 1000
        timings["consent"] = round(elapsed_ms, 1)
        errors["consent"] = str(exc)
        logger.error("Consent node failed for session %s: %s", session_id, exc)
        _emit_error(ctx, session_id, "consent", str(exc))
        return {  # type: ignore[return-value]
            **state,
            "stage_timings": timings,
            "errors": errors,
        }
# Node 4: Strategy

def node_strategy(state: InnerPipelineState, ctx: NodeContext) -> InnerPipelineState:
    """
    Phase D — Agent 3: Obfuscation strategy recommendation.

    Calls StrategyAgent.run(risk_result, detections, fallback_only).
    Stores result in state["strategy_result"].
    """
    session_id = state.get("session_id", "unknown")
    image_path = state.get("image_path", "")
    fallback_only = state.get("fallback_only", ctx.fallback_only)

    _emit_stage_start(
        ctx, session_id, "strategy", "Strategy",
        "Recommending obfuscation strategies per element",
    )

    t0 = time.time()
    errors = dict(state.get("errors") or {})
    timings = dict(state.get("stage_timings") or {})

    try:
        if ctx.strategy_agent is None:
            raise RuntimeError("StrategyAgent not initialised in NodeContext.")

        risk_result = state.get("risk_result")
        detections = state.get("detections")

        if risk_result is None:
            raise ValueError("Risk assessment results required before strategy stage.")

        from PIL import Image as PILImage  # noqa: PLC0415
        image = state.get("_cached_image") or PILImage.open(image_path).convert("RGB")

        strategy_result = ctx.strategy_agent.run(
            risk_result=risk_result,
            detections=detections,
            image=image,
            fallback_only=fallback_only,
        )

        elapsed_ms = (time.time() - t0) * 1000
        timings["strategy"] = round(elapsed_ms, 1)

        _emit_stage_complete(ctx, session_id, "strategy", elapsed_ms)

        return {  # type: ignore[return-value]
            **state,
            "strategy_result": strategy_result,
            "stage_timings": timings,
            "errors": errors,
        }

    except Exception as exc:
        elapsed_ms = (time.time() - t0) * 1000
        timings["strategy"] = round(elapsed_ms, 1)
        errors["strategy"] = str(exc)
        logger.error("Strategy node failed for session %s: %s", session_id, exc)
        _emit_error(ctx, session_id, "strategy", str(exc))
        return {  # type: ignore[return-value]
            **state,
            "stage_timings": timings,
            "errors": errors,
        }
# Node 5: SAM Segmentation

def node_sam(state: InnerPipelineState, ctx: NodeContext) -> InnerPipelineState:
    """
    Phase D.5 — MobileSAM segmentation (selective, only elements with method != none).

    Uses PrecisionSegmenter from utils/segmentation.py.
    Stores mask paths/arrays in state["seg_results"] keyed by detection_id.
    Skips gracefully if segmentation is unavailable or strategy_result is absent.
    """
    session_id = state.get("session_id", "unknown")
    image_path = state.get("image_path", "")

    _emit_stage_start(
        ctx, session_id, "sam", "SAM Segmentation",
        "Generating precise pixel masks (MobileSAM)",
    )

    t0 = time.time()
    errors = dict(state.get("errors") or {})
    timings = dict(state.get("stage_timings") or {})

    try:
        strategy_result = state.get("strategy_result")
        if strategy_result is None:
            logger.info(
                "No strategy_result available — skipping SAM (session=%s)", session_id
            )
            elapsed_ms = (time.time() - t0) * 1000
            timings["sam"] = round(elapsed_ms, 1)
            _emit_stage_complete(ctx, session_id, "sam", elapsed_ms, {"skipped": True})
            return {  # type: ignore[return-value]
                **state,
                "seg_results": {},
                "stage_timings": timings,
                "errors": errors,
            }

        # Attempt to import segmentation module (optional dep)
        try:
            from utils.segmentation import PrecisionSegmenter  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "MobileSAM not available — skipping SAM segmentation (session=%s)",
                session_id,
            )
            elapsed_ms = (time.time() - t0) * 1000
            timings["sam"] = round(elapsed_ms, 1)
            _emit_stage_complete(ctx, session_id, "sam", elapsed_ms, {"skipped": True})
            return {  # type: ignore[return-value]
                **state,
                "seg_results": {},
                "stage_timings": timings,
                "errors": errors,
            }

        from PIL import Image as PILImage  # noqa: PLC0415
        image = state.get("_cached_image") or PILImage.open(image_path).convert("RGB")

        segmenter = PrecisionSegmenter()
        seg_results = segmenter.run(
            image=image,
            strategy_result=strategy_result,
            output_dir=ctx.output_dir,
        )

        elapsed_ms = (time.time() - t0) * 1000
        timings["sam"] = round(elapsed_ms, 1)

        n_masks = len(seg_results) if seg_results else 0
        _emit_stage_complete(
            ctx, session_id, "sam", elapsed_ms, {"masks_generated": n_masks}
        )

        return {  # type: ignore[return-value]
            **state,
            "seg_results": seg_results or {},
            "stage_timings": timings,
            "errors": errors,
        }

    except Exception as exc:
        elapsed_ms = (time.time() - t0) * 1000
        timings["sam"] = round(elapsed_ms, 1)
        errors["sam"] = str(exc)
        logger.error("SAM node failed for session %s: %s", session_id, exc)
        _emit_error(ctx, session_id, "sam", str(exc))
        return {  # type: ignore[return-value]
            **state,
            "seg_results": {},
            "stage_timings": timings,
            "errors": errors,
        }
# Node 6: Execution

def node_execution(state: InnerPipelineState, ctx: NodeContext) -> InnerPipelineState:
    """
    Phase E — Agent 4: Two-phase execution (apply + VLM verify).

    Calls ExecutionAgent.run(image, strategy_result, seg_results, fallback_only).
    Stores protected image path + execution report.
    """
    session_id = state.get("session_id", "unknown")
    image_path = state.get("image_path", "")
    fallback_only = state.get("fallback_only", ctx.fallback_only)

    _emit_stage_start(
        ctx, session_id, "execution", "Execution",
        "Applying obfuscation to the image",
    )

    t0 = time.time()
    errors = dict(state.get("errors") or {})
    timings = dict(state.get("stage_timings") or {})

    try:
        if ctx.execution_agent is None:
            raise RuntimeError("ExecutionAgent not initialised in NodeContext.")

        strategy_result = state.get("strategy_result")
        seg_results = state.get("seg_results") or {}

        if strategy_result is None:
            raise ValueError("Strategy results required before execution stage.")

        from PIL import Image as PILImage  # noqa: PLC0415
        image = state.get("_cached_image") or PILImage.open(image_path).convert("RGB")

        execution_result = ctx.execution_agent.run(
            image=image,
            image_path=image_path,
            strategy_result=strategy_result,
            seg_results=seg_results,
            fallback_only=fallback_only,
            output_dir=ctx.output_dir,
        )

        elapsed_ms = (time.time() - t0) * 1000
        timings["execution"] = round(elapsed_ms, 1)

        # Extract protected_image_path from execution result
        protected_path = None
        if isinstance(execution_result, dict):
            protected_path = execution_result.get("protected_image_path")
        elif hasattr(execution_result, "protected_image_path"):
            protected_path = execution_result.protected_image_path

        _emit_stage_complete(ctx, session_id, "execution", elapsed_ms)

        return {  # type: ignore[return-value]
            **state,
            "execution_report": execution_result,
            "protected_image_path": protected_path,
            "stage_timings": timings,
            "errors": errors,
        }

    except Exception as exc:
        elapsed_ms = (time.time() - t0) * 1000
        timings["execution"] = round(elapsed_ms, 1)
        errors["execution"] = str(exc)
        logger.error("Execution node failed for session %s: %s", session_id, exc)
        _emit_error(ctx, session_id, "execution", str(exc))
        return {  # type: ignore[return-value]
            **state,
            "stage_timings": timings,
            "errors": errors,
        }
# Node 7: Export

def node_export(state: InnerPipelineState, ctx: NodeContext) -> InnerPipelineState:
    """
    Phase F — Export: JSON results, risk map, and strategy JSON.

    Uses utils/visualization.py to write JSON and PNG outputs.
    Stores paths in state["risk_map_path"] and state["strategy_json_path"].
    """
    session_id = state.get("session_id", "unknown")
    image_path = state.get("image_path", "")

    _emit_stage_start(
        ctx, session_id, "export", "Export",
        "Exporting risk map, JSON results, and protected image",
    )

    t0 = time.time()
    errors = dict(state.get("errors") or {})
    timings = dict(state.get("stage_timings") or {})

    try:
        from PIL import Image as PILImage  # noqa: PLC0415
        from utils.visualization import (  # noqa: PLC0415
            export_risk_results_json,
            generate_risk_map,
            export_strategy_results_json,
        )
        import os  # noqa: PLC0415

        # Determine sample name from image path
        sample_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = ctx.output_dir

        image = state.get("_cached_image") or PILImage.open(image_path).convert("RGB")
        risk_result = state.get("risk_result")
        strategy_result = state.get("strategy_result")

        risk_map_path = None
        strategy_json_path = None

        if risk_result is not None:
            try:
                export_risk_results_json(
                    risk_result=risk_result,
                    output_path=os.path.join(output_dir, f"{sample_name}_risk_results.json"),
                )
                risk_map_path = os.path.join(output_dir, f"{sample_name}_risk_map.png")
                generate_risk_map(
                    image=image,
                    risk_result=risk_result,
                    output_path=risk_map_path,
                )
            except Exception as export_exc:
                logger.warning("Risk export failed: %s", export_exc)
                errors["export_risk"] = str(export_exc)

        if strategy_result is not None:
            try:
                strategy_json_path = os.path.join(
                    output_dir, f"{sample_name}_strategies.json"
                )
                export_strategy_results_json(
                    strategy_result=strategy_result,
                    output_path=strategy_json_path,
                )
            except Exception as strat_exc:
                logger.warning("Strategy export failed: %s", strat_exc)
                errors["export_strategy"] = str(strat_exc)

        elapsed_ms = (time.time() - t0) * 1000
        timings["export"] = round(elapsed_ms, 1)

        _emit_stage_complete(ctx, session_id, "export", elapsed_ms)

        return {  # type: ignore[return-value]
            **state,
            "risk_map_path": risk_map_path,
            "strategy_json_path": strategy_json_path,
            "stage_timings": timings,
            "errors": errors,
        }

    except Exception as exc:
        elapsed_ms = (time.time() - t0) * 1000
        timings["export"] = round(elapsed_ms, 1)
        errors["export"] = str(exc)
        logger.error("Export node failed for session %s: %s", session_id, exc)
        _emit_error(ctx, session_id, "export", str(exc))
        return {  # type: ignore[return-value]
            **state,
            "stage_timings": timings,
            "errors": errors,
        }
