"""
Coordinator Agent tools — read and write operations on pipeline state.

Read tools (no Safety Kernel gate required):
  query_detections      — Return all detected elements for the current session.
  query_risk_assessments — Return risk assessments with severity / consent.
  query_strategies      — Return recommended obfuscation strategies.
  explain_decision      — Build an explanation prompt for a specific element.

Write tools (Safety Kernel gated):
  apply_strategy_change — Change the obfuscation method for one element.
  apply_ignore          — Set method=none for one element (bystander guard applies).
  apply_strengthen      — Increase protection strength for one element.

All write tools call safety_kernel.validate_override() with full element
context extracted from the current pipeline state.  If the Safety Kernel
returns BLOCK the tool raises a ValueError so the caller can surface the
reason to the user.  CHALLENGE results are surfaced as a warning message
that the coordinator includes in response_text.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
# Canonical value sets (mirrors CLAUDE.md)

VALID_METHODS = frozenset({
    "blur", "pixelate", "solid_overlay", "silhouette", "inpaint",
    "avatar_replace", "generative_replace", "none",
})

# Text types that carry a legal/regulatory protection requirement (GDPR,
# PCI-DSS, HIPAA etc.) and must never have their protection removed.
_LEGAL_TEXT_TYPES = frozenset({"ssn", "credit_card", "password", "personal_number"})


def _compute_legal_requirement(
    element_type: Optional[str],
    severity: Optional[str],
    assessment: Optional[Dict],
) -> bool:
    """
    Return True when the element falls under a legal/regulatory protection
    requirement — i.e. it is a CRITICAL text element whose text_type is one
    of the legally-protected categories (SSN, credit-card, password, …).

    The result is forwarded to safety_kernel.validate_override() as the
    ``legal_requirement`` flag, which triggers an unconditional BLOCK on any
    attempt to weaken or remove protection.
    """
    if (element_type or "").lower() == "text" and (severity or "").lower() == "critical":
        text_type = ((assessment or {}).get("text_type") or "").lower()
        return text_type in _LEGAL_TEXT_TYPES
    return False


# Helpers — extract element context from pipeline state

def _find_assessment(pipeline_state: Dict, detection_id: str) -> Optional[Dict]:
    """
    Locate a risk assessment dict by detection_id.

    Searches in:
      pipeline_state["risk_result"] if it is a dict with "assessments" key
      pipeline_state["risk_result"].assessments if it is a model object
    """
    risk_result = (pipeline_state or {}).get("risk_result")
    if risk_result is None:
        return None

    # Handle both dict and model-object representations
    if isinstance(risk_result, dict):
        assessments = risk_result.get("risk_assessments", risk_result.get("assessments", []))
    elif hasattr(risk_result, "risk_assessments"):
        assessments = risk_result.risk_assessments
    elif hasattr(risk_result, "assessments"):
        assessments = risk_result.assessments
    else:
        return None

    for a in assessments:
        a_dict = a if isinstance(a, dict) else (a.dict() if hasattr(a, "dict") else {})
        if a_dict.get("detection_id") == detection_id:
            return a_dict
    return None


def _find_strategy(pipeline_state: Dict, detection_id: str) -> Optional[Dict]:
    """Locate a strategy dict by detection_id."""
    strategy_result = (pipeline_state or {}).get("strategy_result")
    if strategy_result is None:
        return None

    if isinstance(strategy_result, dict):
        strategies = strategy_result.get("strategies", [])
    elif hasattr(strategy_result, "strategies"):
        strategies = strategy_result.strategies
    else:
        return None

    for s in strategies:
        s_dict = s if isinstance(s, dict) else (s.dict() if hasattr(s, "dict") else {})
        if s_dict.get("detection_id") == detection_id:
            return s_dict
    return None


def _all_assessments(pipeline_state: Dict) -> List[Dict]:
    """Return all risk assessment dicts."""
    risk_result = (pipeline_state or {}).get("risk_result")
    if risk_result is None:
        return []
    if isinstance(risk_result, dict):
        items = risk_result.get("risk_assessments", risk_result.get("assessments", []))
    elif hasattr(risk_result, "risk_assessments"):
        items = risk_result.risk_assessments
    elif hasattr(risk_result, "assessments"):
        items = risk_result.assessments
    else:
        return []
    return [
        a if isinstance(a, dict) else (a.dict() if hasattr(a, "dict") else {})
        for a in items
    ]


def _all_strategies(pipeline_state: Dict) -> List[Dict]:
    """Return all strategy dicts."""
    strategy_result = (pipeline_state or {}).get("strategy_result")
    if strategy_result is None:
        return []
    if isinstance(strategy_result, dict):
        items = strategy_result.get("strategies", [])
    elif hasattr(strategy_result, "strategies"):
        items = strategy_result.strategies
    else:
        return []
    return [
        s if isinstance(s, dict) else (s.dict() if hasattr(s, "dict") else {})
        for s in items
    ]
# READ TOOLS

def query_detections(pipeline_state: Optional[Dict]) -> Dict[str, Any]:
    """
    Return a summary of all detected elements.

    Returns a dict with:
      detections: list of {detection_id, element_type, bbox, confidence}
      total: int
      error: str (if pipeline has not reached detection stage)
    """
    if pipeline_state is None:
        return {"detections": [], "total": 0, "error": "No active pipeline state."}

    detections = pipeline_state.get("detections")
    if detections is None:
        return {
            "detections": [],
            "total": 0,
            "error": "Detection stage has not completed yet.",
        }

    if isinstance(detections, dict):
        items = detections.get("detections", [])
    elif hasattr(detections, "detections"):
        items = detections.detections
    else:
        items = []

    result = []
    for det in items:
        d = det if isinstance(det, dict) else (det.dict() if hasattr(det, "dict") else {})
        result.append({
            "detection_id": d.get("detection_id"),
            "element_type": d.get("element_type"),
            "bbox": d.get("bbox"),
            "confidence": d.get("confidence"),
        })

    return {"detections": result, "total": len(result)}


def query_risk_assessments(pipeline_state: Optional[Dict]) -> Dict[str, Any]:
    """
    Return all risk assessments with severity, consent, and screen state.

    Returns a dict with:
      assessments: list of {detection_id, element_type, severity, risk_type,
                             consent_status, screen_state, reasoning}
      has_critical: bool
      total: int
      error: str (if risk stage not yet complete)
    """
    if pipeline_state is None:
        return {
            "assessments": [], "has_critical": False, "total": 0,
            "error": "No active pipeline state.",
        }

    assessments = _all_assessments(pipeline_state)
    if not assessments and pipeline_state.get("risk_result") is None:
        return {
            "assessments": [], "has_critical": False, "total": 0,
            "error": "Risk assessment stage has not completed yet.",
        }

    summarised = []
    for a in assessments:
        summarised.append({
            "detection_id":   a.get("detection_id"),
            "element_type":   a.get("element_type"),
            "severity":       a.get("severity"),
            "risk_type":      a.get("risk_type"),
            "consent_status": a.get("consent_status"),
            "screen_state":   a.get("screen_state"),
            "reasoning":      a.get("reasoning"),
            "text_type":      a.get("text_type"),
        })

    has_critical = any(
        a.get("severity", "").lower() == "critical" for a in assessments
    )

    return {
        "assessments": summarised,
        "has_critical": has_critical,
        "total": len(summarised),
    }


def query_strategies(pipeline_state: Optional[Dict]) -> Dict[str, Any]:
    """
    Return all recommended obfuscation strategies.

    Returns a dict with:
      strategies: list of {detection_id, element_type, method, parameters,
                            severity, screen_state}
      protected_count: int (method != none)
      skipped_count: int  (method == none)
      error: str (if strategy stage not yet complete)
    """
    if pipeline_state is None:
        return {
            "strategies": [], "protected_count": 0, "skipped_count": 0,
            "error": "No active pipeline state.",
        }

    strategies = _all_strategies(pipeline_state)
    if not strategies and pipeline_state.get("strategy_result") is None:
        return {
            "strategies": [], "protected_count": 0, "skipped_count": 0, "total": 0,
            "error": "Strategy stage has not completed yet.",
        }

    summarised = []
    for s in strategies:
        method = s.get("recommended_method") or s.get("method")
        # Normalise enum values to plain strings
        if hasattr(method, "value"):
            method = method.value
        summarised.append({
            "detection_id": s.get("detection_id"),
            "element_type": s.get("element_type") or s.get("element"),
            "method":       method,
            "parameters":   s.get("parameters", {}),
            "severity":     s.get("severity"),
            "screen_state": s.get("screen_state"),
            "consent_status": s.get("consent_status"),
        })

    protected_count = sum(
        1 for s in strategies
        if (s.get("recommended_method") or s.get("method")) not in {None, "none"}
    )
    skipped_count = len(strategies) - protected_count

    return {
        "strategies": summarised,
        "protected_count": protected_count,
        "skipped_count": skipped_count,
        "total": len(summarised),
    }


def explain_decision(
    pipeline_state: Optional[Dict],
    detection_id: str,
    user_question: str,
) -> Tuple[str, str]:
    """
    Build the (system_prompt, user_message) pair for the VLM explanation prompt.

    Returns a tuple ready for the VLM intent classifier / explanation generator.
    Raises ValueError if the detection_id is not found in the pipeline state.
    """
    if pipeline_state is None:
        raise ValueError("No active pipeline state — cannot explain decision.")

    assessment = _find_assessment(pipeline_state, detection_id)
    if assessment is None:
        raise ValueError(
            f"Element '{detection_id}' not found in risk assessments. "
            "Have you run the risk assessment stage?"
        )

    strategy = _find_strategy(pipeline_state, detection_id) or {}

    # Determine safety floor description
    severity = (assessment.get("severity") or "").lower()
    element_type = (assessment.get("element_type") or "").lower()
    consent_status = (assessment.get("consent_status") or "").lower()

    if element_type == "text" and severity == "critical":
        safety_floor = "Cannot remove protection from CRITICAL text (SSN/CC/password)."
    elif element_type == "face" and consent_status in ("none", "unclear"):
        safety_floor = "Cannot remove protection from unrecognised bystander faces."
    elif element_type == "text" and severity == "high":
        safety_floor = "Cannot remove protection from HIGH-risk text."
    else:
        safety_floor = None

    # Import here to avoid circular import at module scope
    from agents.coordinator.coordinator_prompts import build_explain_prompt

    return build_explain_prompt(
        detection_id=detection_id,
        element_type=assessment.get("element_type", "unknown"),
        risk_assessment=assessment,
        strategy=strategy,
        safety_floor=safety_floor,
        user_question=user_question,
    )
# WRITE TOOLS (Safety Kernel gated)

def _validate_and_apply(
    safety_kernel,
    session_id: str,
    detection_id: str,
    override_type: str,
    original_value: str,
    requested_value: str,
    element_type: str,
    severity: str,
    consent_status: Optional[str] = None,
    screen_state: Optional[str] = None,
    user_confirmed: bool = False,
    assessment: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Call safety_kernel.validate_override() and return a result dict.

    Raises ValueError if the action is BLOCK.
    Returns a dict with 'approved_value', 'safety_action', 'reason', and
    optionally 'challenge_message' if the action was CHALLENGE.

    The ``assessment`` dict is used to compute the ``legal_requirement`` flag
    (True for CRITICAL text elements with legally-protected text_type values
    such as SSN, credit-card, password, or personal_number).
    """
    from backend.services.safety_kernel import SafetyAction  # noqa: PLC0415

    record = safety_kernel.validate_override(
        session_id=session_id,
        detection_id=detection_id,
        override_type=override_type,
        element_type=element_type,
        original_value=original_value,
        requested_value=requested_value,
        severity=severity,
        consent_status=consent_status,
        screen_state=screen_state,
        legal_requirement=_compute_legal_requirement(element_type, severity, assessment),
        user_confirmed=user_confirmed,
    )

    if record.safety_action == SafetyAction.BLOCK:
        raise ValueError(
            f"Safety Kernel BLOCKED override for '{detection_id}': {record.reason} "
            f"[rule={record.rule_id}]"
        )

    result: Dict[str, Any] = {
        "detection_id": detection_id,
        "approved_value": record.approved_value,
        "safety_action": record.safety_action.value,
        "rule_id": record.rule_id,
        "reason": record.reason,
    }

    if record.safety_action == SafetyAction.CHALLENGE:
        result["challenge_message"] = record.reason
        result["requires_confirmation"] = True

    return result


def apply_strategy_change(
    safety_kernel,
    pipeline_state: Dict,
    session_id: str,
    detection_id: str,
    new_method: str,
    user_confirmed: bool = False,
) -> Dict[str, Any]:
    """
    Change the obfuscation method for a single element.

    The Safety Kernel enforces:
      - Cannot set method=none for bystander faces (BLOCK)
      - Cannot set method=none for CRITICAL text (BLOCK)
      - Explicit-consent face → none triggers CHALLENGE (soft guard)
      - Verified-off screen → adding protection triggers CHALLENGE

    Args:
        safety_kernel:  SafetyKernel instance.
        pipeline_state: Current InnerPipelineState dict.
        session_id:     Active session UUID.
        detection_id:   Target element ID.
        new_method:     Requested obfuscation method (must be in VALID_METHODS).
        user_confirmed: True if user has already confirmed a CHALLENGE prompt.

    Returns:
        Dict with approved_value, safety_action, and optional challenge_message.

    Raises:
        ValueError: if new_method is not valid or Safety Kernel returns BLOCK.
    """
    if new_method not in VALID_METHODS:
        raise ValueError(
            f"Unknown method '{new_method}'. Valid methods: {sorted(VALID_METHODS)}"
        )

    assessment = _find_assessment(pipeline_state, detection_id)
    if assessment is None:
        raise ValueError(
            f"No risk assessment found for '{detection_id}'. "
            "Ensure the risk stage has completed before modifying strategies."
        )

    strategy = _find_strategy(pipeline_state, detection_id) or {}
    _raw_method = strategy.get("recommended_method") or strategy.get("method") or "none"
    original_method = _raw_method.value if hasattr(_raw_method, "value") else (_raw_method or "none")

    return _validate_and_apply(
        safety_kernel=safety_kernel,
        session_id=session_id,
        detection_id=detection_id,
        override_type="strategy_method",
        original_value=original_method,
        requested_value=new_method,
        element_type=assessment.get("element_type", "unknown"),
        severity=assessment.get("severity", "medium"),
        consent_status=assessment.get("consent_status"),
        screen_state=assessment.get("screen_state"),
        user_confirmed=user_confirmed,
        assessment=assessment,
    )


def apply_ignore(
    safety_kernel,
    pipeline_state: Dict,
    session_id: str,
    detection_id: str,
    user_confirmed: bool = False,
) -> Dict[str, Any]:
    """
    Set method=none for a single element (remove its protection).

    Delegates to apply_strategy_change with new_method="none".
    Bystander faces are unconditionally blocked by the Safety Kernel.

    Args:
        safety_kernel:  SafetyKernel instance.
        pipeline_state: Current InnerPipelineState dict.
        session_id:     Active session UUID.
        detection_id:   Target element ID.
        user_confirmed: True if user confirmed a preceding CHALLENGE prompt.

    Returns:
        Dict with approved_value, safety_action, and optional challenge_message.

    Raises:
        ValueError: if Safety Kernel returns BLOCK.
    """
    return apply_strategy_change(
        safety_kernel=safety_kernel,
        pipeline_state=pipeline_state,
        session_id=session_id,
        detection_id=detection_id,
        new_method="none",
        user_confirmed=user_confirmed,
    )


def apply_strengthen(
    safety_kernel,
    pipeline_state: Dict,
    session_id: str,
    detection_id: str,
    new_method: Optional[str] = None,
    expand_px: int = 10,
    user_confirmed: bool = False,
) -> Dict[str, Any]:
    """
    Strengthen protection for a single element.

    Two strengthening modes:
      1. Upgrade method: change to a stronger method (e.g., blur → solid_overlay).
         The Safety Kernel's FP2_STRENGTHEN_METHOD fast-path always allows this.
      2. Expand bbox: keep existing method but add expand_px to
         pending_modifications so the execution agent widens the mask.

    If new_method is None, the current method is kept and only expand_px is
    recorded in pending_modifications.

    Args:
        safety_kernel:  SafetyKernel instance.
        pipeline_state: Current InnerPipelineState dict.
        session_id:     Active session UUID.
        detection_id:   Target element ID.
        new_method:     Stronger method to switch to, or None for expand-only.
        expand_px:      Pixels to expand the protection region (default 10).
        user_confirmed: Passed through to Safety Kernel for CHALLENGE rules.

    Returns:
        Dict with approved_value, safety_action, expand_px, reason.

    Raises:
        ValueError: if Safety Kernel returns BLOCK (rare for strengthen actions).
    """
    assessment = _find_assessment(pipeline_state, detection_id)
    if assessment is None:
        raise ValueError(
            f"No risk assessment found for '{detection_id}'."
        )

    strategy = _find_strategy(pipeline_state, detection_id) or {}
    _raw_method = strategy.get("recommended_method") or strategy.get("method") or "none"
    original_method = _raw_method.value if hasattr(_raw_method, "value") else (_raw_method or "none")

    # If no method upgrade is requested, treat as an expand-only strengthen
    if new_method is None:
        new_method = original_method

    if new_method not in VALID_METHODS:
        raise ValueError(
            f"Unknown method '{new_method}'. Valid methods: {sorted(VALID_METHODS)}"
        )

    result = _validate_and_apply(
        safety_kernel=safety_kernel,
        session_id=session_id,
        detection_id=detection_id,
        override_type="strategy_method",
        original_value=original_method,
        requested_value=new_method,
        element_type=assessment.get("element_type", "unknown"),
        severity=assessment.get("severity", "medium"),
        consent_status=assessment.get("consent_status"),
        screen_state=assessment.get("screen_state"),
        user_confirmed=user_confirmed,
        assessment=assessment,
    )

    result["expand_px"] = expand_px
    return result
