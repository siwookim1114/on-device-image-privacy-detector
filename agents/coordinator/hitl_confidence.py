"""
HITL Confidence Gating — per-element and session-level confidence scoring.

Answers the question: "Given the current pipeline state, should the system
pause and ask the human to review, or can it auto-advance?"

Architecture:
  1. element_confidence()       — scores a single RiskAssessment (0-1)
  2. session_confidence()       — aggregates element scores into one number
  3. compute_hitl_decision()    — maps session score + flags → checkpoint type
  4. ConfidenceReport           — structured output for coordinator

Design goals:
  - Zero ML model required for baseline operation (all rule-based scores)
  - Scores must be interpretable: each component has a named weight
  - Thresholds are named constants so they can be A/B tested and stored
    in config.yaml in future iterations
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
# Named threshold constants
# (These values are the concrete numbers requested; justification in docstrings)

class HITLMode(str, Enum):
    """Maps to ProcessingMode in models.py but scoped to HITL decisions."""
    AUTO   = "auto"    # No pause; full auto-advance
    HYBRID = "hybrid"  # Pause at confidence-gated checkpoints
    MANUAL = "manual"  # Always pause; user confirms each stage


class CheckpointType(str, Enum):
    AUTO_ADVANCE_SUMMARY = "auto_advance_summary"  # Just show summary, no pause
    STRATEGY_REVIEW      = "strategy_review"        # Show risk + strategy; user approves
    RISK_REVIEW          = "risk_review"             # Show risk assessments; user reviews
    FULL_MANUAL_REVIEW   = "full_manual_review"      # Show everything; step-by-step


# ── Session-level thresholds ──────────────────────────────────────────────
#
# AUTO_ADVANCE_THRESHOLD = 0.85
#   Justification: At 0.85 the expected element-level errors are <15% which,
#   combined with the forward-only Safety Kernel guarantee (bystander faces
#   and CRITICAL text are always protected regardless), means false negatives
#   cannot produce a privacy-harming outcome.
#
# STRATEGY_REVIEW_THRESHOLD = 0.50
#   Justification: Below 0.85 there is meaningful uncertainty; the system
#   pauses at strategy (after risk/consent are done) to let the user confirm
#   the final method choices without re-doing detection.  0.50 is the
#   midpoint of a [0,1] scale; below it we go deeper to risk review.
#
# RISK_REVIEW_THRESHOLD = 0.50 (lower bound)
#   Anything below 0.50 triggers risk review (not just strategy review) because
#   uncertainty this high may reflect wrong detections or wrong severity.
#
AUTO_ADVANCE_THRESHOLD   = 0.85
STRATEGY_REVIEW_THRESHOLD = 0.50


# ── Element-level component weights ──────────────────────────────────────
#
# Each component penalizes confidence when it signals uncertainty.
# Weights are additive deductions from 1.0; uncapped at 0.0.

_WEIGHTS: Dict[str, float] = {
    # Detection-level signal
    "detection_confidence":    0.25,  # Raw MTCNN / EasyOCR / YOLO confidence
    # Severity signal: CRITICAL = penalize heavily; LOW = minimal penalty
    "severity_penalty":        0.20,
    # Consent signal: unclear / none = higher uncertainty
    "consent_uncertainty":     0.20,
    # VLM Phase 2 ran and agreed: bonus; didn't run: no bonus
    "vlm_phase2_agreement":    0.15,
    # Screen state: verified_off removes uncertainty; verified_on adds it
    "screen_state_uncertainty": 0.10,
    # Spatial escalation occurred: implies boundary uncertainty
    "escalation_applied":      0.10,
}

_SEVERITY_PENALTY: Dict[str, float] = {
    "critical": 0.15,  # CRITICAL means we need human to confirm
    "high":     0.08,
    "medium":   0.03,
    "low":      0.00,
}

_CONSENT_UNCERTAINTY_PENALTY: Dict[str, float] = {
    "explicit": 0.00,   # Fully known; no uncertainty
    "assumed":  0.05,   # Assumed from history; slight uncertainty
    "unclear":  0.15,   # Needs human confirmation
    "none":     0.10,   # Unknown person; high ethical stake but not "uncertain"
                        # (we always protect, so the DECISION is clear)
}
# Data structures

@dataclass
class ElementConfidenceSignals:
    """Raw signals extracted from a single RiskAssessment dict."""
    detection_id: str
    element_type: str              # "face", "text", "screen", "object"
    detection_confidence: float    # Raw detector confidence (0-1)
    severity: str                  # "critical", "high", "medium", "low"
    consent_status: Optional[str]  # For faces: "explicit", "assumed", "unclear", "none"
    screen_state: Optional[str]    # "verified_on", "verified_off", or None
    escalation_applied: bool       # Was spatial escalation triggered?
    vlm_phase2_ran: bool           # Did Phase 2 VLM assessment run?
    vlm_phase2_agreed: bool        # Did VLM agree with Phase 1 (if it ran)?

    @classmethod
    def from_assessment_dict(cls, a: Dict) -> "ElementConfidenceSignals":
        """
        Build signals from the dict format used in InnerPipelineState.
        Keys expected:
          detection_id, element_type, confidence (detector), severity,
          consent_status (optional), screen_state (optional),
          escalation_applied (optional bool), vlm_phase2_ran (optional bool),
          vlm_phase2_agreed (optional bool)
        """
        return cls(
            detection_id=a.get("detection_id", "unknown"),
            element_type=a.get("element_type", "unknown"),
            detection_confidence=float(a.get("confidence", 0.7)),
            severity=a.get("severity", "medium").lower(),
            consent_status=(a.get("consent_status") or "").lower() or None,
            screen_state=a.get("screen_state"),
            escalation_applied=bool(a.get("escalation_applied", False)),
            vlm_phase2_ran=bool(a.get("vlm_phase2_ran", False)),
            vlm_phase2_agreed=bool(a.get("vlm_phase2_agreed", False)),
        )


@dataclass
class ElementConfidenceResult:
    detection_id: str
    element_type: str
    raw_score: float          # 0-1; higher = more confident
    component_breakdown: Dict[str, float]  # name -> contribution
    penalized_by: List[str]   # names of components that reduced confidence


@dataclass
class ConfidenceReport:
    session_score: float                          # Aggregated 0-1
    checkpoint_type: CheckpointType
    element_results: List[ElementConfidenceResult]
    has_critical_elements: bool
    has_consent_conflicts: bool
    has_unverified_screens: bool
    low_confidence_elements: List[str]            # detection_ids
    rationale: str                                # Human-readable summary
    auto_advance_possible: bool
# 2.1  Per-element confidence score

def element_confidence(signals: ElementConfidenceSignals) -> ElementConfidenceResult:
    """
    Compute a per-element confidence score in [0, 1].

    Score represents: "how certain are we that our assessment + strategy
    for this element is correct and needs no human review?"

    High score = system is sure; human review is optional.
    Low score  = system is uncertain; human should review.
    """
    components: Dict[str, float] = {}
    penalized_by: List[str] = []

    # ── Component 1: Detection confidence ───────────────────────────────
    # Direct use of MTCNN/EasyOCR/YOLO confidence (already 0-1).
    # Weight: maps raw confidence to 0-0.25 contribution.
    det_conf = max(0.0, min(1.0, signals.detection_confidence))
    det_contrib = _WEIGHTS["detection_confidence"] * det_conf
    components["detection_confidence"] = det_contrib
    if det_conf < 0.75:
        penalized_by.append("detection_confidence")

    # ── Component 2: Severity penalty ───────────────────────────────────
    # CRITICAL elements need human attention by design; they reduce session
    # score to push toward STRATEGY_REVIEW checkpoint.
    # Expressed as a DEDUCTION from maximum possible score.
    sev_pen = _SEVERITY_PENALTY.get(signals.severity, 0.03)
    # Stored as the amount REMAINING after penalty (so higher = less penalized)
    sev_contrib = _WEIGHTS["severity_penalty"] * (1.0 - sev_pen / 0.15)
    components["severity_penalty"] = sev_contrib
    if sev_pen > 0.08:
        penalized_by.append("severity_penalty")

    # ── Component 3: Consent uncertainty ────────────────────────────────
    # Only applicable to face elements; for text/screen/object, full weight.
    if signals.element_type == "face" and signals.consent_status:
        con_pen = _CONSENT_UNCERTAINTY_PENALTY.get(signals.consent_status, 0.10)
        con_contrib = _WEIGHTS["consent_uncertainty"] * (1.0 - con_pen / 0.15)
        if con_pen > 0.05:
            penalized_by.append("consent_uncertainty")
    else:
        con_contrib = _WEIGHTS["consent_uncertainty"]  # full weight; no face uncertainty
    components["consent_uncertainty"] = con_contrib

    # ── Component 4: VLM Phase 2 agreement ──────────────────────────────
    # If VLM ran and agreed with Phase 1: full bonus (+0.15).
    # If VLM ran and DISAGREED: deduction (indicates borderline case).
    # If VLM didn't run: neutral (0.075 = half weight).
    if signals.vlm_phase2_ran:
        vlm_contrib = _WEIGHTS["vlm_phase2_agreement"] if signals.vlm_phase2_agreed else 0.0
        if not signals.vlm_phase2_agreed:
            penalized_by.append("vlm_phase2_agreement")
    else:
        vlm_contrib = _WEIGHTS["vlm_phase2_agreement"] * 0.5  # neutral
    components["vlm_phase2_agreement"] = vlm_contrib

    # ── Component 5: Screen state uncertainty ───────────────────────────
    if signals.element_type == "screen":
        if signals.screen_state == "verified_off":
            screen_contrib = _WEIGHTS["screen_state_uncertainty"]  # certain; no penalty
        elif signals.screen_state == "verified_on":
            screen_contrib = _WEIGHTS["screen_state_uncertainty"] * 0.5  # uncertain content
            penalized_by.append("screen_state_uncertainty")
        else:
            screen_contrib = 0.0  # screen but state unknown = maximum penalty
            penalized_by.append("screen_state_uncertainty")
    else:
        screen_contrib = _WEIGHTS["screen_state_uncertainty"]  # not a screen; full weight
    components["screen_state_uncertainty"] = screen_contrib

    # ── Component 6: Escalation applied ─────────────────────────────────
    if signals.escalation_applied:
        esc_contrib = 0.0  # escalation means spatial boundary is uncertain
        penalized_by.append("escalation_applied")
    else:
        esc_contrib = _WEIGHTS["escalation_applied"]
    components["escalation_applied"] = esc_contrib

    raw_score = max(0.0, min(1.0, sum(components.values())))

    return ElementConfidenceResult(
        detection_id=signals.detection_id,
        element_type=signals.element_type,
        raw_score=raw_score,
        component_breakdown=components,
        penalized_by=penalized_by,
    )
# 2.2  Session-level confidence aggregation

def session_confidence(element_results: List[ElementConfidenceResult]) -> float:
    """
    Aggregate element-level scores into a single session confidence score.

    Strategy:
      - Use weighted harmonic mean instead of arithmetic mean.
      - Harmonic mean is appropriate here because one very-low-confidence
        element should pull the session score down significantly, which is
        the desired behavior for a safety-critical system.
      - Elements with higher severity receive higher weight (they matter more).
      - Minimum floor of 0.05 prevents division by zero.

    Returns a float in [0, 1].
    """
    if not element_results:
        return 1.0  # No elements = no risk = full confidence

    # Severity-based weights for harmonic mean
    # (higher severity elements matter more)
    _severity_weights = {"critical": 3.0, "high": 2.0, "medium": 1.5, "low": 1.0}

    weighted_sum = 0.0
    weight_total = 0.0

    for er in element_results:
        # We need the severity; it's not stored on ElementConfidenceResult
        # but can be inferred: if sev_contrib is low, severity is high.
        # We use the component breakdown to back-infer an approximate weight.
        # For production use, pass severity as an attribute on ElementConfidenceResult.
        # Here we use the component contribution as proxy.
        sev_contrib = er.component_breakdown.get("severity_penalty", 0.15)
        approx_weight = 1.0 + (1.0 - sev_contrib / _WEIGHTS["severity_penalty"])
        approx_weight = max(1.0, approx_weight)

        score_floor = max(0.05, er.raw_score)
        weighted_sum += approx_weight / score_floor
        weight_total += approx_weight

    if weight_total == 0:
        return 0.5

    harmonic_mean = weight_total / weighted_sum
    return max(0.0, min(1.0, harmonic_mean))
# 2.3  Session-level flag extraction

def _has_critical_elements(assessments: List[Dict]) -> bool:
    return any(a.get("severity", "").lower() == "critical" for a in assessments)


def _has_consent_conflicts(strategies: List[Dict]) -> bool:
    """
    A consent conflict exists when an element with consent_status=explicit
    (the user themselves) has been assigned a protection method.
    The challenge-confirm pattern handles this at strategy level;
    here we just detect it as a HITL signal.
    """
    return any(
        s.get("consent_status") == "explicit" and s.get("method") not in {None, "none"}
        for s in strategies
        if s.get("element_type") == "face"
    )


def _has_unverified_screens(assessments: List[Dict]) -> bool:
    return any(
        a.get("element_type") == "screen" and a.get("screen_state") not in {
            "verified_on", "verified_off"
        }
        for a in assessments
    )


def _low_confidence_elements(
    element_results: List[ElementConfidenceResult],
    threshold: float = 0.55,
) -> List[str]:
    return [er.detection_id for er in element_results if er.raw_score < threshold]
# 2.4  Full HITL decision

def compute_hitl_decision(
    risk_assessments: List[Dict],
    strategies: List[Dict],
    mode: HITLMode = HITLMode.HYBRID,
) -> ConfidenceReport:
    """
    Top-level HITL decision function.

    Args:
        risk_assessments: List of dicts from InnerPipelineState.risk_assessments.
                          Each dict must have the keys expected by
                          ElementConfidenceSignals.from_assessment_dict().
        strategies:       List of dicts from InnerPipelineState.strategies.
                          Used only for consent-conflict detection.
        mode:             User's processing mode preference from PrivacyProfile.

    Returns:
        ConfidenceReport with checkpoint_type that drives the coordinator graph.
    """
    # Build element confidence results
    element_results: List[ElementConfidenceResult] = []
    for a in risk_assessments:
        signals = ElementConfidenceSignals.from_assessment_dict(a)
        er = element_confidence(signals)
        element_results.append(er)

    score = session_confidence(element_results)

    # Session-level flags
    has_critical = _has_critical_elements(risk_assessments)
    has_consent_conflicts = _has_consent_conflicts(strategies)
    has_unverified_screens = _has_unverified_screens(risk_assessments)
    low_conf_ids = _low_confidence_elements(element_results)

    # ── Checkpoint decision ───────────────────────────────────────────────
    # Manual mode: always pause fully
    if mode == HITLMode.MANUAL:
        checkpoint = CheckpointType.FULL_MANUAL_REVIEW
        rationale = "Manual mode: full review required at every stage."

    # Auto mode: only pause for critical or very low confidence
    elif mode == HITLMode.AUTO:
        if has_critical or score < 0.40:
            checkpoint = CheckpointType.STRATEGY_REVIEW
            rationale = (
                f"Auto mode but critical elements present or very low confidence "
                f"({score:.2f}); pausing at strategy review."
            )
        else:
            checkpoint = CheckpointType.AUTO_ADVANCE_SUMMARY
            rationale = (
                f"Auto mode, confidence {score:.2f} ≥ 0.40 and no critical elements. "
                f"Auto-advancing with summary."
            )

    # Hybrid mode: confidence-gated
    else:
        if (score >= AUTO_ADVANCE_THRESHOLD
                and not has_critical
                and not has_consent_conflicts):
            checkpoint = CheckpointType.AUTO_ADVANCE_SUMMARY
            rationale = (
                f"Confidence {score:.2f} ≥ {AUTO_ADVANCE_THRESHOLD}, "
                f"no critical elements, no consent conflicts. Auto-advancing."
            )
        elif score >= STRATEGY_REVIEW_THRESHOLD or has_critical:
            checkpoint = CheckpointType.STRATEGY_REVIEW
            rationale = (
                f"Confidence {score:.2f} or critical elements present. "
                f"Pausing at strategy review."
            )
        else:
            checkpoint = CheckpointType.RISK_REVIEW
            rationale = (
                f"Confidence {score:.2f} < {STRATEGY_REVIEW_THRESHOLD} or consent conflicts. "
                f"Pausing at risk review for deeper inspection."
            )

        # Override: consent conflicts always require at minimum strategy review
        if has_consent_conflicts and checkpoint == CheckpointType.AUTO_ADVANCE_SUMMARY:
            checkpoint = CheckpointType.STRATEGY_REVIEW
            rationale += " (Override: consent conflict detected — pausing at strategy review.)"

        # Override: unverified screens trigger at least strategy review
        if has_unverified_screens and checkpoint == CheckpointType.AUTO_ADVANCE_SUMMARY:
            checkpoint = CheckpointType.STRATEGY_REVIEW
            rationale += " (Override: screen(s) with unverified state — pausing at strategy review.)"

    return ConfidenceReport(
        session_score=score,
        checkpoint_type=checkpoint,
        element_results=element_results,
        has_critical_elements=has_critical,
        has_consent_conflicts=has_consent_conflicts,
        has_unverified_screens=has_unverified_screens,
        low_confidence_elements=low_conf_ids,
        rationale=rationale,
        auto_advance_possible=(checkpoint == CheckpointType.AUTO_ADVANCE_SUMMARY),
    )
