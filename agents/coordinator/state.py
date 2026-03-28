"""
LangGraph state schemas for the Coordinator Agent.

Two state TypedDicts:
  InnerPipelineState — mirrors one full pipeline session's data as it
                       flows through detection → export.
  CoordinatorState   — outer graph state including conversation, intent,
                       HITL decisions, and a reference to the inner state.

Additional TypedDicts for multi-agent enhancements:
  DisagreementEvent  — records Phase 1 vs Phase 2 disagreement.
  PipelineSnapshot   — immutable checkpoint for undo.

Stage ordering and dependency constants are also defined here so that
all graph nodes import from a single authoritative source.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
# Stage constants (canonical names used throughout the coordinator graph)

#: Ordered list of stage names, earliest first.
STAGE_ORDER: List[str] = [
    "detection",
    "risk",
    "consent",
    "strategy",
    "sam",
    "execution",
    "export",
]

#: Maps each stage to the list of stages that *must* re-run when that stage
#: is invalidated (forward-invalidation only).
STAGE_DEPENDENCY_MAP: Dict[str, List[str]] = {
    "detection": ["risk", "consent", "strategy", "sam", "execution", "export"],
    "risk":      ["strategy", "sam", "execution", "export"],
    "consent":   ["strategy", "sam", "execution", "export"],
    "strategy":  ["sam", "execution", "export"],
    "sam":       ["execution", "export"],
    "execution": ["export"],
    "export":    [],
}
# DisagreementEvent — records Phase 1 vs Phase 2 disagreement

class DisagreementEvent(TypedDict):
    """Records when Phase 1 and Phase 2 disagree on an element."""
    stage: str                    # "risk" or "strategy"
    detection_id: str
    field: str                    # "severity" or "method"
    phase1_value: str
    phase2_value: str
    reasoning: str
# PipelineSnapshot — immutable checkpoint for undo

class PipelineSnapshot(TypedDict, total=False):
    """Immutable checkpoint for undo."""
    snapshot_id: str
    timestamp: float
    entry_stage: str
    detections: Optional[dict]
    risk_result: Optional[dict]
    strategy_result: Optional[dict]
    execution_report: Optional[dict]
    protected_image_path: Optional[str]
    seg_results: Optional[dict]
    risk_map_path: Optional[str]
    strategy_json_path: Optional[str]
# InnerPipelineState

class InnerPipelineState(TypedDict, total=False):
    """
    All mutable data for one pipeline run (Phases A-F).

    Keys are optional (total=False) because the state is built up
    incrementally as the pipeline advances stage by stage.  Callers must
    guard against missing keys rather than assuming all fields are present.

    Canonical value sets
    --------------------
    stage names  : detection | risk | consent | strategy | sam | execution | export
    severity     : critical | high | medium | low
    methods      : blur | pixelate | solid_overlay | inpaint |
                   avatar_replace | generative_replace | none
    """

    # Session identity
    session_id: str                      # UUID hex string

    # Input
    image_path: str                      # Absolute path to source image

    # Phase A — Detection (Agent 1)
    detections: Optional[Any]            # DetectionResults model instance or dict

    # Phase B — Risk Assessment (Agent 2)
    risk_result: Optional[Any]           # RiskAnalysisResult model instance or dict

    # Phase C — Consent Identity (Agent 2.5)
    # Enriched into risk_result / strategy_result; no separate field needed,
    # but we keep a convenience list of identity-augmented assessment dicts.
    identity_assessments: Optional[List[Dict[str, Any]]]

    # Phase D — Strategy (Agent 3)
    strategy_result: Optional[Any]       # StrategyRecommendations or dict

    # Phase D.5 — SAM Segmentation
    seg_results: Optional[Dict[str, Any]]  # detection_id → mask path / array

    # Phase E — Execution (Agent 4)
    execution_report: Optional[Any]      # ExecutionReport or dict

    # Phase F — Export
    protected_image_path: Optional[str]  # Absolute path to protected image
    risk_map_path: Optional[str]
    strategy_json_path: Optional[str]

    # Re-execution entry point (set by reexecution_graph)
    entry_stage: Optional[str]

    # Ablation / feature flags
    fallback_only: bool                  # True = no VLM calls at all

    # User-requested modifications (accumulated between HITL pauses)
    # Each item: {"detection_id": str, "type": str, "value": str, ...}
    pending_modifications: List[Dict[str, Any]]

    # Snapshot of the last batch of applied modifications (for multi-turn "change back")
    # Preserved before pending_modifications is cleared during pipeline re-run
    last_applied_modifications: List[Dict[str, Any]]

    # Per-stage timing (stage_name → elapsed_ms)
    stage_timings: Dict[str, float]

    # Non-fatal errors keyed by stage name
    errors: Dict[str, str]

    # Phase 1 vs Phase 2 disagreements recorded after risk and strategy stages
    phase_disagreements: List[Dict[str, Any]]   # List[DisagreementEvent]

    # Cached PIL Image to avoid re-opening the same file multiple times
    _cached_image: Any                   # PIL.Image.Image or None
# CoordinatorState

class CoordinatorState(TypedDict, total=False):
    """
    Outer LangGraph state for the Coordinator Agent session.

    The coordinator wraps one InnerPipelineState (created when the user
    first says "process this image") and augments it with conversation
    management, HITL confidence gating, and intent routing.
    """

    # Session identity (same UUID as InnerPipelineState.session_id)
    session_id: str

    # NL conversation history: list of {"role": "user"|"assistant", "content": str}
    conversation_history: List[Dict[str, str]]

    # Latest classified intent (ParsedIntent dataclass serialised to dict)
    current_intent: Optional[Dict[str, Any]]

    # The active inner pipeline state (mutated as stages advance)
    pipeline_state: Optional[InnerPipelineState]

    # HITL gate outcome from hitl_confidence.compute_hitl_decision()
    # CheckpointType string: auto_advance_summary | strategy_review |
    #                        risk_review | full_manual_review
    hitl_checkpoint: Optional[str]

    # True when the graph is paused waiting for the user to approve / reject
    pending_user_decision: bool

    # Full ConfidenceReport dataclass serialised to dict (for UI display)
    hitl_confidence: Optional[Dict[str, Any]]

    # Text to send back to the user as the coordinator's reply
    response_text: str

    # Optional action suggestions surfaced to the UI
    suggestions: List[str]

    # Describes any pipeline action taken this turn
    # e.g. "rerun_from:strategy" | "strategy_modified:det_001" | "export_complete"
    pipeline_action_taken: Optional[str]

    # Append-only audit trail for this coordinator session
    # (separate from the Safety Kernel's OverrideRecord trail)
    audit_trail: List[Dict[str, Any]]

    # Immutable snapshots for undo — max 10 kept, newest first
    snapshots: List[Dict[str, Any]]       # List[PipelineSnapshot]

    # Grouped HITL presentation data for the UI
    hitl_presentation: Optional[Dict[str, Any]]

    # Pending safety challenge awaiting user confirmation.
    # Dict with keys: detection_id, action, method (or None when no challenge is active)
    pending_challenge: Optional[Dict[str, Any]]
