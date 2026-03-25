"""
Coordinator Agent — ML/Adaptive components.

Modules:
  intent_classifier    — Hybrid regex+VLM intent classification
  hitl_confidence      — Per-element and session confidence scoring
  adaptive_learning    — Preference learning without VLM retraining
  coordinator_prompts  — Qwen3-VL-30B prompt templates
  reexecution_graph    — Stage dependency graph for selective re-execution
  evaluation_metrics   — Precision/recall/F1 metrics for paper benchmarking
"""
from agents.coordinator.intent_classifier import (
    IntentAction,
    ParsedIntent,
    hybrid_classify,
    needs_clarification,
    generate_clarification_prompt,
    build_vlm_intent_prompt,
    parse_vlm_intent_response,
)
from agents.coordinator.hitl_confidence import (
    HITLMode,
    CheckpointType,
    ConfidenceReport,
    compute_hitl_decision,
    element_confidence,
    session_confidence,
    AUTO_ADVANCE_THRESHOLD,
    STRATEGY_REVIEW_THRESHOLD,
)
from agents.coordinator.adaptive_learning import (
    PreferenceManager,
    MethodPreferenceLearner,
    ThresholdOverrideLearner,
    ConsentRateLearner,
    MIN_OVERRIDES_FOR_PREFERENCE,
    PREFERENCE_CONSISTENCY_THRESHOLD,
)
from agents.coordinator.coordinator_prompts import (
    build_explain_prompt,
    build_summary_prompt,
    parse_summary_response,
    build_reexecution_explain_prompt,
)
from agents.coordinator.reexecution_graph import (
    PipelineStage,
    ModificationType,
    ReExecutionPlan,
    compute_reexecution_plan,
    intent_to_modification_type,
    sam_masks_still_valid,
)
from agents.coordinator.evaluation_metrics import (
    run_full_evaluation,
    compute_intent_classification_metrics,
    compute_hitl_decision_metrics,
    compute_reexecution_metrics,
    compute_satisfaction_correlation,
    INTENT_TARGET_MACRO_F1,
    HITL_TARGET_MISSED_CRITICAL_RATE_MAX,
    REEXECUTION_TARGET_FALSE_SKIP_RATE_MAX,
)

from agents.coordinator.state import (
    InnerPipelineState,
    CoordinatorState,
    DisagreementEvent,
    PipelineSnapshot,
    STAGE_ORDER,
    STAGE_DEPENDENCY_MAP,
)
from agents.coordinator.tools import (
    query_detections,
    query_risk_assessments,
    query_strategies,
    explain_decision,
    apply_strategy_change,
    apply_ignore,
    apply_strengthen,
)
from agents.coordinator.nodes import NodeContext
from agents.coordinator.coordinator_graph import build_coordinator_graph
from agents.coordinator.main import CoordinatorSession

__all__ = [
    # Original ML components
    "IntentAction", "ParsedIntent", "hybrid_classify",
    "needs_clarification", "generate_clarification_prompt",
    "build_vlm_intent_prompt", "parse_vlm_intent_response",
    "HITLMode", "CheckpointType", "ConfidenceReport", "compute_hitl_decision",
    "element_confidence", "session_confidence",
    "AUTO_ADVANCE_THRESHOLD", "STRATEGY_REVIEW_THRESHOLD",
    "PreferenceManager", "MethodPreferenceLearner",
    "ThresholdOverrideLearner", "ConsentRateLearner",
    "MIN_OVERRIDES_FOR_PREFERENCE", "PREFERENCE_CONSISTENCY_THRESHOLD",
    "build_explain_prompt", "build_summary_prompt",
    "parse_summary_response", "build_reexecution_explain_prompt",
    "PipelineStage", "ModificationType", "ReExecutionPlan",
    "compute_reexecution_plan", "intent_to_modification_type",
    "sam_masks_still_valid",
    "run_full_evaluation", "compute_intent_classification_metrics",
    "compute_hitl_decision_metrics", "compute_reexecution_metrics",
    "compute_satisfaction_correlation",
    "INTENT_TARGET_MACRO_F1", "HITL_TARGET_MISSED_CRITICAL_RATE_MAX",
    "REEXECUTION_TARGET_FALSE_SKIP_RATE_MAX",
    # LangGraph orchestration layer
    "InnerPipelineState", "CoordinatorState",
    "DisagreementEvent", "PipelineSnapshot",
    "STAGE_ORDER", "STAGE_DEPENDENCY_MAP",
    "query_detections", "query_risk_assessments", "query_strategies",
    "explain_decision",
    "apply_strategy_change", "apply_ignore", "apply_strengthen",
    "NodeContext",
    "build_coordinator_graph",
    "CoordinatorSession",
]
