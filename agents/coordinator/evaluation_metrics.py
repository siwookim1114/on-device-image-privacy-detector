"""
Evaluation Metrics for ML Components of the Coordinator Agent.

Defines what to measure to validate the ML components and how to compute
each metric. All functions operate on collected logs / ground-truth datasets
so they can be run offline on collected session data.

Metrics are grouped into four areas:
  1. Intent classification quality     (precision/recall/F1 per action type)
  2. HITL decision quality             (false pause rate, missed critical rate)
  3. Selective re-execution accuracy   (did we skip a stage that should have re-run?)
  4. User satisfaction correlation     (Pearson r between confidence threshold and rating)
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
# 6.1  Intent classification metrics

@dataclass
class IntentClassificationSample:
    """One labeled sample for intent classification evaluation."""
    query: str
    true_action: str        # Ground-truth IntentAction value
    predicted_action: str   # System-predicted IntentAction value
    predicted_confidence: float
    used_vlm: bool          # Did the system escalate to VLM?
    latency_ms: float


@dataclass
class IntentClassificationReport:
    """Per-class and overall precision/recall/F1 for intent classification."""
    per_class: Dict[str, Dict[str, float]]  # {action: {precision, recall, f1, support}}
    macro_f1: float
    weighted_f1: float
    vlm_escalation_rate: float   # Fraction of queries that required VLM
    avg_latency_ms: float
    confusion_matrix: Dict[str, Dict[str, int]]  # {true: {pred: count}}


def compute_intent_classification_metrics(
    samples: List[IntentClassificationSample],
) -> IntentClassificationReport:
    """
    Compute precision, recall, F1 per intent action class.

    Uses macro and weighted averaging to handle class imbalance
    (APPROVE/REJECT queries are less frequent than QUERY/PROCESS).
    """
    all_classes = sorted(set(
        s.true_action for s in samples
    ) | set(s.predicted_action for s in samples))

    # Build confusion matrix
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in samples:
        confusion[s.true_action][s.predicted_action] += 1

    # Per-class metrics
    per_class: Dict[str, Dict[str, float]] = {}
    for cls in all_classes:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in all_classes if other != cls)
        fn = sum(confusion[cls][other] for other in all_classes if other != cls)

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = (2 * precision * recall) / max(precision + recall, 1e-9)
        support   = sum(confusion[cls].values())

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   float(support),
        }

    # Macro F1 (unweighted mean across classes)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / max(len(per_class), 1)

    # Weighted F1 (support-weighted mean)
    total_support = sum(v["support"] for v in per_class.values())
    weighted_f1 = (
        sum(v["f1"] * v["support"] for v in per_class.values())
        / max(total_support, 1)
    )

    vlm_escalation_rate = (
        sum(1 for s in samples if s.used_vlm) / max(len(samples), 1)
    )
    avg_latency = sum(s.latency_ms for s in samples) / max(len(samples), 1)

    return IntentClassificationReport(
        per_class=per_class,
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        vlm_escalation_rate=round(vlm_escalation_rate, 4),
        avg_latency_ms=round(avg_latency, 2),
        confusion_matrix={k: dict(v) for k, v in confusion.items()},
    )


# Target thresholds for paper evaluation:
INTENT_TARGET_MACRO_F1    = 0.90
INTENT_TARGET_WEIGHTED_F1 = 0.93
INTENT_TARGET_VLM_RATE_MAX = 0.25  # VLM should only be needed for <25% of queries
# 6.2  HITL decision quality metrics

@dataclass
class HITLDecisionSample:
    """
    One session's HITL decision for evaluation.

    Ground truth is determined retrospectively:
      - Should have paused = True if user made a meaningful override AFTER
        auto-advance (i.e., system was wrong to auto-advance)
      - Should have paused = False if user approved without changes
        (i.e., auto-advance was correct)
      - Had critical = True if any CRITICAL element was in the image
      - Missed critical = True if CRITICAL element was auto-advanced without pause
    """
    session_id: str
    checkpoint_type: str           # What system decided (auto/strategy/risk)
    session_confidence: float      # System's confidence score
    user_made_override: bool       # Did user change anything after decision?
    had_critical_element: bool
    system_paused: bool            # True if checkpoint != AUTO_ADVANCE_SUMMARY
    should_have_paused: bool       # Ground truth


@dataclass
class HITLDecisionReport:
    """Quality metrics for HITL gating decisions."""
    # False pause rate: system paused when it didn't need to (precision for "no-pause")
    false_pause_rate: float
    # Missed critical rate: system auto-advanced when it had a CRITICAL element
    # This is the safety-critical metric; target < 1%
    missed_critical_rate: float
    # True pause rate: when system paused, user actually needed to make changes
    useful_pause_rate: float
    # Threshold calibration: is the 0.85 threshold appropriate?
    optimal_threshold: float  # Estimated from ROC curve
    # Per-threshold breakdown
    threshold_analysis: Dict[float, Dict[str, float]]


def compute_hitl_decision_metrics(
    samples: List[HITLDecisionSample],
) -> HITLDecisionReport:
    """
    Compute HITL decision quality metrics.

    Key safety invariant: missed_critical_rate MUST be < 0.01 (1%).
    If it exceeds this, the AUTO_ADVANCE_THRESHOLD should be raised.
    """
    n = len(samples)
    if n == 0:
        return HITLDecisionReport(
            false_pause_rate=0.0,
            missed_critical_rate=0.0,
            useful_pause_rate=0.0,
            optimal_threshold=0.85,
            threshold_analysis={},
        )

    # False pause rate: paused but user made no changes (unnecessary pause)
    paused = [s for s in samples if s.system_paused]
    unnecessary_pauses = [s for s in paused if not s.user_made_override]
    false_pause_rate = len(unnecessary_pauses) / max(len(paused), 1)

    # Missed critical rate: had CRITICAL element but system auto-advanced
    critical_sessions = [s for s in samples if s.had_critical_element]
    missed_criticals = [s for s in critical_sessions if not s.system_paused]
    missed_critical_rate = len(missed_criticals) / max(len(critical_sessions), 1)

    # Useful pause rate: paused AND user made override
    useful_pauses = [s for s in paused if s.user_made_override]
    useful_pause_rate = len(useful_pauses) / max(len(paused), 1)

    # Threshold analysis: compute pause quality at different thresholds
    thresholds = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    threshold_analysis: Dict[float, Dict[str, float]] = {}

    for t in thresholds:
        # Simulate pausing at threshold t
        would_pause = [s for s in samples if s.session_confidence < t]
        would_advance = [s for s in samples if s.session_confidence >= t]

        # Missed criticals at this threshold
        mc = sum(1 for s in would_advance if s.had_critical_element)
        mc_rate = mc / max(len(critical_sessions), 1)

        # False pauses at this threshold
        fp = sum(1 for s in would_pause if not s.user_made_override)
        fp_rate = fp / max(len(would_pause), 1)

        # F1-like score balancing false pauses and missed criticals
        # We weight missed criticals 5x more (safety-critical)
        penalty = fp_rate + 5.0 * mc_rate
        threshold_analysis[t] = {
            "missed_critical_rate": round(mc_rate, 4),
            "false_pause_rate": round(fp_rate, 4),
            "combined_penalty": round(penalty, 4),
            "pause_count": len(would_pause),
        }

    # Find optimal threshold (minimum combined penalty)
    optimal_threshold = min(
        threshold_analysis.keys(),
        key=lambda t: threshold_analysis[t]["combined_penalty"],
    )

    return HITLDecisionReport(
        false_pause_rate=round(false_pause_rate, 4),
        missed_critical_rate=round(missed_critical_rate, 4),
        useful_pause_rate=round(useful_pause_rate, 4),
        optimal_threshold=optimal_threshold,
        threshold_analysis=threshold_analysis,
    )


# Target thresholds for paper evaluation:
HITL_TARGET_MISSED_CRITICAL_RATE_MAX = 0.01   # 1%: hard safety invariant
HITL_TARGET_FALSE_PAUSE_RATE_MAX     = 0.35   # 35%: at most 35% of pauses are unnecessary
HITL_TARGET_USEFUL_PAUSE_RATE_MIN    = 0.65   # 65%: at least 65% of pauses are useful
# 6.3  Selective re-execution accuracy metrics

@dataclass
class ReExecutionSample:
    """
    One re-execution decision for evaluation.

    Ground truth: did the skipped stage actually produce different output?
    (Determined by running it and comparing to cached output.)
    """
    session_id: str
    modification_type: str
    skipped_stages: List[str]
    stages_that_would_have_changed: List[str]  # Ground truth: which stages changed
    time_saved_ms: float                        # Time saved by skipping
    correctness: bool  # True if skip was correct (skipped stages had same output)


@dataclass
class ReExecutionReport:
    """Accuracy and latency metrics for selective re-execution."""
    skip_precision: float    # Of stages we skipped, fraction that were safe to skip
    skip_recall: float       # Of stages safe to skip, fraction we actually skipped
    false_skip_rate: float   # Fraction of skips that caused incorrect results
    avg_time_saved_ms: float
    avg_speedup: float
    per_modification_breakdown: Dict[str, Dict[str, float]]


def compute_reexecution_metrics(
    samples: List[ReExecutionSample],
) -> ReExecutionReport:
    """
    Compute selective re-execution accuracy metrics.

    false_skip_rate is the primary safety metric here:
      - A false skip means we skipped a stage that SHOULD have re-run
        (producing stale/incorrect output without the user knowing).
      - Target: < 2% (extremely rare, as incorrect skips could leave
        sensitive content unprotected).
    """
    if not samples:
        return ReExecutionReport(
            skip_precision=1.0,
            skip_recall=1.0,
            false_skip_rate=0.0,
            avg_time_saved_ms=0.0,
            avg_speedup=1.0,
            per_modification_breakdown={},
        )

    # Per-stage skip accuracy
    total_skips = 0
    correct_skips = 0
    incorrect_skips = 0
    possible_skips = 0
    skipped_possible = 0

    for s in samples:
        for stage in s.skipped_stages:
            total_skips += 1
            if stage in s.stages_that_would_have_changed:
                incorrect_skips += 1  # False skip: should have re-run
            else:
                correct_skips += 1

        # How many stages COULD have been skipped but weren't?
        all_stages = {"detect", "risk", "consent", "strategy", "sam", "execution", "export"}
        safe_to_skip = all_stages - set(s.stages_that_would_have_changed)
        for stage in safe_to_skip:
            possible_skips += 1
            if stage in s.skipped_stages:
                skipped_possible += 1

    skip_precision = correct_skips / max(total_skips, 1)
    skip_recall    = skipped_possible / max(possible_skips, 1)
    false_skip_rate = incorrect_skips / max(total_skips, 1)

    avg_time_saved = sum(s.time_saved_ms for s in samples) / len(samples)

    # Estimate speedup from time saved (approximate)
    full_pipeline_ms = 22500  # 22.5s reference from blueprint
    avg_rerun_ms = full_pipeline_ms - avg_time_saved
    avg_speedup = full_pipeline_ms / max(avg_rerun_ms, 100)

    # Per-modification type breakdown
    per_mod: Dict[str, List[ReExecutionSample]] = defaultdict(list)
    for s in samples:
        per_mod[s.modification_type].append(s)

    per_modification_breakdown: Dict[str, Dict[str, float]] = {}
    for mod_type, mod_samples in per_mod.items():
        mod_skips = sum(len(s.skipped_stages) for s in mod_samples)
        mod_wrong = sum(
            sum(1 for st in s.skipped_stages if st in s.stages_that_would_have_changed)
            for s in mod_samples
        )
        mod_saved = sum(s.time_saved_ms for s in mod_samples) / len(mod_samples)
        per_modification_breakdown[mod_type] = {
            "false_skip_rate": round(mod_wrong / max(mod_skips, 1), 4),
            "avg_time_saved_ms": round(mod_saved, 1),
            "n_samples": float(len(mod_samples)),
        }

    return ReExecutionReport(
        skip_precision=round(skip_precision, 4),
        skip_recall=round(skip_recall, 4),
        false_skip_rate=round(false_skip_rate, 4),
        avg_time_saved_ms=round(avg_time_saved, 1),
        avg_speedup=round(avg_speedup, 2),
        per_modification_breakdown=per_modification_breakdown,
    )


# Target thresholds for paper evaluation:
REEXECUTION_TARGET_FALSE_SKIP_RATE_MAX = 0.02  # 2%: almost no incorrect skips
REEXECUTION_TARGET_SKIP_RECALL_MIN     = 0.70  # 70%: skip at least 70% of skippable stages
REEXECUTION_TARGET_AVG_SPEEDUP_MIN     = 2.0   # 2x: at least 2x faster than full re-run
# 6.4  User satisfaction correlation with confidence thresholds

@dataclass
class UserSatisfactionSample:
    """
    Session-level user satisfaction rating.
    Collected post-session via a 1-5 star rating or binary thumbs-up.
    """
    session_id: str
    session_confidence: float
    checkpoint_type: str
    user_rating: float        # 1-5 scale (or 0/1 binary)
    user_made_changes: bool
    n_elements: int
    n_critical: int
    processing_mode: str      # "auto", "hybrid", "manual"


def compute_satisfaction_correlation(
    samples: List[UserSatisfactionSample],
) -> Dict[str, float]:
    """
    Compute Pearson correlation between confidence thresholds and user satisfaction.

    Returns a dict with:
      - pearson_r: correlation between session_confidence and user_rating
      - pearson_r_checkpoint: correlation between checkpoint_type (ordinal) and rating
      - mean_rating_by_mode: {mode: mean_rating}
      - mean_rating_by_checkpoint: {checkpoint_type: mean_rating}
    """
    if len(samples) < 3:
        return {"error": "insufficient samples"}

    # Pearson r between confidence and rating
    n = len(samples)
    confs = [s.session_confidence for s in samples]
    ratings = [s.user_rating for s in samples]

    mean_conf = sum(confs) / n
    mean_rating = sum(ratings) / n

    numerator = sum((c - mean_conf) * (r - mean_rating) for c, r in zip(confs, ratings))
    denom_conf = math.sqrt(sum((c - mean_conf) ** 2 for c in confs))
    denom_rating = math.sqrt(sum((r - mean_rating) ** 2 for r in ratings))
    denom = denom_conf * denom_rating

    pearson_r = numerator / max(denom, 1e-9)

    # Ordinal encoding of checkpoint type
    checkpoint_ordinal = {
        "auto_advance_summary": 3,
        "strategy_review": 2,
        "risk_review": 1,
        "full_manual_review": 0,
    }
    ck_encoded = [checkpoint_ordinal.get(s.checkpoint_type, 1) for s in samples]
    mean_ck = sum(ck_encoded) / n

    num_ck = sum((c - mean_ck) * (r - mean_rating) for c, r in zip(ck_encoded, ratings))
    denom_ck = math.sqrt(sum((c - mean_ck) ** 2 for c in ck_encoded)) * denom_rating
    pearson_r_checkpoint = num_ck / max(denom_ck, 1e-9)

    # Mean rating by mode
    by_mode: Dict[str, List[float]] = defaultdict(list)
    for s in samples:
        by_mode[s.processing_mode].append(s.user_rating)
    mean_by_mode = {k: round(sum(v) / len(v), 3) for k, v in by_mode.items()}

    # Mean rating by checkpoint type
    by_ck: Dict[str, List[float]] = defaultdict(list)
    for s in samples:
        by_ck[s.checkpoint_type].append(s.user_rating)
    mean_by_ck = {k: round(sum(v) / len(v), 3) for k, v in by_ck.items()}

    return {
        "pearson_r_confidence_vs_rating": round(pearson_r, 4),
        "pearson_r_checkpoint_vs_rating": round(pearson_r_checkpoint, 4),
        "mean_rating": round(mean_rating, 3),
        "mean_rating_by_mode": mean_by_mode,
        "mean_rating_by_checkpoint": mean_by_ck,
        "n_samples": n,
    }
# 6.5  Combined evaluation runner (for paper benchmarking)

@dataclass
class FullEvaluationReport:
    intent_report:      IntentClassificationReport
    hitl_report:        HITLDecisionReport
    reexecution_report: ReExecutionReport
    satisfaction:       Dict[str, float]
    passes_targets:     Dict[str, bool]    # Which targets are met


def run_full_evaluation(
    intent_samples: List[IntentClassificationSample],
    hitl_samples: List[HITLDecisionSample],
    reexecution_samples: List[ReExecutionSample],
    satisfaction_samples: List[UserSatisfactionSample],
) -> FullEvaluationReport:
    """
    Run all four metric groups and check against paper targets.
    """
    ir = compute_intent_classification_metrics(intent_samples)
    hr = compute_hitl_decision_metrics(hitl_samples)
    rr = compute_reexecution_metrics(reexecution_samples)
    sr = compute_satisfaction_correlation(satisfaction_samples)

    passes = {
        "intent_macro_f1":          ir.macro_f1 >= INTENT_TARGET_MACRO_F1,
        "intent_weighted_f1":       ir.weighted_f1 >= INTENT_TARGET_WEIGHTED_F1,
        "intent_vlm_rate":          ir.vlm_escalation_rate <= INTENT_TARGET_VLM_RATE_MAX,
        "hitl_missed_critical":     hr.missed_critical_rate <= HITL_TARGET_MISSED_CRITICAL_RATE_MAX,
        "hitl_false_pause":         hr.false_pause_rate <= HITL_TARGET_FALSE_PAUSE_RATE_MAX,
        "hitl_useful_pause":        hr.useful_pause_rate >= HITL_TARGET_USEFUL_PAUSE_RATE_MIN,
        "reexecution_false_skip":   rr.false_skip_rate <= REEXECUTION_TARGET_FALSE_SKIP_RATE_MAX,
        "reexecution_skip_recall":  rr.skip_recall >= REEXECUTION_TARGET_SKIP_RECALL_MIN,
        "reexecution_speedup":      rr.avg_speedup >= REEXECUTION_TARGET_AVG_SPEEDUP_MIN,
    }

    return FullEvaluationReport(
        intent_report=ir,
        hitl_report=hr,
        reexecution_report=rr,
        satisfaction=sr,
        passes_targets=passes,
    )
