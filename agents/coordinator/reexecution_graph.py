"""
Selective Re-Execution Dependency Graph.

Answers: "Given that the user changed X, which pipeline stages must re-run
and which can use cached results?"

Design:
  - Represented as a DAG where nodes are pipeline stages and edges mean
    "downstream stage depends on upstream stage's output".
  - Forward-invalidation: if stage S is modified, all stages downstream
    of S must re-run. Upstream stages are always cached.
  - Confidence-based skip decisions: if the modification is ONLY to
    execution parameters (method/strength) and the element mask has not
    changed, SAM can also be skipped.
  - The graph is deterministic (no ML model required). The only ML component
    is the confidence check that decides if SAM masks are still valid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set
# Stage definitions

class PipelineStage(str, Enum):
    DETECT    = "detect"
    RISK      = "risk"
    CONSENT   = "consent"
    STRATEGY  = "strategy"
    SAM       = "sam"
    EXECUTION = "execution"
    EXPORT    = "export"


# Stage ordering (index = topological order)
_STAGE_ORDER: Dict[PipelineStage, int] = {
    PipelineStage.DETECT:    0,
    PipelineStage.RISK:      1,
    PipelineStage.CONSENT:   2,
    PipelineStage.STRATEGY:  3,
    PipelineStage.SAM:       4,
    PipelineStage.EXECUTION: 5,
    PipelineStage.EXPORT:    6,
}

# Downstream dependency map: stage → all stages that depend on it
_DOWNSTREAM: Dict[PipelineStage, FrozenSet[PipelineStage]] = {
    PipelineStage.DETECT:    frozenset({
        PipelineStage.RISK, PipelineStage.CONSENT,
        PipelineStage.STRATEGY, PipelineStage.SAM,
        PipelineStage.EXECUTION, PipelineStage.EXPORT,
    }),
    PipelineStage.RISK:      frozenset({
        PipelineStage.STRATEGY, PipelineStage.SAM,
        PipelineStage.EXECUTION, PipelineStage.EXPORT,
    }),
    PipelineStage.CONSENT:   frozenset({
        PipelineStage.STRATEGY, PipelineStage.SAM,
        PipelineStage.EXECUTION, PipelineStage.EXPORT,
    }),
    PipelineStage.STRATEGY:  frozenset({
        PipelineStage.SAM, PipelineStage.EXECUTION, PipelineStage.EXPORT,
    }),
    PipelineStage.SAM:       frozenset({
        PipelineStage.EXECUTION, PipelineStage.EXPORT,
    }),
    PipelineStage.EXECUTION: frozenset({
        PipelineStage.EXPORT,
    }),
    PipelineStage.EXPORT:    frozenset(),
}

# CONSENT is INDEPENDENT of RISK (same faces; risk doesn't affect face matching)
# This asymmetry is intentional: modifying risk does not re-run consent.
_CONSENT_INDEPENDENT_OF_RISK = True
# Modification types (what the user changed)

class ModificationType(str, Enum):
    # User ran full pipeline for the first time
    FULL_PIPELINE = "full_pipeline"
    # User changed a strategy method (blur → pixelate) without changing severity
    METHOD_ONLY_CHANGE = "method_only_change"
    # User changed a strategy method AND the element mask needs updating
    METHOD_WITH_MASK_CHANGE = "method_with_mask_change"
    # User changed severity of a risk assessment
    SEVERITY_CHANGE = "severity_change"
    # User changed consent status for a face
    CONSENT_CHANGE = "consent_change"
    # User ignored (removed protection from) an element
    IGNORE_ELEMENT = "ignore_element"
    # User strengthened existing protection (expand_px > 0 or stronger method)
    STRENGTHEN_ONLY = "strengthen_only"
    # User added a new protection region via coordinates
    ADD_REGION = "add_region"
    # User changed detection parameters (would invalidate everything)
    DETECTION_CHANGE = "detection_change"
# Concrete skip rules

@dataclass
class ReExecutionPlan:
    """
    Result of the dependency graph computation.
    """
    must_rerun: List[PipelineStage]      # Ordered list of stages to re-execute
    can_skip: List[PipelineStage]         # Stages that can use cached results
    entry_stage: PipelineStage           # Where to enter the pipeline DAG
    modification_type: ModificationType
    sam_masks_valid: bool                # If True, SAM can also be skipped
    rationale: str                       # Human-readable explanation
    estimated_speedup: float             # Approximate speedup vs full re-run (1x = no speedup)


# Approximate stage latencies (seconds) for speedup estimation
# Matches Table in COORDINATOR_BLUEPRINT.md Section 11
_STAGE_LATENCY_S: Dict[PipelineStage, float] = {
    PipelineStage.DETECT:    4.0,   # middle of 3-5s range
    PipelineStage.RISK:      0.75,
    PipelineStage.CONSENT:   2.5,
    PipelineStage.STRATEGY:  0.75,
    PipelineStage.SAM:       10.0,  # middle of 8-12s range
    PipelineStage.EXECUTION: 2.5,
    PipelineStage.EXPORT:    1.5,
}

_FULL_PIPELINE_LATENCY_S = sum(_STAGE_LATENCY_S.values())  # 22.5s


def _ordered_stages(stages: Set[PipelineStage]) -> List[PipelineStage]:
    return sorted(stages, key=lambda s: _STAGE_ORDER[s])


def _estimated_latency(stages: List[PipelineStage]) -> float:
    return sum(_STAGE_LATENCY_S.get(s, 1.0) for s in stages)
# 5.1  Mask validity check
#
# The key ML question: "If the user only changed the obfuscation method
# (e.g., blur → pixelate), are the existing SAM masks still valid?"
#
# Answer: Yes, if ALL of:
#   1. The bounding box of the element has NOT changed (same detection)
#   2. The element type is the same (face mask is not valid for text)
#   3. The new method still uses a mask (not solid_overlay which uses bbox only)
#
# Method → uses_mask mapping:
#   blur           : True  (applies blur to SAM mask region)
#   pixelate       : True  (applies pixelation to SAM mask region)
#   solid_overlay  : False (applies solid rect to bbox; mask not used)
#   avatar_replace : True  (replaces SAM mask region)
#   inpaint        : True  (inpaints SAM mask region)
#   none           : False (no protection)

_METHOD_USES_MASK: Dict[str, bool] = {
    "blur":              True,
    "pixelate":          True,
    "solid_overlay":     False,
    "avatar_replace":    True,
    "inpaint":           True,
    "generative_replace": True,
    "none":              False,
}


def sam_masks_still_valid(
    old_strategy: Dict,
    new_method: str,
    detection_bbox_changed: bool = False,
) -> bool:
    """
    Predict whether existing SAM masks can be reused after a method change.

    Args:
        old_strategy:         The strategy dict before the user's change.
        new_method:           The new method the user requested.
        detection_bbox_changed: If the element's bbox was re-detected differently.

    Returns:
        True if existing SAM mask is valid for the new method.

    This function is the primary ML optimization for selective re-execution:
    skipping SAM saves 8-12s per element.
    """
    if detection_bbox_changed:
        return False  # Bbox changed → mask is spatially wrong

    old_method = old_strategy.get("method", "none")
    old_uses_mask = _METHOD_USES_MASK.get(old_method, False)
    new_uses_mask = _METHOD_USES_MASK.get(new_method, False)

    if not old_uses_mask:
        # Old method didn't generate a mask (e.g., solid_overlay)
        # New method may need one → must re-run SAM
        return False

    if not new_uses_mask:
        # New method doesn't use mask → mask doesn't matter → skip SAM
        return True

    # Both use masks and bbox is unchanged: mask is valid
    # (SAM output is deterministic given same bbox + same image)
    return True
# 5.2  Dependency graph resolver

def compute_reexecution_plan(
    modification_type: ModificationType,
    modified_elements: Optional[List[str]] = None,  # detection_ids
    old_strategies: Optional[List[Dict]] = None,
    new_strategies: Optional[List[Dict]] = None,
    detection_bbox_changed: bool = False,
) -> ReExecutionPlan:
    """
    Compute which stages must re-run given a user modification.

    This is a DETERMINISTIC function — no randomness, no ML model.
    The SAM validity check is the only inference-based component.

    Args:
        modification_type:      What the user changed.
        modified_elements:      Which detection IDs were affected (optional).
        old_strategies:         Strategy dicts before modification.
        new_strategies:         Strategy dicts after modification.
        detection_bbox_changed: If any detection bbox was re-computed.

    Returns:
        ReExecutionPlan with ordered must_rerun list and entry_stage.
    """
    all_stages = set(PipelineStage)
    must_rerun: Set[PipelineStage] = set()
    sam_valid = True
    rationale_parts: List[str] = []

    if modification_type == ModificationType.FULL_PIPELINE:
        must_rerun = all_stages
        entry = PipelineStage.DETECT
        sam_valid = False
        rationale_parts.append("Full pipeline run from detection.")

    elif modification_type == ModificationType.DETECTION_CHANGE:
        must_rerun = all_stages
        entry = PipelineStage.DETECT
        sam_valid = False
        rationale_parts.append("Detection parameters changed; all stages invalidated.")

    elif modification_type == ModificationType.SEVERITY_CHANGE:
        # Risk changed → strategy downstream must re-run; consent is independent
        must_rerun = {
            PipelineStage.STRATEGY, PipelineStage.SAM,
            PipelineStage.EXECUTION, PipelineStage.EXPORT,
        }
        entry = PipelineStage.STRATEGY
        sam_valid = False  # Strategy changed → new mask needed
        rationale_parts.append(
            "Severity changed: strategy, SAM, execution, and export must re-run. "
            "Detection, risk, and consent results are cached."
        )

    elif modification_type == ModificationType.CONSENT_CHANGE:
        # Consent change → strategy downstream (risk is independent of consent)
        must_rerun = {
            PipelineStage.STRATEGY, PipelineStage.SAM,
            PipelineStage.EXECUTION, PipelineStage.EXPORT,
        }
        entry = PipelineStage.STRATEGY
        sam_valid = False
        rationale_parts.append(
            "Consent status changed: strategy onwards must re-run. "
            "Detection, risk, and consent results are cached."
        )

    elif modification_type == ModificationType.METHOD_ONLY_CHANGE:
        # Determine if SAM masks are still valid
        if (old_strategies and new_strategies):
            # Check each modified element
            all_masks_valid = True
            for old_s, new_s in zip(old_strategies, new_strategies):
                if old_s.get("detection_id") == new_s.get("detection_id"):
                    new_method = new_s.get("method", "none")
                    valid = sam_masks_still_valid(old_s, new_method, detection_bbox_changed)
                    if not valid:
                        all_masks_valid = False
                        break
            sam_valid = all_masks_valid
        else:
            sam_valid = False

        if sam_valid:
            # SAM masks are reusable: skip strategy + SAM, re-run only execution + export
            must_rerun = {PipelineStage.EXECUTION, PipelineStage.EXPORT}
            entry = PipelineStage.EXECUTION
            rationale_parts.append(
                "Method changed (e.g., blur→pixelate) but SAM masks are still valid. "
                "Only execution and export need to re-run. "
                "SAM segmentation is reused from cache."
            )
        else:
            must_rerun = {
                PipelineStage.SAM, PipelineStage.EXECUTION, PipelineStage.EXPORT,
            }
            entry = PipelineStage.SAM
            rationale_parts.append(
                "Method changed and new method requires a different SAM mask. "
                "SAM, execution, and export must re-run."
            )

    elif modification_type == ModificationType.METHOD_WITH_MASK_CHANGE:
        must_rerun = {
            PipelineStage.SAM, PipelineStage.EXECUTION, PipelineStage.EXPORT,
        }
        entry = PipelineStage.SAM
        sam_valid = False
        rationale_parts.append(
            "Method changed and bbox or element type changed. "
            "SAM must re-run to generate new mask."
        )

    elif modification_type == ModificationType.IGNORE_ELEMENT:
        # Ignoring an element removes its mask from SAM and its obfuscation
        must_rerun = {
            PipelineStage.STRATEGY, PipelineStage.SAM,
            PipelineStage.EXECUTION, PipelineStage.EXPORT,
        }
        entry = PipelineStage.STRATEGY
        sam_valid = False
        rationale_parts.append(
            "Element ignored: strategy updated, SAM re-runs (no mask for ignored element), "
            "execution and export re-run."
        )

    elif modification_type == ModificationType.STRENGTHEN_ONLY:
        # Only strengthen: execution with expand_px change; SAM masks still valid
        must_rerun = {PipelineStage.EXECUTION, PipelineStage.EXPORT}
        entry = PipelineStage.EXECUTION
        sam_valid = True  # Expand is handled in execution, not SAM
        rationale_parts.append(
            "Strengthening protection: only execution and export need to re-run. "
            "SAM masks are reused; bbox expansion is applied at execution time."
        )

    elif modification_type == ModificationType.ADD_REGION:
        # Adding a new region: no SAM needed (solid overlay on user-specified coords)
        must_rerun = {PipelineStage.EXECUTION, PipelineStage.EXPORT}
        entry = PipelineStage.EXECUTION
        sam_valid = True
        rationale_parts.append(
            "New protection region added: execution and export re-run to apply "
            "protection to the new region. Existing segmentation unchanged."
        )

    else:
        # Unknown modification: conservative full re-run
        must_rerun = all_stages
        entry = PipelineStage.DETECT
        sam_valid = False
        rationale_parts.append("Unknown modification type: full pipeline re-run for safety.")

    # Compute can_skip as complement of must_rerun
    can_skip = sorted(all_stages - must_rerun, key=lambda s: _STAGE_ORDER[s])
    must_rerun_ordered = _ordered_stages(must_rerun)

    # Speedup calculation
    rerun_latency = _estimated_latency(must_rerun_ordered)
    speedup = _FULL_PIPELINE_LATENCY_S / max(rerun_latency, 0.1)

    return ReExecutionPlan(
        must_rerun=must_rerun_ordered,
        can_skip=can_skip,
        entry_stage=entry,
        modification_type=modification_type,
        sam_masks_valid=sam_valid,
        rationale=" ".join(rationale_parts),
        estimated_speedup=round(speedup, 1),
    )
# 5.3  Intent → ModificationType mapping
# (bridges intent_classifier.py to reexecution_graph.py)

def intent_to_modification_type(
    intent_action: str,
    method_specified: Optional[str],
    old_strategy: Optional[Dict] = None,
) -> ModificationType:
    """
    Map a classified intent to the appropriate ModificationType.

    This is used by the coordinator graph's ROUTE_INTENT node to determine
    the re-execution plan before entering the inner pipeline.

    Args:
        intent_action:    IntentAction value as string.
        method_specified: The new method if specified (e.g., "pixelate").
        old_strategy:     The existing strategy dict, used to check mask validity.

    Returns:
        ModificationType for use in compute_reexecution_plan().
    """
    if intent_action == "process":
        return ModificationType.FULL_PIPELINE

    if intent_action == "strengthen":
        return ModificationType.STRENGTHEN_ONLY

    if intent_action == "ignore":
        return ModificationType.IGNORE_ELEMENT

    if intent_action == "modify_strategy":
        if method_specified and old_strategy:
            old_method = old_strategy.get("method", "none")
            # Check if mask validity changes with new method
            masks_still_valid = sam_masks_still_valid(
                old_strategy, method_specified, detection_bbox_changed=False
            )
            if masks_still_valid:
                return ModificationType.METHOD_ONLY_CHANGE
            else:
                return ModificationType.METHOD_WITH_MASK_CHANGE
        return ModificationType.METHOD_ONLY_CHANGE  # Default: assume mask valid

    # Default: treat unknown write actions as full re-run (conservative)
    return ModificationType.FULL_PIPELINE
