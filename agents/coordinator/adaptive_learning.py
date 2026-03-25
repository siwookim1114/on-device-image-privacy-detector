"""
Adaptive Learning from HITL Feedback.

The system learns user preferences WITHOUT retraining the VLM.
All learning is stored in PrivacyProfile.threshold_overrides and the
MongoDB ConsentHistory + approval_rate fields that already exist.

Three learning mechanisms:
  1. MethodPreferenceLearner   — if user consistently changes blur→pixelate, adapt
  2. ThresholdOverrideLearner  — if user consistently protects/ignores an element
                                  category, adjust threshold_overrides
  3. ConsentRateLearner        — existing approval_rate logic in ConsentHistory;
                                  this module exposes a unified update interface

Design principle: Every learning signal must be REVERSIBLE from the audit trail.
No weights are permanently committed after a single override.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
# Named constants (all tunable via config.yaml in the future)

# Minimum number of consistent overrides before a preference is promoted
# to a default recommendation.
# Justification: 3 is consistent with learning.min_appearances_for_trust
# in config.yaml; below 3 we have insufficient evidence.
MIN_OVERRIDES_FOR_PREFERENCE = 3

# Fraction of overrides in the same direction needed to call it a preference.
# At 0.75, a user must change blur→pixelate 3 out of 4 times before we adapt.
PREFERENCE_CONSISTENCY_THRESHOLD = 0.75

# Maximum method preference weight (how strongly we weight learned preference
# against VLM recommendation in the final strategy selection).
# 0.6 means learned preference can influence but not fully override VLM.
MAX_METHOD_PREFERENCE_WEIGHT = 0.6

# Decay factor per pipeline session: preferences from old sessions fade.
# At 0.95, a 3-session-old preference retains 0.95^3 = 0.857 of its weight.
PREFERENCE_DECAY_PER_SESSION = 0.95

# Keys used in threshold_overrides (PrivacyProfile dict field)
# Format: "{element_category}" -> RiskLevel string
THRESHOLD_KEY_FACE_BYSTANDER   = "face_bystander"
THRESHOLD_KEY_FACE_KNOWN       = "face_known"
THRESHOLD_KEY_FACE_SELF        = "face_self"
THRESHOLD_KEY_TEXT_PII_GENERIC = "text_pii_generic"
THRESHOLD_KEY_TEXT_NUMERIC     = "text_numeric_fragment"
THRESHOLD_KEY_SCREEN_ON        = "screen_verified_on"
THRESHOLD_KEY_SCREEN_OFF       = "screen_verified_off"
# 3.1  Method preference learning

@dataclass
class MethodPreferenceRecord:
    """
    Tracks how many times a user changed from a VLM-recommended method
    to an alternative, grouped by element_type.

    Stored outside PrivacyProfile (in MongoDB, collection: method_preferences)
    because it needs per-session timestamping.
    """
    element_type: str                      # "face", "text", "screen"
    from_method: str                       # What VLM recommended
    to_method: str                         # What user chose
    count: int = 0
    sessions: List[str] = field(default_factory=list)  # session IDs
    last_updated: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    decayed_weight: float = 1.0            # Accumulated decay factor


@dataclass
class LearnedMethodPreference:
    """
    A stabilized preference: when consistent enough, use this method as default.
    """
    element_type: str
    preferred_method: str                  # The method to prefer
    confidence: float                      # 0-1; how confident we are in preference
    based_on_overrides: int                # How many overrides drove this
    active: bool = True                    # Can be disabled by user


class MethodPreferenceLearner:
    """
    Learns per-element-type method preferences from user overrides.

    Usage:
        learner = MethodPreferenceLearner(preference_store)
        learner.record_override("face", from_method="blur", to_method="pixelate", session_id="abc")
        pref = learner.get_preferred_method("face")
        # pref.preferred_method == "pixelate" after MIN_OVERRIDES_FOR_PREFERENCE overrides
    """

    def __init__(self, store: Dict[str, List[MethodPreferenceRecord]]):
        """
        Args:
            store: Mutable dict {element_type: [MethodPreferenceRecord, ...]}
                   Pass an in-memory dict for testing; MongoDB dict for production.
        """
        self._store = store

    def record_override(
        self,
        element_type: str,
        from_method: str,
        to_method: str,
        session_id: str,
    ) -> None:
        """
        Record that the user changed from_method → to_method for element_type.
        Must be called once per HITL override event from the audit trail.
        """
        if element_type not in self._store:
            self._store[element_type] = []

        # Find existing record for this (from, to) pair
        key = f"{from_method}→{to_method}"
        existing = None
        for r in self._store[element_type]:
            if f"{r.from_method}→{r.to_method}" == key:
                existing = r
                break

        if existing is not None:
            # Apply decay to all existing records (simulate aging)
            for r in self._store[element_type]:
                r.decayed_weight *= PREFERENCE_DECAY_PER_SESSION

            existing.count += 1
            existing.decayed_weight += 1.0  # New override gets fresh weight
            if session_id not in existing.sessions:
                existing.sessions.append(session_id)
            existing.last_updated = datetime.utcnow().isoformat()
        else:
            new_record = MethodPreferenceRecord(
                element_type=element_type,
                from_method=from_method,
                to_method=to_method,
                count=1,
                sessions=[session_id],
                decayed_weight=1.0,
            )
            self._store[element_type].append(new_record)

    def get_preferred_method(
        self, element_type: str, current_recommendation: Optional[str] = None
    ) -> Optional[LearnedMethodPreference]:
        """
        Return a LearnedMethodPreference if a stable preference exists.

        Returns None if:
          - Fewer than MIN_OVERRIDES_FOR_PREFERENCE total overrides
          - No single direction has >= PREFERENCE_CONSISTENCY_THRESHOLD ratio
          - All preferences have decayed below a useful threshold
        """
        records = self._store.get(element_type, [])
        if not records:
            return None

        # Total raw count (used for minimum threshold check — decay is for consistency,
        # not for the minimum count gate; decay ensures old preferences fade but
        # we still need at least MIN_OVERRIDES_FOR_PREFERENCE distinct override events)
        total_raw_count = sum(r.count for r in records)
        if total_raw_count < MIN_OVERRIDES_FOR_PREFERENCE:
            return None

        # Use decay-weighted count for consistency scoring (recent overrides matter more)
        total_weight = sum(r.decayed_weight for r in records)

        # Find dominant to_method by decay-weighted count
        method_weights: Dict[str, float] = defaultdict(float)
        for r in records:
            method_weights[r.to_method] += r.decayed_weight

        best_method = max(method_weights, key=lambda m: method_weights[m])
        best_weight = method_weights[best_method]
        consistency = best_weight / total_weight

        if consistency < PREFERENCE_CONSISTENCY_THRESHOLD:
            return None

        # Scale preference confidence: stronger if consistency is higher
        # and more overrides have accumulated
        pref_confidence = min(
            MAX_METHOD_PREFERENCE_WEIGHT,
            consistency * math.log1p(total_weight) / math.log1p(20),
        )

        return LearnedMethodPreference(
            element_type=element_type,
            preferred_method=best_method,
            confidence=pref_confidence,
            based_on_overrides=int(total_weight),
        )

    def should_override_recommendation(
        self,
        element_type: str,
        vlm_recommendation: str,
    ) -> Tuple[bool, Optional[str], float]:
        """
        Decide if the learned preference should override the VLM recommendation.

        Returns:
            (should_override: bool, preferred_method: Optional[str], confidence: float)
        """
        pref = self.get_preferred_method(element_type, vlm_recommendation)
        if pref is None:
            return False, None, 0.0
        if pref.preferred_method == vlm_recommendation:
            return False, None, 0.0  # Already aligned; no override needed
        return True, pref.preferred_method, pref.confidence
# 3.2  Threshold override learning

class ThresholdOverrideLearner:
    """
    Learns when the system's default severity threshold is wrong for this user.

    Example:
      - User consistently changes text_numeric_fragment from MEDIUM to LOW
        (they don't care about 4-digit numbers).
      - After MIN_OVERRIDES_FOR_PREFERENCE consistent overrides,
        write "text_numeric_fragment": "low" to PrivacyProfile.threshold_overrides.

    This directly encodes learned preferences without retraining the VLM.
    The existing threshold_overrides dict is read by FaceRiskAssessmentTool
    and TextRiskAssessmentTool in Agent 2 Phase 1.
    """

    # Severity ordering for up/down detection
    _SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def __init__(
        self,
        threshold_overrides: Dict[str, str],  # Live reference to PrivacyProfile.threshold_overrides
        override_history: Dict[str, List[Dict]],  # MongoDB collection proxy
    ):
        self._overrides = threshold_overrides
        self._history = override_history

    def record_severity_change(
        self,
        element_category: str,  # e.g., THRESHOLD_KEY_FACE_BYSTANDER
        from_severity: str,
        to_severity: str,
        session_id: str,
    ) -> None:
        """Record that the user changed the effective severity for a category."""
        if element_category not in self._history:
            self._history[element_category] = []

        self._history[element_category].append({
            "from": from_severity.lower(),
            "to": to_severity.lower(),
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def try_update_threshold(self, element_category: str) -> Optional[str]:
        """
        Check if a stable preference exists and if so write it to threshold_overrides.

        Returns the new threshold value if updated, else None.
        """
        history = self._history.get(element_category, [])
        if len(history) < MIN_OVERRIDES_FOR_PREFERENCE:
            return None

        # Take most recent N records (don't use infinitely old data)
        recent = history[-10:]
        to_counts: Dict[str, int] = defaultdict(int)
        for record in recent:
            to_counts[record["to"]] += 1

        best_to = max(to_counts, key=lambda k: to_counts[k])
        consistency = to_counts[best_to] / len(recent)

        if consistency < PREFERENCE_CONSISTENCY_THRESHOLD:
            return None

        # Write to PrivacyProfile.threshold_overrides
        self._overrides[element_category] = best_to
        return best_to

    def get_effective_threshold(
        self, element_category: str, system_default: str
    ) -> str:
        """
        Return the effective threshold for a category, respecting learned overrides.
        Falls back to system_default if no learned preference exists.
        """
        return self._overrides.get(element_category, system_default)
# 3.3  Consent rate learning integration

class ConsentRateLearner:
    """
    Thin wrapper around existing ConsentHistory.approval_rate logic.

    The existing system in consent_identity_agent.py already uses:
      - learning.min_appearances_for_trust = 3
      - learning.trust_approval_threshold = 0.8
      - learning.risk_decay_per_approval = 0.1

    This class provides the bridge from HITL override events to the
    ConsentHistory update calls. It does NOT change the learning algorithm —
    it just makes the call site explicit.
    """

    def update_from_hitl(
        self,
        consent_history: Any,  # ConsentHistory model instance
        hitl_action: str,       # "approved" | "protected" | "ignored"
        person_id: str,
    ) -> None:
        """
        Update consent history based on a HITL decision.

        approved  → increment times_approved (used for approval_rate)
        protected → increment times_protected (used for protection_rate)
        ignored   → no update (user explicitly skipped; ambiguous signal)
        """
        consent_history.times_appeared += 1

        if hitl_action == "approved":
            consent_history.times_approved += 1
            consent_history.last_consent_decision = "approved"
        elif hitl_action == "protected":
            consent_history.times_protected += 1
            consent_history.last_consent_decision = "protected"

    def should_auto_advance_consent(
        self,
        consent_history: Any,  # ConsentHistory
        trust_threshold: float = 0.80,
        min_appearances: int = 3,
    ) -> bool:
        """
        Returns True if consent history is strong enough to skip HITL
        consent review for this person.

        Mirrors the logic in config.yaml:
          learning.trust_approval_threshold = 0.8
          learning.min_appearances_for_trust = 3
        """
        if consent_history.times_appeared < min_appearances:
            return False
        return consent_history.approval_rate >= trust_threshold
# 3.4  Unified preference manager (public API)

class PreferenceManager:
    """
    Unified interface for all three learning mechanisms.
    Coordinator calls this; individual learners are internal.

    Two construction paths:

    1. Zero-argument (in-memory, for testing and simple use):
           pm = PreferenceManager()

    2. From a PrivacyProfile (production, with MongoDB-backed stores):
           pm = PreferenceManager.from_profile(privacy_profile, mongo_store)

    Coordinator call contract (coordinator_graph.py):
        pm.record_method_override(element_type, old_method, new_method, session_id)
        pm.adapt_strategy(strategy_dict, session_id)  -> adapted dict
    """

    def __init__(
        self,
        method_learner: Optional[MethodPreferenceLearner] = None,
        threshold_learner: Optional[ThresholdOverrideLearner] = None,
        consent_learner: Optional[ConsentRateLearner] = None,
    ):
        # Allow zero-argument construction with sensible in-memory defaults
        self.method_learner: MethodPreferenceLearner = (
            method_learner
            if method_learner is not None
            else MethodPreferenceLearner(store={})
        )
        self.threshold_learner: ThresholdOverrideLearner = (
            threshold_learner
            if threshold_learner is not None
            else ThresholdOverrideLearner(threshold_overrides={}, override_history={})
        )
        self.consent_learner: ConsentRateLearner = (
            consent_learner if consent_learner is not None else ConsentRateLearner()
        )

    @classmethod
    def from_profile(
        cls,
        privacy_profile,  # PrivacyProfile instance
        method_store: Optional[Dict] = None,
        threshold_history: Optional[Dict] = None,
    ) -> "PreferenceManager":
        """
        Build from an existing PrivacyProfile.
        Pass MongoDB-backed dicts in production; plain dicts for testing.
        """
        return cls(
            method_learner=MethodPreferenceLearner(
                store=method_store if method_store is not None else {}
            ),
            threshold_learner=ThresholdOverrideLearner(
                threshold_overrides=privacy_profile.threshold_overrides,
                override_history=threshold_history if threshold_history is not None else {},
            ),
            consent_learner=ConsentRateLearner(),
        )

    def record_method_override(
        self,
        element_type: str,
        from_method: str,
        to_method: str,
        session_id: str = "",
    ) -> None:
        """Record a user HITL override of the recommended method."""
        self.method_learner.record_override(
            element_type, from_method, to_method, session_id
        )

    def adapt_strategy(
        self, strategy: Dict[str, Any], session_id: str = ""
    ) -> Dict[str, Any]:
        """
        Apply learned preferences to a strategy dict before presenting to user.
        Only modifies the method if a stable preference exists.
        The original VLM recommendation is preserved in 'original_method' for audit.

        Supports two field naming conventions used across the codebase:
          - element_type / method            (Agent 3 strategy dicts)
          - element / recommended_method     (coordinator strategy dicts)
        """
        # Resolve element type — prefer 'element_type', fall back to 'element'
        element_type = strategy.get("element_type") or strategy.get("element", "")

        # Resolve current method — prefer 'method', fall back to 'recommended_method'
        current_method = strategy.get("method") or strategy.get("recommended_method", "")

        if not element_type or not current_method:
            return strategy

        should_override, preferred, conf = (
            self.method_learner.should_override_recommendation(
                element_type, current_method
            )
        )

        if should_override and preferred:
            strategy = dict(strategy)
            strategy["original_method"] = current_method
            # Write back to whichever key was present
            if "method" in strategy:
                strategy["method"] = preferred
            else:
                strategy["recommended_method"] = preferred
            strategy["method_adapted_from_preference"] = True
            strategy["preference_confidence"] = round(conf, 3)

        return strategy

    def get_effective_threshold(
        self, element_category: str, system_default: str
    ) -> str:
        """Return the effective risk threshold considering learned overrides."""
        return self.threshold_learner.get_effective_threshold(
            element_category, system_default
        )
    # Extended convenience API

    def record_severity_override(
        self,
        element_type: str,
        old_severity: str,
        new_severity: str,
        session_id: str = "",
    ) -> None:
        """Record a user severity override for threshold learning.

        Delegates to ThresholdOverrideLearner.record_severity_change using
        element_type as the category key.  After recording, attempts to
        promote the learned threshold immediately if consistency is met.
        """
        self.threshold_learner.record_severity_change(
            element_type, old_severity, new_severity, session_id
        )
        self.threshold_learner.try_update_threshold(element_type)

    def record_consent_decision(
        self,
        person_id: str,
        approved: bool,
    ) -> None:
        """Record a consent approval/rejection for a known person.

        ConsentRateLearner operates on ConsentHistory model instances
        (managed by consent_identity_agent).  This method is a no-op stub
        that satisfies the coordinator interface; callers that hold a live
        ConsentHistory object should call consent_learner.update_from_hitl
        directly.
        """
        # Stub: the coordinator passes person_id + bool; the actual
        # ConsentHistory object lives in consent_identity_agent.  Nothing
        # to update here without that object reference.
        pass  # noqa: PIE790

    def get_learned_preferences(self) -> Dict[str, Any]:
        """Return a summary of all learned preferences for debugging/export.

        Returns a dict with two top-level keys:
          - "method_preferences": {element_type: {from→to: count, ...}, ...}
          - "threshold_overrides": the live threshold_overrides dict
        """
        # Summarise method preference store
        method_summary: Dict[str, Any] = {}
        for etype, records in self.method_learner._store.items():
            method_summary[etype] = {
                f"{r.from_method}→{r.to_method}": {
                    "count": r.count,
                    "decayed_weight": round(r.decayed_weight, 4),
                    "last_updated": r.last_updated,
                }
                for r in records
            }
        return {
            "method_preferences": method_summary,
            "threshold_overrides": dict(self.threshold_learner._overrides),
        }

    def start_session(self, session_id: str = "") -> None:
        """Called at session start — applies temporal decay to method preferences.

        Iterates over all MethodPreferenceRecord entries and multiplies their
        decayed_weight by PREFERENCE_DECAY_PER_SESSION so that older signals
        fade relative to newer ones.
        """
        for records in self.method_learner._store.values():
            for record in records:
                record.decayed_weight *= PREFERENCE_DECAY_PER_SESSION

    def reset(self) -> None:
        """Reset all learned preferences to a clean in-memory state."""
        self.method_learner = MethodPreferenceLearner(store={})
        self.threshold_learner = ThresholdOverrideLearner(
            threshold_overrides={}, override_history={}
        )
        self.consent_learner = ConsentRateLearner()
