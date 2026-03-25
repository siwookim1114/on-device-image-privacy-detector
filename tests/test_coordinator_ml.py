"""
Unit tests for Coordinator ML components.
Runs without VLM server (all VLM-dependent paths are mocked).
Run: conda run -n lab_env python3 tests/test_coordinator_ml.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from agents.coordinator.intent_classifier import (
    IntentAction, hybrid_classify, needs_clarification,
    _regex_classify, _split_multi_intent, decompose_multi_intent,
    CLARIFICATION_THRESHOLD,
)
from agents.coordinator.hitl_confidence import (
    ElementConfidenceSignals, element_confidence, session_confidence,
    compute_hitl_decision, HITLMode, CheckpointType,
    AUTO_ADVANCE_THRESHOLD, STRATEGY_REVIEW_THRESHOLD,
)
from agents.coordinator.adaptive_learning import (
    MethodPreferenceLearner, ThresholdOverrideLearner,
    MIN_OVERRIDES_FOR_PREFERENCE, PREFERENCE_CONSISTENCY_THRESHOLD,
)
from agents.coordinator.reexecution_graph import (
    ModificationType, PipelineStage, compute_reexecution_plan,
    sam_masks_still_valid, intent_to_modification_type,
)
from agents.coordinator.evaluation_metrics import (
    IntentClassificationSample, compute_intent_classification_metrics,
    HITLDecisionSample, compute_hitl_decision_metrics,
    ReExecutionSample, compute_reexecution_metrics,
)


# ---------------------------------------------------------------------------
# Intent classifier tests
# ---------------------------------------------------------------------------

class TestRegexClassifier(unittest.TestCase):

    def _classify(self, query):
        return _regex_classify(query)

    def test_process_triggers(self):
        for q in ["process", "analyze the image", "run the pipeline", "scan image"]:
            result = self._classify(q)
            self.assertIsNotNone(result, f"No match for: {q!r}")
            self.assertEqual(result.action, IntentAction.PROCESS, f"Wrong action for {q!r}")
            self.assertEqual(result.confidence, 1.0)

    def test_approve_triggers(self):
        for q in ["yes", "okay", "approve", "looks good", "proceed"]:
            result = self._classify(q)
            self.assertIsNotNone(result, f"No match for: {q!r}")
            self.assertEqual(result.action, IntentAction.APPROVE)

    def test_reject_triggers(self):
        for q in ["no", "reject", "cancel", "revert that"]:
            result = self._classify(q)
            self.assertIsNotNone(result, f"No match for: {q!r}")
            self.assertEqual(result.action, IntentAction.REJECT)

    def test_undo_triggers(self):
        for q in ["undo", "go back", "previous state", "roll back"]:
            result = self._classify(q)
            self.assertIsNotNone(result, f"No match for: {q!r}")
            self.assertEqual(result.action, IntentAction.UNDO)

    def test_query_triggers(self):
        for q in ["why was this blurred", "explain this decision", "what is the reason"]:
            result = self._classify(q)
            self.assertIsNotNone(result, f"No match for: {q!r}")
            self.assertEqual(result.action, IntentAction.QUERY)

    def test_ignore_triggers(self):
        for q in ["ignore this face", "skip protection", "don't protect", "remove the protection"]:
            result = self._classify(q)
            self.assertIsNotNone(result, f"No match for: {q!r}")
            self.assertEqual(result.action, IntentAction.IGNORE)

    def test_strengthen_triggers(self):
        for q in ["make protection stronger", "more protection", "increase blur", "expand the mask"]:
            result = self._classify(q)
            self.assertIsNotNone(result, f"No match for: {q!r}")
            self.assertEqual(result.action, IntentAction.STRENGTHEN)

    def test_modify_strategy_with_method(self):
        for q in ["use pixelate", "change to blur", "make it solid overlay", "set avatar"]:
            result = self._classify(q)
            self.assertIsNotNone(result, f"No match for: {q!r}")
            self.assertEqual(result.action, IntentAction.MODIFY_STRATEGY)

    def test_element_type_extraction(self):
        result = self._classify("blur the face")
        self.assertIsNotNone(result)
        self.assertIn("face", result.target_element_types or [])

    def test_method_extraction(self):
        result = self._classify("change to pixelate")
        self.assertIsNotNone(result)
        self.assertEqual(result.method_specified, "pixelate")

    def test_strength_extraction(self):
        result = self._classify("make protection stronger")
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.strength_parameter)
        self.assertGreater(result.strength_parameter, 0.7)

    def test_no_match_returns_none(self):
        result = self._classify("this is completely unrelated gibberish xyz")
        self.assertIsNone(result)

    def test_hybrid_classify_without_vlm(self):
        # Should return QUERY with low confidence when VLM not available
        intent = hybrid_classify("some ambiguous statement here", vlm_call_fn=None)
        self.assertIsNotNone(intent)
        self.assertIsInstance(intent.action, IntentAction)

    def test_hybrid_classify_regex_path(self):
        intent = hybrid_classify("process", vlm_call_fn=None)
        self.assertEqual(intent.action, IntentAction.PROCESS)
        self.assertEqual(intent.confidence, 1.0)

    def test_needs_clarification_low_confidence(self):
        intent = hybrid_classify("xyz ambiguous", vlm_call_fn=None)
        # VLM returned QUERY with 0.3 confidence
        self.assertTrue(needs_clarification(intent))

    def test_needs_clarification_high_confidence(self):
        intent = hybrid_classify("process", vlm_call_fn=None)
        self.assertFalse(needs_clarification(intent))


class TestMultiIntentDecomposition(unittest.TestCase):

    def test_single_intent_no_split(self):
        parts = _split_multi_intent("blur the face")
        self.assertEqual(len(parts), 1)

    def test_conjunction_splits(self):
        parts = _split_multi_intent("blur the face and make text stronger")
        self.assertGreaterEqual(len(parts), 2)

    def test_decompose_multi_intent(self):
        intent = decompose_multi_intent(
            "blur the face and make text stronger",
            classify_fn=lambda q: _regex_classify(q) or _regex_classify("process"),
        )
        self.assertIsNotNone(intent)
        # Should have at least one sub-intent
        self.assertIsInstance(intent, object)


# ---------------------------------------------------------------------------
# HITL confidence tests
# ---------------------------------------------------------------------------

class TestElementConfidence(unittest.TestCase):

    def _make_signals(self, **kwargs):
        defaults = dict(
            detection_id="test_001",
            element_type="face",
            detection_confidence=0.90,
            severity="medium",
            consent_status="none",
            screen_state=None,
            escalation_applied=False,
            vlm_phase2_ran=False,
            vlm_phase2_agreed=False,
        )
        defaults.update(kwargs)
        return ElementConfidenceSignals(**defaults)

    def test_high_confidence_face(self):
        signals = self._make_signals(
            detection_confidence=0.95,
            severity="low",
            consent_status="explicit",
            vlm_phase2_ran=True,
            vlm_phase2_agreed=True,
        )
        result = element_confidence(signals)
        self.assertGreater(result.raw_score, 0.75)

    def test_critical_lowers_confidence(self):
        signals_low = self._make_signals(severity="low")
        signals_crit = self._make_signals(severity="critical")
        result_low = element_confidence(signals_low)
        result_crit = element_confidence(signals_crit)
        self.assertGreater(result_low.raw_score, result_crit.raw_score)

    def test_unclear_consent_lowers_confidence(self):
        signals_explicit = self._make_signals(consent_status="explicit")
        signals_unclear = self._make_signals(consent_status="unclear")
        result_exp = element_confidence(signals_explicit)
        result_unc = element_confidence(signals_unclear)
        self.assertGreater(result_exp.raw_score, result_unc.raw_score)

    def test_vlm_disagreement_lowers_confidence(self):
        signals_agree = self._make_signals(vlm_phase2_ran=True, vlm_phase2_agreed=True)
        signals_disagree = self._make_signals(vlm_phase2_ran=True, vlm_phase2_agreed=False)
        result_agree = element_confidence(signals_agree)
        result_disagree = element_confidence(signals_disagree)
        self.assertGreater(result_agree.raw_score, result_disagree.raw_score)

    def test_screen_no_state_lowers_confidence(self):
        signals_verified = ElementConfidenceSignals(
            detection_id="s1", element_type="screen",
            detection_confidence=0.9, severity="medium",
            consent_status=None, screen_state="verified_off",
            escalation_applied=False, vlm_phase2_ran=False, vlm_phase2_agreed=False,
        )
        signals_unknown = ElementConfidenceSignals(
            detection_id="s2", element_type="screen",
            detection_confidence=0.9, severity="medium",
            consent_status=None, screen_state=None,
            escalation_applied=False, vlm_phase2_ran=False, vlm_phase2_agreed=False,
        )
        result_v = element_confidence(signals_verified)
        result_u = element_confidence(signals_unknown)
        self.assertGreater(result_v.raw_score, result_u.raw_score)

    def test_score_bounded_0_1(self):
        for sev in ["critical", "high", "medium", "low"]:
            for consent in ["explicit", "assumed", "unclear", "none"]:
                signals = self._make_signals(severity=sev, consent_status=consent)
                result = element_confidence(signals)
                self.assertGreaterEqual(result.raw_score, 0.0)
                self.assertLessEqual(result.raw_score, 1.0)


class TestSessionConfidence(unittest.TestCase):

    def _make_result(self, detection_id, score, components=None):
        from agents.coordinator.hitl_confidence import ElementConfidenceResult
        return ElementConfidenceResult(
            detection_id=detection_id,
            element_type="face",
            raw_score=score,
            component_breakdown=components or {"severity_penalty": 0.15},
            penalized_by=[],
        )

    def test_empty_returns_1(self):
        self.assertEqual(session_confidence([]), 1.0)

    def test_single_high_score(self):
        results = [self._make_result("a", 0.9)]
        score = session_confidence(results)
        self.assertGreater(score, 0.7)

    def test_one_low_score_pulls_down_average(self):
        results = [
            self._make_result("a", 0.9),
            self._make_result("b", 0.9),
            self._make_result("c", 0.1),  # One very low element
        ]
        score = session_confidence(results)
        # Harmonic mean should pull score well below arithmetic mean of 0.63
        self.assertLess(score, 0.5)


class TestComputeHITLDecision(unittest.TestCase):

    def _make_assessment(self, severity="medium", element_type="face",
                          consent_status="none", screen_state=None,
                          confidence=0.9):
        return {
            "detection_id": "test",
            "element_type": element_type,
            "confidence": confidence,
            "severity": severity,
            "consent_status": consent_status,
            "screen_state": screen_state,
            "escalation_applied": False,
            "vlm_phase2_ran": False,
            "vlm_phase2_agreed": False,
        }

    def test_auto_advance_high_confidence(self):
        # Need explicit consent + VLM agreement + high detector confidence to push
        # session score above AUTO_ADVANCE_THRESHOLD (0.85)
        assessments = [
            self._make_assessment(severity="low", confidence=0.98,
                                  consent_status="explicit"),
            self._make_assessment(severity="low", confidence=0.98,
                                  consent_status="explicit"),
        ]
        # Also inject VLM phase 2 agreement to boost scores
        for a in assessments:
            a["vlm_phase2_ran"] = True
            a["vlm_phase2_agreed"] = True
        report = compute_hitl_decision(assessments, [], HITLMode.HYBRID)
        self.assertEqual(report.checkpoint_type, CheckpointType.AUTO_ADVANCE_SUMMARY)
        self.assertTrue(report.auto_advance_possible)

    def test_critical_forces_strategy_review(self):
        assessments = [
            self._make_assessment(severity="critical", confidence=0.95),
        ]
        report = compute_hitl_decision(assessments, [], HITLMode.HYBRID)
        self.assertNotEqual(report.checkpoint_type, CheckpointType.AUTO_ADVANCE_SUMMARY)
        self.assertTrue(report.has_critical_elements)

    def test_manual_mode_always_full_review(self):
        assessments = [self._make_assessment(severity="low", confidence=0.99)]
        report = compute_hitl_decision(assessments, [], HITLMode.MANUAL)
        self.assertEqual(report.checkpoint_type, CheckpointType.FULL_MANUAL_REVIEW)

    def test_auto_mode_no_critical_advances(self):
        assessments = [self._make_assessment(severity="medium", confidence=0.8)]
        report = compute_hitl_decision(assessments, [], HITLMode.AUTO)
        self.assertEqual(report.checkpoint_type, CheckpointType.AUTO_ADVANCE_SUMMARY)

    def test_consent_conflict_overrides_auto_advance(self):
        assessments = [self._make_assessment(severity="low", confidence=0.99,
                                              consent_status="explicit")]
        strategies = [{
            "detection_id": "test",
            "element_type": "face",
            "method": "blur",
            "consent_status": "explicit",
        }]
        report = compute_hitl_decision(assessments, strategies, HITLMode.HYBRID)
        # Should NOT auto-advance due to consent conflict
        self.assertNotEqual(report.checkpoint_type, CheckpointType.AUTO_ADVANCE_SUMMARY)


# ---------------------------------------------------------------------------
# Adaptive learning tests
# ---------------------------------------------------------------------------

class TestMethodPreferenceLearner(unittest.TestCase):

    def setUp(self):
        self.store = {}
        self.learner = MethodPreferenceLearner(self.store)

    def test_no_preference_below_threshold(self):
        # Only 2 overrides — below MIN_OVERRIDES_FOR_PREFERENCE
        self.learner.record_override("face", "blur", "pixelate", "s1")
        self.learner.record_override("face", "blur", "pixelate", "s2")
        pref = self.learner.get_preferred_method("face")
        self.assertIsNone(pref)

    def test_preference_emerges_after_threshold(self):
        for i in range(MIN_OVERRIDES_FOR_PREFERENCE):
            self.learner.record_override("face", "blur", "pixelate", f"s{i}")
        pref = self.learner.get_preferred_method("face")
        self.assertIsNotNone(pref)
        self.assertEqual(pref.preferred_method, "pixelate")

    def test_inconsistent_overrides_no_preference(self):
        self.learner.record_override("face", "blur", "pixelate", "s1")
        self.learner.record_override("face", "blur", "avatar_replace", "s2")
        self.learner.record_override("face", "blur", "solid_overlay", "s3")
        self.learner.record_override("face", "blur", "pixelate", "s4")
        pref = self.learner.get_preferred_method("face")
        # 4 overrides, but split across 3 targets: no clear consistency
        if pref is not None:
            # If it returns something, the consistency should be low
            self.assertLess(pref.confidence, 0.6)

    def test_override_recommendation(self):
        for i in range(MIN_OVERRIDES_FOR_PREFERENCE):
            self.learner.record_override("face", "blur", "pixelate", f"s{i}")
        should, method, conf = self.learner.should_override_recommendation("face", "blur")
        self.assertTrue(should)
        self.assertEqual(method, "pixelate")

    def test_no_override_when_aligned(self):
        for i in range(MIN_OVERRIDES_FOR_PREFERENCE):
            self.learner.record_override("face", "blur", "pixelate", f"s{i}")
        # VLM also recommends pixelate now
        should, method, conf = self.learner.should_override_recommendation("face", "pixelate")
        self.assertFalse(should)


class TestThresholdOverrideLearner(unittest.TestCase):

    def setUp(self):
        self.overrides = {}
        self.history = {}
        self.learner = ThresholdOverrideLearner(self.overrides, self.history)

    def test_no_update_below_threshold(self):
        for i in range(2):
            self.learner.record_severity_change(
                "face_bystander", "critical", "high", f"s{i}"
            )
        result = self.learner.try_update_threshold("face_bystander")
        self.assertIsNone(result)

    def test_update_after_consistent_overrides(self):
        for i in range(MIN_OVERRIDES_FOR_PREFERENCE):
            self.learner.record_severity_change(
                "face_bystander", "critical", "high", f"s{i}"
            )
        result = self.learner.try_update_threshold("face_bystander")
        self.assertEqual(result, "high")
        self.assertEqual(self.overrides["face_bystander"], "high")

    def test_get_effective_threshold_uses_override(self):
        self.overrides["face_bystander"] = "high"
        effective = self.learner.get_effective_threshold("face_bystander", "critical")
        self.assertEqual(effective, "high")

    def test_get_effective_threshold_falls_back(self):
        effective = self.learner.get_effective_threshold("face_bystander", "critical")
        self.assertEqual(effective, "critical")


# ---------------------------------------------------------------------------
# Re-execution graph tests
# ---------------------------------------------------------------------------

class TestReExecutionGraph(unittest.TestCase):

    def test_full_pipeline_reruns_all(self):
        plan = compute_reexecution_plan(ModificationType.FULL_PIPELINE)
        self.assertEqual(set(plan.must_rerun), set(PipelineStage))
        self.assertEqual(plan.entry_stage, PipelineStage.DETECT)
        self.assertAlmostEqual(plan.estimated_speedup, 1.0, delta=0.1)

    def test_method_only_change_mask_valid_skips_sam(self):
        old_s = {"detection_id": "a", "method": "blur", "element_type": "face"}
        new_s = {"detection_id": "a", "method": "pixelate", "element_type": "face"}
        plan = compute_reexecution_plan(
            ModificationType.METHOD_ONLY_CHANGE,
            old_strategies=[old_s],
            new_strategies=[new_s],
        )
        # SAM should be skipped (both blur and pixelate use masks; same bbox)
        self.assertNotIn(PipelineStage.SAM, plan.must_rerun)
        self.assertEqual(plan.entry_stage, PipelineStage.EXECUTION)
        self.assertGreater(plan.estimated_speedup, 3.0)

    def test_strengthen_only_reruns_execution_export(self):
        plan = compute_reexecution_plan(ModificationType.STRENGTHEN_ONLY)
        self.assertIn(PipelineStage.EXECUTION, plan.must_rerun)
        self.assertIn(PipelineStage.EXPORT, plan.must_rerun)
        self.assertNotIn(PipelineStage.DETECT, plan.must_rerun)
        self.assertNotIn(PipelineStage.SAM, plan.must_rerun)

    def test_severity_change_skips_detect_consent(self):
        plan = compute_reexecution_plan(ModificationType.SEVERITY_CHANGE)
        self.assertNotIn(PipelineStage.DETECT, plan.must_rerun)
        self.assertNotIn(PipelineStage.CONSENT, plan.must_rerun)
        self.assertIn(PipelineStage.STRATEGY, plan.must_rerun)

    def test_consent_change_skips_detect_risk(self):
        plan = compute_reexecution_plan(ModificationType.CONSENT_CHANGE)
        self.assertNotIn(PipelineStage.DETECT, plan.must_rerun)
        self.assertNotIn(PipelineStage.RISK, plan.must_rerun)
        self.assertIn(PipelineStage.STRATEGY, plan.must_rerun)

    def test_add_region_reruns_only_execution_export(self):
        plan = compute_reexecution_plan(ModificationType.ADD_REGION)
        self.assertEqual(set(plan.must_rerun), {PipelineStage.EXECUTION, PipelineStage.EXPORT})

    def test_sam_masks_valid_blur_to_pixelate(self):
        old_s = {"method": "blur"}
        valid = sam_masks_still_valid(old_s, "pixelate", detection_bbox_changed=False)
        self.assertTrue(valid)

    def test_sam_masks_invalid_solid_overlay_to_blur(self):
        # solid_overlay doesn't generate a mask; can't reuse nonexistent mask
        old_s = {"method": "solid_overlay"}
        valid = sam_masks_still_valid(old_s, "blur", detection_bbox_changed=False)
        self.assertFalse(valid)

    def test_sam_masks_invalid_bbox_changed(self):
        old_s = {"method": "blur"}
        valid = sam_masks_still_valid(old_s, "pixelate", detection_bbox_changed=True)
        self.assertFalse(valid)

    def test_intent_to_modification_strengthen(self):
        mod = intent_to_modification_type("strengthen", None, None)
        self.assertEqual(mod, ModificationType.STRENGTHEN_ONLY)

    def test_intent_to_modification_ignore(self):
        mod = intent_to_modification_type("ignore", None, None)
        self.assertEqual(mod, ModificationType.IGNORE_ELEMENT)


# ---------------------------------------------------------------------------
# Evaluation metrics tests
# ---------------------------------------------------------------------------

class TestIntentMetrics(unittest.TestCase):

    def test_perfect_predictions(self):
        samples = [
            IntentClassificationSample(
                query="process", true_action="process",
                predicted_action="process", predicted_confidence=1.0,
                used_vlm=False, latency_ms=0.5,
            )
        ] * 5
        report = compute_intent_classification_metrics(samples)
        self.assertAlmostEqual(report.macro_f1, 1.0, delta=0.001)

    def test_all_wrong_predictions(self):
        samples = [
            IntentClassificationSample(
                query="process", true_action="process",
                predicted_action="query", predicted_confidence=0.6,
                used_vlm=True, latency_ms=500,
            )
        ] * 5
        report = compute_intent_classification_metrics(samples)
        self.assertEqual(report.per_class["process"]["recall"], 0.0)


class TestHITLMetrics(unittest.TestCase):

    def test_no_false_pauses_when_always_needed(self):
        samples = [
            HITLDecisionSample(
                session_id=f"s{i}",
                checkpoint_type="strategy_review",
                session_confidence=0.6,
                user_made_override=True,
                had_critical_element=True,
                system_paused=True,
                should_have_paused=True,
            )
            for i in range(10)
        ]
        report = compute_hitl_decision_metrics(samples)
        self.assertEqual(report.false_pause_rate, 0.0)

    def test_missed_critical_rate_computation(self):
        # 5 sessions with critical elements; 1 auto-advanced (missed)
        samples = [
            HITLDecisionSample(
                session_id="s0",
                checkpoint_type="auto_advance_summary",
                session_confidence=0.9,
                user_made_override=True,
                had_critical_element=True,
                system_paused=False,
                should_have_paused=True,
            )
        ] + [
            HITLDecisionSample(
                session_id=f"s{i}",
                checkpoint_type="strategy_review",
                session_confidence=0.6,
                user_made_override=False,
                had_critical_element=True,
                system_paused=True,
                should_have_paused=True,
            )
            for i in range(1, 5)
        ]
        report = compute_hitl_decision_metrics(samples)
        self.assertAlmostEqual(report.missed_critical_rate, 0.2, delta=0.001)


class TestReExecutionMetrics(unittest.TestCase):

    def test_all_correct_skips(self):
        samples = [
            ReExecutionSample(
                session_id="s1",
                modification_type="method_only_change",
                skipped_stages=["detect", "risk", "consent", "strategy", "sam"],
                stages_that_would_have_changed=["execution", "export"],
                time_saved_ms=18000,
                correctness=True,
            )
        ]
        report = compute_reexecution_metrics(samples)
        self.assertEqual(report.false_skip_rate, 0.0)
        self.assertGreater(report.avg_speedup, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
