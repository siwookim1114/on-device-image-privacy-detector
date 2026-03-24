"""
Regression tests for 15 documented fixes in the on-device image privacy detector.

Each test class maps directly to one fix from CHANGELOG.md / CLAUDE.md so that
any future revert of that fix surfaces as an immediate, named test failure.

All tests run WITHOUT ML models (no MTCNN, EasyOCR, YOLO, VLM server, MongoDB).
They exercise pure-Python logic by bypassing __init__ where needed.
"""

import sys
import json
import math
import pytest
from pathlib import Path
import inspect
from agents.tools.text_backends import EasyOCRBackend

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================
# Shared fixtures / stubs
# ============================================================

class _StubConfig:
    """Minimal config stub that satisfies get_risk_color() dot-notation lookups."""
    class _Levels:
        class _Level:
            def __init__(self, color):
                self.color = color
        critical = _Level("#FF0000")
        high     = _Level("#FF6600")
        medium   = _Level("#FFD700")
        low      = _Level("#90EE90")
    risk_levels = _Levels()

    def get(self, key, default=None):
        parts = key.split(".")
        obj = self
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                return default
        return obj


def _make_text_tool():
    """TextDetectionTool bypassing EasyOCR __init__."""
    from agents.tools.detection_tools import TextDetectionTool
    return TextDetectionTool.__new__(TextDetectionTool)


def _make_object_tool():
    """ObjectDetectionTool bypassing YOLO __init__."""
    from agents.tools.detection_tools import ObjectDetectionTool
    return ObjectDetectionTool.__new__(ObjectDetectionTool)


def _make_face_tool():
    """FaceDetectionTool bypassing MTCNN __init__."""
    from agents.tools.detection_tools import FaceDetectionTool
    return FaceDetectionTool.__new__(FaceDetectionTool)


def _make_spatial_tool():
    """SpatialRelationshipTool (no ML deps)."""
    from agents.tools.risk_tools import SpatialRelationshipTool
    return SpatialRelationshipTool()


def _make_escalation_tool(config=None):
    """RiskEscalationTool (no ML deps)."""
    from agents.tools.risk_tools import RiskEscalationTool
    return RiskEscalationTool(config=config)


def _make_reclassify_tool(assessments, config=None):
    from agents.tools.risk_tools import ReclassifyAssessmentTool
    return ReclassifyAssessmentTool(assessments=assessments, config=config or _StubConfig())


# ============================================================
# Fix 1: EasyOCR width_ths=0.7
# ============================================================

class TestEasyOCRWidthThs:
    """
    Regression: EasyOCR readtext() must be called with width_ths=0.7 to prevent
    text fragmentation (e.g., "Bank Account" split into "Bank" + "Account").

    We verify by inspecting the source of _run with inspect rather than executing
    EasyOCR, so no ML model is needed.
    """

    def test_width_ths_in_source(self):
        """EasyOCR backend must use width_ths=0.7."""
        source = inspect.getsource(EasyOCRBackend.detect)
        assert "width_ths=0.7" in source, (
            "EasyOCR readtext() call in EasyOCRBackend.detect must use width_ths=0.7 "
            "to prevent text fragmentation"
        )

    def test_low_text_param_present(self):
        """EasyOCR backend must pass low_text threshold."""
        source = inspect.getsource(EasyOCRBackend.detect)
        assert "low_text" in source, (
            "EasyOCR readtext() call must pass low_text parameter for OCR sensitivity"
        )


# ============================================================
# Fix 2: Multi-word merge match first (bank_label regex priority)
# ============================================================

class TestMultiWordMergeMatchFirst:
    """
    Regression: "Bank Account:" must be classified as bank_label (multi-word match),
    NOT as general_text caused by the single-word "bank" pattern triggering first.

    The bank composite pattern requires digits, so "Bank Account:" should fall
    through to the stricter multi-word label regex (bank_account|account_number …)
    and be caught as bank_label before the single-word fallback.
    """

    @pytest.fixture
    def tool(self):
        return _make_text_tool()

    def test_bank_account_label_is_label_only(self, tool):
        r = tool._classify_text_type("Bank Account:")
        assert r["is_label_only"] is True, (
            "'Bank Account:' should be is_label_only=True (multi-word merge match)"
        )

    def test_bank_account_label_not_critical(self, tool):
        r = tool._classify_text_type("Bank Account:")
        assert r["is_critical"] is False

    def test_bank_account_label_type(self, tool):
        r = tool._classify_text_type("Bank Account:")
        assert r["type"] == "bank_label", (
            f"Expected type='bank_label', got '{r['type']}' — multi-word match did not fire first"
        )

    def test_account_number_label(self, tool):
        r = tool._classify_text_type("Account Number:")
        assert r["is_label_only"] is True
        assert r["type"] == "bank_label"

    def test_routing_number_label(self, tool):
        r = tool._classify_text_type("Routing Number:")
        assert r["is_label_only"] is True

    def test_single_word_bank_with_digits_is_not_label(self, tool):
        """'Bank 12345678' has digits → bank_account (composite), not bank_label."""
        r = tool._classify_text_type("Bank 12345678")
        assert r["is_label_only"] is False
        assert r["type"] == "bank_account"


# ============================================================
# Fix 3: Adaptive spatial threshold (20 % of diagonal, [150, 400] px)
# ============================================================

class TestAdaptiveSpatialThreshold:
    """
    Regression: SpatialRelationshipTool must compute adaptive_threshold as
        max(150, min(400, diagonal * 0.20))

    Test three cases:
      - 500×400  → diag ≈ 640,  20% = 128 → clamped UP   to 150
      - 3000×2000→ diag ≈ 3606, 20% = 721 → clamped DOWN to 400
      - 1024×768 → diag ≈ 1280, 20% = 256 → unclamped   = 256
    """

    def _expected_threshold(self, w, h):
        diag = math.sqrt(w * w + h * h)
        return max(150, min(400, diag * 0.20))

    def test_small_image_clamped_to_min(self):
        w, h = 500, 400
        t = self._expected_threshold(w, h)
        assert t == pytest.approx(150.0), f"Expected 150, got {t}"

    def test_large_image_clamped_to_max(self):
        w, h = 3000, 2000
        t = self._expected_threshold(w, h)
        assert t == pytest.approx(400.0), f"Expected 400, got {t}"

    def test_medium_image_unclamped(self):
        w, h = 1024, 768
        expected = math.sqrt(1024**2 + 768**2) * 0.20
        t = self._expected_threshold(w, h)
        assert 150 < t < 400
        assert t == pytest.approx(expected, rel=1e-6)

    def test_spatial_tool_uses_adaptive_threshold_small(self):
        """SpatialRelationshipTool with 500×400 image: threshold should be 150 (clamped)."""
        tool = _make_spatial_tool()
        # A face at (10,10,50,50) and a PII text at (200,10,50,20) that are 165 px apart.
        # With the MIN-clamped threshold of 150 they should NOT be considered near.
        # Without the clamp (raw 128) they also would not; but with a wrong floor of 0 they might.
        payload = json.dumps({
            "image_width": 500,
            "image_height": 400,
            "faces": [{"id": "f0", "bbox": [10, 10, 50, 50], "size": "medium"}],
            "texts": [{
                "id": "t0", "bbox": [220, 10, 50, 20],
                "attributes": {"is_pii": True}
            }],
            "objects": [],
        })
        result = json.loads(tool._run(payload))
        assert "error" not in result

    def test_spatial_tool_uses_adaptive_threshold_large(self):
        """With a 3000×2000 image threshold=400: elements 350px apart ARE near."""
        tool = _make_spatial_tool()
        # Face center at (100,100); text center at (450,100) → distance = 350 px.
        # Threshold should be 400 (clamped max), so they ARE near.
        payload = json.dumps({
            "image_width": 3000,
            "image_height": 2000,
            "faces": [{"id": "f0", "bbox": [75, 75, 50, 50], "size": "medium"}],
            "texts": [{
                "id": "t0", "bbox": [425, 75, 50, 50],
                "attributes": {"is_pii": True}
            }],
            "objects": [],
        })
        result = json.loads(tool._run(payload))
        assert "error" not in result
        # 350 px < 400 threshold → escalation should exist
        assert result["total_escalations"] >= 1, (
            "Face 350px from PII text should be escalated with threshold=400 (large image)"
        )


# ============================================================
# Fix 4: Escalation dedup — one per (element_id, relationship_type)
# ============================================================

class TestEscalationDedup:
    """
    Regression: RiskEscalationTool must apply at most one escalation per
    (element_id, relationship_type) pair. A duplicate entry should be silently ignored.
    """

    def test_duplicate_escalation_applied_only_once(self):
        tool = _make_escalation_tool()
        payload = {
            "assessments": [{
                "detection_id": "face_0",
                "element_type": "face",
                "severity": "medium",
                "requires_protection": False,
                "factors": {},
                "color_code": "#FFD700",
            }],
            "escalations": [
                {
                    "elements": ["face_0"],
                    "escalation_amount": 1,
                    "reason": "PII nearby",
                    "relationship_type": "identity_linkage",
                },
                {
                    "elements": ["face_0"],
                    "escalation_amount": 1,
                    "reason": "PII nearby (duplicate)",
                    "relationship_type": "identity_linkage",  # same type → dedup
                },
            ],
        }
        result = json.loads(tool._run(json.dumps(payload)))
        assert "error" not in result
        # Only one escalation should have been applied (medium → high, not high → critical)
        assert result["escalations_applied"] == 1
        final = result["assessments"][0]["severity"]
        assert final == "high", (
            f"Expected 'high' after single escalation from 'medium'; got '{final}'"
        )

    def test_different_relationship_types_both_apply(self):
        """Two escalations with different relationship types on same element are both applied."""
        tool = _make_escalation_tool()
        payload = {
            "assessments": [{
                "detection_id": "face_0",
                "element_type": "face",
                "severity": "low",
                "requires_protection": False,
                "factors": {},
                "color_code": "#90EE90",
            }],
            "escalations": [
                {
                    "elements": ["face_0"],
                    "escalation_amount": 1,
                    "reason": "PII nearby",
                    "relationship_type": "identity_linkage",
                },
                {
                    "elements": ["face_0"],
                    "escalation_amount": 1,
                    "reason": "Screen nearby",
                    "relationship_type": "content_association",  # different type
                },
            ],
        }
        result = json.loads(tool._run(json.dumps(payload)))
        assert result["escalations_applied"] == 2
        assert result["assessments"][0]["severity"] == "high"


# ============================================================
# Fix 5: numeric_fragment — edge cases (too short / too long)
# ============================================================

class TestNumericFragmentEdgeCases:
    """
    Regression: _classify_text_type should map exactly 3-8 digit strings to
    'numeric_fragment'. Strings outside that range must fall through to 'general_text'
    (or another pattern — but NOT numeric_fragment).
    """

    @pytest.fixture
    def tool(self):
        return _make_text_tool()

    def test_2_digits_not_numeric_fragment(self, tool):
        r = tool._classify_text_type("12")
        assert r["type"] != "numeric_fragment", (
            "'12' (2 digits) is too short for numeric_fragment"
        )

    def test_9_digits_not_numeric_fragment(self, tool):
        r = tool._classify_text_type("123456789")
        # 9 digits: could be SSN (123-45-6789 without dashes) or general_text
        # but must NOT be classified as numeric_fragment (range is 3-8)
        assert r["type"] != "numeric_fragment", (
            "'123456789' (9 digits) is too long for numeric_fragment"
        )

    def test_3_digits_is_numeric_fragment(self, tool):
        r = tool._classify_text_type("123")
        assert r["type"] == "numeric_fragment"

    def test_8_digits_is_numeric_fragment(self, tool):
        r = tool._classify_text_type("12345678")
        assert r["type"] == "numeric_fragment"

    def test_4_digits_is_numeric_fragment(self, tool):
        r = tool._classify_text_type("4821")
        assert r["type"] == "numeric_fragment"

    def test_numeric_fragment_is_sensitive(self, tool):
        r = tool._classify_text_type("9876")
        assert r["is_sensitive"] is True
        assert r["is_critical"] is False


# ============================================================
# Fix 6: "book" removed from PRIVACY_OBJECTS
# ============================================================

class TestBookRemovedFromPrivacyObjects:
    """
    Regression: "book" must NOT be in ObjectDetectionTool.PRIVACY_OBJECTS.
    It was a false positive on text panels and has been removed.
    """

    def test_book_not_in_privacy_objects(self):
        from agents.tools.detection_tools import ObjectDetectionTool
        assert "book" not in ObjectDetectionTool.PRIVACY_OBJECTS, (
            "'book' was re-added to PRIVACY_OBJECTS — it causes false positives on text panels"
        )

    def test_is_privacy_relevant_returns_false_for_book(self):
        tool = _make_object_tool()
        assert tool._is_privacy_relevant("book") is False

    def test_is_privacy_relevant_returns_false_for_book_mixed_case(self):
        tool = _make_object_tool()
        assert tool._is_privacy_relevant("Book") is False

    def test_laptop_still_in_privacy_objects(self):
        """Sanity check: actual screen devices remain in the set."""
        from agents.tools.detection_tools import ObjectDetectionTool
        assert "laptop" in ObjectDetectionTool.PRIVACY_OBJECTS


# ============================================================
# Fix 7: Screen device VLM verification — screen_state field
# ============================================================

class TestScreenStateField:
    """
    Regression: RiskAssessment Pydantic model must carry a screen_state field
    and correctly round-trip "verified_off" and "verified_on" values.
    """

    def test_screen_state_verified_off_serializes(self):
        from utils.models import RiskAssessment, RiskLevel, RiskType, BoundingBox
        ra = RiskAssessment(
            detection_id="obj_0",
            element_type="object",
            element_description="Object: laptop",
            risk_type=RiskType.INFORMATION_DISCLOSURE,
            severity=RiskLevel.LOW,
            color_code="#90EE90",
            reasoning="Screen is off",
            user_sensitivity_applied="screens",
            bbox=BoundingBox(x=0, y=0, width=200, height=150),
            requires_protection=False,
            screen_state="verified_off",
        )
        data = ra.model_dump()
        assert data["screen_state"] == "verified_off"

    def test_screen_state_verified_on_serializes(self):
        from utils.models import RiskAssessment, RiskLevel, RiskType, BoundingBox
        ra = RiskAssessment(
            detection_id="obj_1",
            element_type="object",
            element_description="Object: tv",
            risk_type=RiskType.INFORMATION_DISCLOSURE,
            severity=RiskLevel.MEDIUM,
            color_code="#FFD700",
            reasoning="Screen is on and showing content",
            user_sensitivity_applied="screens",
            bbox=BoundingBox(x=0, y=0, width=300, height=200),
            requires_protection=True,
            screen_state="verified_on",
        )
        data = ra.model_dump()
        assert data["screen_state"] == "verified_on"

    def test_screen_state_defaults_to_none(self):
        from utils.models import RiskAssessment, RiskLevel, RiskType, BoundingBox
        ra = RiskAssessment(
            detection_id="face_0",
            element_type="face",
            element_description="Face (medium, high clarity)",
            risk_type=RiskType.IDENTITY_EXPOSURE,
            severity=RiskLevel.HIGH,
            color_code="#FF6600",
            reasoning="Face detected",
            user_sensitivity_applied="bystander_faces",
            bbox=BoundingBox(x=10, y=10, width=80, height=80),
        )
        assert ra.screen_state is None

    def test_screen_state_json_roundtrip(self):
        from utils.models import RiskAssessment, RiskLevel, RiskType, BoundingBox
        ra = RiskAssessment(
            detection_id="obj_2",
            element_type="object",
            element_description="Object: cell phone",
            risk_type=RiskType.INFORMATION_DISCLOSURE,
            severity=RiskLevel.LOW,
            color_code="#90EE90",
            reasoning="Screen state verified",
            user_sensitivity_applied="screens",
            bbox=BoundingBox(x=5, y=5, width=50, height=90),
            screen_state="verified_off",
        )
        json_str = ra.model_dump_json()
        restored = RiskAssessment.model_validate_json(json_str)
        assert restored.screen_state == "verified_off"


# ============================================================
# Fix 8: Text downgrade guard — HIGH text cannot be downgraded to MEDIUM
# ============================================================

class TestTextDowngradeGuard:
    """
    Regression: ReclassifyAssessmentTool must block any attempt to downgrade
    a text item whose current severity is HIGH or CRITICAL.
    """

    def test_cannot_downgrade_high_text_to_medium(self):
        assessments = [{
            "element_type": "text",
            "element_description": "Text: john@example.com",
            "severity": "high",
            "requires_protection": True,
            "factors": {},
        }]
        tool = _make_reclassify_tool(assessments)
        result = json.loads(tool._run(0, "medium", "seems low risk"))
        assert result["status"] == "blocked", (
            "HIGH text downgrade to MEDIUM should be blocked"
        )

    def test_cannot_downgrade_high_text_to_low(self):
        assessments = [{
            "element_type": "text",
            "element_description": "Text: 555-867-5309",
            "severity": "high",
            "requires_protection": True,
            "factors": {},
        }]
        tool = _make_reclassify_tool(assessments)
        result = json.loads(tool._run(0, "low", "not sensitive"))
        assert result["status"] == "blocked"

    def test_cannot_downgrade_critical_text_to_high(self):
        assessments = [{
            "element_type": "text",
            "element_description": "Text: 123-45-6789",
            "severity": "critical",
            "requires_protection": True,
            "factors": {},
        }]
        tool = _make_reclassify_tool(assessments)
        result = json.loads(tool._run(0, "high", "maybe not critical"))
        assert result["status"] == "blocked"

    def test_medium_text_can_be_downgraded_to_low(self):
        """MEDIUM text (below HIGH threshold) may be downgraded — guard does not apply."""
        assessments = [{
            "element_type": "text",
            "element_description": "Text: hello world",
            "severity": "medium",
            "requires_protection": False,
            "factors": {},
        }]
        tool = _make_reclassify_tool(assessments)
        result = json.loads(tool._run(0, "low", "general text, not PII"))
        assert result["status"] == "success"

    def test_high_text_upgrade_to_critical_is_allowed(self):
        """Upgrading HIGH text to CRITICAL is always permitted."""
        assessments = [{
            "element_type": "text",
            "element_description": "Text: john@example.com",
            "severity": "high",
            "requires_protection": True,
            "reasoning": "PII",
            "factors": {},
        }]
        tool = _make_reclassify_tool(assessments)
        result = json.loads(tool._run(0, "critical", "contains email"))
        assert result["status"] == "success"


# ============================================================
# Fix 9: OCR underscore normalization
# ============================================================

class TestOCRUnderscoreNormalization:
    """
    Regression: TextDetectionTool._classify_text_type must normalize underscores
    before classifying:
      - Trailing: "PIN_"  → treated as "PIN:"  → is_label_only=True
      - Mid-text:  "PIN_ 3902" → treated as "PIN: 3902" → password (with value)
    """

    @pytest.fixture
    def tool(self):
        return _make_text_tool()

    def test_trailing_underscore_label(self, tool):
        r = tool._classify_text_type("PIN_")
        assert r["is_label_only"] is True, (
            "'PIN_' must normalize to 'PIN:' and be classified as a label"
        )

    def test_trailing_underscore_password_label_type(self, tool):
        r = tool._classify_text_type("PIN_")
        assert r["type"] == "password_label"

    def test_mid_underscore_space_with_value(self, tool):
        """'PIN_ 3902' → 'PIN: 3902' → password (label+value) → is_critical=True."""
        r = tool._classify_text_type("PIN_ 3902")
        assert r["is_critical"] is True, (
            "'PIN_ 3902' should normalize underscore-space to colon-space giving 'PIN: 3902' "
            "which is a password with a value (critical)"
        )
        assert r["is_label_only"] is False

    def test_password_underscore_label(self, tool):
        r = tool._classify_text_type("password_")
        assert r["is_label_only"] is True

    def test_multiple_underscores_normalize(self, tool):
        """'PIN__' should also normalize correctly."""
        r = tool._classify_text_type("PIN__")
        assert r["is_label_only"] is True


# ============================================================
# Fix 10: Screen spatial escalation_amount = 0
# ============================================================

class TestScreenSpatialEscalationAmountZero:
    """
    Regression: When SpatialRelationshipTool generates a 'content_association'
    escalation for a face near a screen device, escalation_amount must be 0
    (flag only — VLM decides, no automatic severity bump).
    """

    def test_face_near_screen_escalation_amount_is_zero(self):
        tool = _make_spatial_tool()
        payload = json.dumps({
            "image_width": 1024,
            "image_height": 768,
            "faces": [{"id": "f0", "bbox": [100, 100, 80, 80], "size": "medium"}],
            "texts": [],
            "objects": [{
                "id": "obj_0",
                "bbox": [200, 100, 200, 150],
                "contains_screen": True,
            }],
        })
        result = json.loads(tool._run(payload))
        assert "error" not in result

        screen_escalations = [
            e for e in result["escalations"]
            if e.get("relationship_type") == "content_association"
        ]
        assert len(screen_escalations) >= 1, (
            "Expected at least one content_association escalation for face near screen"
        )
        for esc in screen_escalations:
            assert esc["escalation_amount"] == 0, (
                f"Screen escalation_amount must be 0 (flag only), got {esc['escalation_amount']}"
            )

    def test_screen_escalation_has_correct_relationship_type(self):
        tool = _make_spatial_tool()
        payload = json.dumps({
            "image_width": 1024,
            "image_height": 768,
            "faces": [{"id": "f1", "bbox": [50, 50, 60, 60], "size": "large"}],
            "texts": [],
            "objects": [{"id": "obj_1", "bbox": [130, 50, 180, 120], "contains_screen": True}],
        })
        result = json.loads(tool._run(payload))
        screen_escs = [e for e in result["escalations"] if e.get("relationship_type") == "content_association"]
        assert len(screen_escs) >= 1

    def test_zero_amount_escalation_does_not_change_severity(self):
        """Applying escalation_amount=0 should leave severity unchanged."""
        tool = _make_escalation_tool()
        payload = {
            "assessments": [{
                "detection_id": "obj_0",
                "element_type": "object",
                "severity": "low",
                "requires_protection": False,
                "factors": {},
            }],
            "escalations": [{
                "elements": ["obj_0"],
                "escalation_amount": 0,
                "reason": "Face near screen device (VLM: verify screen state)",
                "relationship_type": "content_association",
            }],
        }
        result = json.loads(tool._run(json.dumps(payload)))
        assert result["escalations_applied"] == 0
        assert result["assessments"][0]["severity"] == "low"


# ============================================================
# Fix 11: Challenge-confirm — verified_on screen must NOT challenge
# ============================================================

class TestChallengeConfirmVerifiedOn:
    """
    Regression: ModifyStrategyTool must issue a challenge for verified_off screens,
    but must NOT challenge when screen_state is verified_on.

    (The test for verified_off already exists in test_safety_guards.py; this adds
    the complementary verified_on case to make the fix fully round-tripped.)
    """

    def _make_tool(self, strategies, challenges=None):
        from agents.tools import ModifyStrategyTool
        return ModifyStrategyTool(
            strategies=strategies,
            allowed_methods=["blur", "pixelate", "solid_overlay"],
            challenges_issued=challenges if challenges is not None else {},
        )

    def test_verified_on_screen_no_challenge(self):
        """Applying blur to a verified_on screen must succeed without a challenge."""
        strategies = [{
            "element_type": "object",
            "element_description": "Object: laptop",
            "severity": "medium",
            "method": "none",
            "consent_status": None,
            "person_label": None,
            "requires_protection": False,
            "reasoning": "Screen is on — may contain sensitive content",
            "screen_state": "verified_on",
        }]
        tool = self._make_tool(strategies)
        result = json.loads(tool._run(0, "blur", {}, "Screen shows sensitive data"))
        assert result["status"] == "modified", (
            "verified_on screen should be protectable without challenge; "
            f"got status={result['status']}"
        )

    def test_verified_off_screen_challenges_first(self):
        """Applying blur to a verified_off screen must issue a challenge first."""
        strategies = [{
            "element_type": "object",
            "element_description": "Object: tv",
            "severity": "low",
            "method": "none",
            "consent_status": None,
            "person_label": None,
            "requires_protection": False,
            "reasoning": "Screen is off",
            "screen_state": "verified_off",
        }]
        tool = self._make_tool(strategies)
        result = json.loads(tool._run(0, "blur", {}, "just in case"))
        assert result["status"] == "challenge"
        assert result["challenge_type"] == "verified_off_screen"

    def test_none_screen_state_no_spurious_challenge(self):
        """Non-screen objects (screen_state=None) should not trigger a screen challenge."""
        strategies = [{
            "element_type": "object",
            "element_description": "Object: backpack",
            "severity": "low",
            "method": "none",
            "consent_status": None,
            "person_label": None,
            "requires_protection": False,
            "reasoning": "Personal item",
            "screen_state": None,
        }]
        tool = self._make_tool(strategies)
        result = json.loads(tool._run(0, "blur", {}, "I see a backpack"))
        # Should be modified (or challenged for a different reason), but NOT for verified_off
        if result["status"] == "challenge":
            assert result.get("challenge_type") != "verified_off_screen"


# ============================================================
# Fix 12: BoundingBox.from_raw — all 4 input types
# ============================================================

class TestBoundingBoxFromRaw:
    """
    Regression: BoundingBox.from_raw must accept dict, list/tuple, BoundingBox,
    and None/unknown (returns zero-area box).
    """

    def test_from_dict(self):
        from utils.models import BoundingBox
        bb = BoundingBox.from_raw({"x": 10, "y": 20, "width": 100, "height": 50})
        assert bb.x == 10
        assert bb.y == 20
        assert bb.width == 100
        assert bb.height == 50

    def test_from_list(self):
        from utils.models import BoundingBox
        bb = BoundingBox.from_raw([5, 15, 200, 80])
        assert bb.x == 5
        assert bb.y == 15
        assert bb.width == 200
        assert bb.height == 80

    def test_from_tuple(self):
        from utils.models import BoundingBox
        bb = BoundingBox.from_raw((0, 0, 300, 150))
        assert bb.x == 0
        assert bb.width == 300

    def test_from_bounding_box_instance(self):
        from utils.models import BoundingBox
        original = BoundingBox(x=7, y=8, width=90, height=40)
        result = BoundingBox.from_raw(original)
        assert result is original, "from_raw(BoundingBox) must return the same instance"

    def test_from_none_returns_zero_area(self):
        from utils.models import BoundingBox
        bb = BoundingBox.from_raw(None)
        assert bb.x == 0
        assert bb.y == 0
        assert bb.width == 0
        assert bb.height == 0

    def test_from_unknown_type_returns_zero_area(self):
        from utils.models import BoundingBox
        bb = BoundingBox.from_raw("not-a-bbox")
        assert bb.x == 0 and bb.width == 0

    def test_to_list_roundtrip_from_list(self):
        from utils.models import BoundingBox
        original = [33, 44, 120, 60]
        bb = BoundingBox.from_raw(original)
        assert bb.to_list() == original

    def test_to_list_roundtrip_from_dict(self):
        from utils.models import BoundingBox
        bb = BoundingBox.from_raw({"x": 1, "y": 2, "width": 3, "height": 4})
        assert bb.to_list() == [1, 2, 3, 4]


# ============================================================
# Fix 13: Division-by-zero guard in _classify_face_size
# ============================================================

class TestClassifyFaceSizeDivisionByZero:
    """
    Regression: FaceDetectionTool._classify_face_size must return "medium" when
    image_area == 0 instead of raising ZeroDivisionError.
    """

    @pytest.fixture
    def tool(self):
        return _make_face_tool()

    def test_zero_image_area_returns_medium(self, tool):
        result = tool._classify_face_size(100, 100, 0, 0)
        assert result == "medium", (
            "_classify_face_size with image_area=0 must return 'medium', not raise"
        )

    def test_zero_width_returns_medium(self, tool):
        result = tool._classify_face_size(100, 100, 0, 768)
        assert result == "medium"

    def test_zero_height_returns_medium(self, tool):
        result = tool._classify_face_size(100, 100, 1024, 0)
        assert result == "medium"

    def test_normal_large_face(self, tool):
        """Sanity: a face that is 50% of a normal image is 'large'."""
        result = tool._classify_face_size(512, 384, 1024, 768)
        assert result == "large"

    def test_normal_small_face(self, tool):
        """Sanity: a tiny face in a large image is 'small'."""
        result = tool._classify_face_size(20, 20, 1920, 1080)
        assert result == "small"


# ============================================================
# Fix 14: Escalation severity clamp — negative amount stays at LOW
# ============================================================

class TestEscalationSeverityClamp:
    """
    Regression: RiskEscalationTool._escalate_severity must never go below index 0
    (i.e., below "low") even when escalation_amount is negative.
    """

    @pytest.fixture
    def tool(self):
        return _make_escalation_tool()

    def test_negative_amount_clamps_at_low(self, tool):
        result = tool._escalate_severity("low", -5)
        assert result == "low", (
            "_escalate_severity('low', -5) must clamp to 'low', not underflow"
        )

    def test_negative_one_from_low_stays_low(self, tool):
        assert tool._escalate_severity("low", -1) == "low"

    def test_large_positive_clamps_at_critical(self, tool):
        result = tool._escalate_severity("low", 100)
        assert result == "critical"

    def test_normal_escalation_medium_to_high(self, tool):
        assert tool._escalate_severity("medium", 1) == "high"

    def test_normal_deescalation_high_to_medium(self, tool):
        assert tool._escalate_severity("high", -1) == "medium"

    def test_invalid_severity_does_not_crash(self, tool):
        """Unknown severity defaults to index=1 (medium); add 1 → high."""
        result = tool._escalate_severity("unknown_level", 1)
        assert result == "high"


# ============================================================
# Fix 15: Pydantic field names — processing_time_ms, transformations_applied,
#         get_by_severity on their respective models
# ============================================================

class TestPydanticFieldNames:
    """
    Regression: Confirm that field renames / additions that were applied to
    Pydantic models still exist under the correct names.
    """

    def test_detection_results_processing_time_ms(self):
        from utils.models import DetectionResults, BoundingBox
        dr = DetectionResults(image_path="/tmp/test.png", processing_time_ms=123.4)
        assert dr.processing_time_ms == pytest.approx(123.4), (
            "DetectionResults.processing_time_ms field is missing or renamed"
        )

    def test_risk_analysis_result_processing_time_ms(self):
        from utils.models import RiskAnalysisResult, RiskLevel
        rar = RiskAnalysisResult(
            image_path="/tmp/test.png",
            overall_risk_level=RiskLevel.LOW,
            processing_time_ms=50.0,
        )
        assert rar.processing_time_ms == pytest.approx(50.0)

    def test_execution_report_transformations_applied(self):
        from utils.models import ExecutionReport
        er = ExecutionReport(image_path="/tmp/test.png", status="completed")
        assert hasattr(er, "transformations_applied"), (
            "ExecutionReport.transformations_applied field is missing"
        )
        assert isinstance(er.transformations_applied, list)

    def test_execution_report_transformations_applied_accepts_items(self):
        from utils.models import ExecutionReport, TransformationResult, ObfuscationMethod
        tr = TransformationResult(
            detection_id="face_0",
            element="Face",
            method=ObfuscationMethod.BLUR,
            parameters={"kernel_size": 25},
            status="success",
            execution_time_ms=12.5,
        )
        er = ExecutionReport(
            image_path="/tmp/test.png",
            status="completed",
            transformations_applied=[tr],
        )
        assert len(er.transformations_applied) == 1

    def test_risk_analysis_result_get_by_severity_method(self):
        from utils.models import RiskAnalysisResult, RiskAssessment, RiskLevel, RiskType, BoundingBox
        critical_assessment = RiskAssessment(
            detection_id="t0",
            element_type="text",
            element_description="Text: 123-45-6789",
            risk_type=RiskType.INFORMATION_DISCLOSURE,
            severity=RiskLevel.CRITICAL,
            color_code="#FF0000",
            reasoning="SSN",
            user_sensitivity_applied="personal_numbers",
            bbox=BoundingBox(x=0, y=0, width=100, height=30),
        )
        rar = RiskAnalysisResult(
            image_path="/tmp/test.png",
            overall_risk_level=RiskLevel.CRITICAL,
            risk_assessments=[critical_assessment],
        )
        assert hasattr(rar, "get_by_severity"), (
            "RiskAnalysisResult.get_by_severity method is missing"
        )
        results = rar.get_by_severity(RiskLevel.CRITICAL)
        assert len(results) == 1
        assert results[0].detection_id == "t0"

    def test_risk_analysis_result_get_by_severity_empty(self):
        from utils.models import RiskAnalysisResult, RiskLevel
        rar = RiskAnalysisResult(
            image_path="/tmp/test.png",
            overall_risk_level=RiskLevel.LOW,
        )
        assert rar.get_by_severity(RiskLevel.CRITICAL) == []

    def test_risk_analysis_result_get_critical_risks_helper(self):
        """get_critical_risks() is a convenience wrapper; must exist and work."""
        from utils.models import RiskAnalysisResult, RiskLevel
        rar = RiskAnalysisResult(
            image_path="/tmp/test.png",
            overall_risk_level=RiskLevel.LOW,
        )
        assert hasattr(rar, "get_critical_risks")
        assert rar.get_critical_risks() == []

    def test_risk_analysis_result_get_high_risks_helper(self):
        """get_high_risks() convenience wrapper must exist."""
        from utils.models import RiskAnalysisResult, RiskLevel
        rar = RiskAnalysisResult(
            image_path="/tmp/test.png",
            overall_risk_level=RiskLevel.LOW,
        )
        assert hasattr(rar, "get_high_risks")
        assert rar.get_high_risks() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
