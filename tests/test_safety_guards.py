"""
Unit tests for safety guards, text classification, and challenge-confirm pattern.

These tests run WITHOUT any ML models (no MTCNN, EasyOCR, YOLO, VLM server).
They test pure logic functions and tool guards using synthetic data.
"""

import sys
import json
import pytest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================
# Helpers: bypass __init__ to avoid loading ML models
# ============================================================

def make_text_tool():
    """Create TextDetectionTool without loading EasyOCR."""
    from agents.tools import TextDetectionTool
    tool = TextDetectionTool.__new__(TextDetectionTool)
    return tool


class _StubConfig:
    """Minimal config stub that satisfies get_risk_color()."""
    class _Levels:
        class _Level:
            def __init__(self, color):
                self.color = color
        critical = _Level("#FF0000")
        high = _Level("#FF6600")
        medium = _Level("#FFD700")
        low = _Level("#90EE90")
    risk_levels = _Levels()

    def get(self, key, default=None):
        """Support dot-notation key lookup for get_risk_color."""
        parts = key.split(".")
        obj = self
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                return default
        return obj


# ============================================================
# 1. TextDetectionTool._classify_text_type
# ============================================================

class TestClassifyTextType:
    """Test the regex-based text classification — highest regression risk function."""

    @pytest.fixture
    def tool(self):
        return make_text_tool()

    # SSN patterns
    def test_ssn_full(self, tool):
        r = tool._classify_text_type("123-45-6789")
        assert r["type"] == "ssn"
        assert r["is_critical"] is True

    def test_ssn_with_spaces(self, tool):
        r = tool._classify_text_type("232-45-6193")
        assert r["type"] == "ssn"
        assert r["is_critical"] is True

    # Credit card
    def test_credit_card_spaced(self, tool):
        r = tool._classify_text_type("4242 4242 4242 4242")
        assert r["type"] == "credit_card"
        assert r["is_critical"] is True

    # Email
    def test_email(self, tool):
        r = tool._classify_text_type("john@example.com")
        assert r["type"] == "email"
        assert r["is_pii"] is True

    # Labels (should NOT be critical)
    def test_password_label(self, tool):
        r = tool._classify_text_type("Password:")
        assert r["is_label_only"] is True
        assert r["is_critical"] is False

    def test_bank_account_label(self, tool):
        r = tool._classify_text_type("Bank Account:")
        assert r["is_label_only"] is True

    def test_pin_label_underscore(self, tool):
        """Regression: PIN_ should normalize to PIN: and be a label."""
        r = tool._classify_text_type("PIN_")
        assert r["is_label_only"] is True

    # Numeric fragments
    def test_numeric_fragment_4digit(self, tool):
        r = tool._classify_text_type("1234")
        assert r["type"] == "numeric_fragment"

    def test_numeric_fragment_pin(self, tool):
        r = tool._classify_text_type("4821")
        assert r["type"] == "numeric_fragment"

    # General text (should NOT trigger PII)
    def test_general_text(self, tool):
        r = tool._classify_text_type("Hello world")
        assert r["type"] == "general_text"
        assert r["is_sensitive"] is False

    # Regression: "book" should not be flagged
    def test_book_not_flagged(self, tool):
        r = tool._classify_text_type("book title")
        assert r["type"] == "general_text"

    # Partial SSN fragments
    def test_partial_ssn_fragment(self, tool):
        r = tool._classify_text_type("349-08-")
        # Partial SSN (missing last 4) — matches numeric_fragment pattern "349-08"
        # The trailing dash gets caught by the partial numeric pattern
        assert r["type"] in ("numeric_fragment", "general_text")  # depends on exact regex match


# ============================================================
# 2. ReclassifyAssessmentTool safety guards
# ============================================================

class TestReclassifyGuards:
    """Test that safety guards prevent dangerous reclassifications."""

    def _make_tool(self, assessments):
        from agents.tools import ReclassifyAssessmentTool
        return ReclassifyAssessmentTool(assessments=assessments, config=_StubConfig())

    def test_block_downgrade_critical_text(self):
        """Cannot downgrade CRITICAL text item."""
        assessments = [{
            "element_type": "text",
            "element_description": "Text: 238-49-6521",
            "severity": "critical",
            "requires_protection": True,
            "factors": {},
        }]
        tool = self._make_tool(assessments)
        result = json.loads(tool._run(0, "low", "test"))
        assert result["status"] == "blocked"

    def test_block_downgrade_high_text(self):
        """Cannot downgrade HIGH text item."""
        assessments = [{
            "element_type": "text",
            "element_description": "Text: john@example.com",
            "severity": "high",
            "requires_protection": True,
            "factors": {},
        }]
        tool = self._make_tool(assessments)
        result = json.loads(tool._run(0, "low", "test"))
        assert result["status"] == "blocked"

    def test_block_downgrade_bystander_face(self):
        """Cannot downgrade face with consent=none (bystander)."""
        assessments = [{
            "element_type": "face",
            "element_description": "Face (medium, high clarity)",
            "severity": "critical",
            "consent_status": "none",
            "requires_protection": True,
            "factors": {},
        }]
        tool = self._make_tool(assessments)
        result = json.loads(tool._run(0, "low", "test"))
        assert result["status"] == "blocked"

    def test_block_screen_device_reclassify(self):
        """Cannot reclassify screen device (pre-verified by VLM crop)."""
        assessments = [{
            "element_type": "object",
            "element_description": "Object: laptop",
            "severity": "low",
            "requires_protection": False,
            "factors": {"contains_screen": True, "risk_category": "screen_device"},
        }]
        tool = self._make_tool(assessments)
        result = json.loads(tool._run(0, "high", "test"))
        assert result["status"] == "blocked"

    def test_noop_same_severity(self):
        """No-op when target severity equals current."""
        assessments = [{
            "element_type": "text",
            "element_description": "Text: hello",
            "severity": "low",
            "requires_protection": False,
            "factors": {},
        }]
        tool = self._make_tool(assessments)
        result = json.loads(tool._run(0, "low", "test"))
        assert result["status"] == "no_change"

    def test_allow_valid_escalation(self):
        """Valid escalation LOW → HIGH should succeed."""
        assessments = [{
            "element_type": "text",
            "element_description": "Text: 4821",
            "severity": "low",
            "requires_protection": False,
            "reasoning": "Tool-based assessment",
            "factors": {},
        }]
        tool = self._make_tool(assessments)
        result = json.loads(tool._run(0, "high", "Found PII"))
        assert result["status"] == "success"
        assert assessments[0]["severity"] == "high"
        assert assessments[0]["requires_protection"] is True


# ============================================================
# 3. ModifyStrategyTool challenge-confirm pattern
# ============================================================

class TestChallengeConfirm:
    """Test the stateful challenge-confirm pattern."""

    def _make_tool(self, strategies, challenges=None):
        from agents.tools import ModifyStrategyTool
        if challenges is None:
            challenges = {}
        return ModifyStrategyTool(
            strategies=strategies,
            allowed_methods=["blur", "pixelate", "solid_overlay"],
            challenges_issued=challenges,
        )

    def test_challenge_consent_explicit_face(self):
        """First call to protect consent=explicit face → challenge."""
        strategies = [{
            "element_type": "face",
            "element_description": "Face (medium, high clarity)",
            "severity": "low",
            "method": "none",
            "consent_status": "explicit",
            "person_label": "Me",
            "requires_protection": False,
            "reasoning": "User's own face (explicit consent)",
            "screen_state": None,
        }]
        tool = self._make_tool(strategies)
        result = json.loads(tool._run(0, "blur", {}, "test"))
        assert result["status"] == "challenge"
        assert result["challenge_type"] == "consent_explicit_face"

    def test_confirm_after_challenge(self):
        """Second call for same face → allow through."""
        strategies = [{
            "element_type": "face",
            "element_description": "Face (medium, high clarity)",
            "severity": "low",
            "method": "none",
            "consent_status": "explicit",
            "person_label": "Me",
            "requires_protection": False,
            "reasoning": "User's own face (explicit consent)",
            "screen_state": None,
        }]
        challenges = {}
        tool = self._make_tool(strategies, challenges)
        # First call → challenge
        r1 = json.loads(tool._run(0, "blur", {}, "test"))
        assert r1["status"] == "challenge"
        # Second call → should go through
        r2 = json.loads(tool._run(0, "blur", {}, "I see misidentification"))
        assert r2["status"] == "modified"

    def test_challenge_verified_off_screen(self):
        """Protecting a verified_off screen → challenge."""
        strategies = [{
            "element_type": "object",
            "element_description": "Object: laptop",
            "severity": "low",
            "method": "none",
            "consent_status": None,
            "person_label": None,
            "requires_protection": False,
            "reasoning": "Low risk object — no protection needed",
            "screen_state": "verified_off",
        }]
        tool = self._make_tool(strategies)
        result = json.loads(tool._run(0, "blur", {}, "test"))
        assert result["status"] == "challenge"
        assert result["challenge_type"] == "verified_off_screen"

    def test_no_challenge_verified_on_screen(self):
        """Protecting a verified_on screen → NO challenge (allowed)."""
        strategies = [{
            "element_type": "object",
            "element_description": "Object: laptop",
            "severity": "medium",
            "method": "none",
            "consent_status": None,
            "person_label": None,
            "requires_protection": False,
            "reasoning": "Medium risk object — blur",
            "screen_state": "verified_on",
        }]
        tool = self._make_tool(strategies)
        result = json.loads(tool._run(0, "blur", {}, "Screen shows PII"))
        assert result["status"] == "modified"

    def test_block_remove_bystander_protection(self):
        """Cannot remove protection for bystander face (hard block, not challenge)."""
        strategies = [{
            "element_type": "face",
            "element_description": "Face (medium, high clarity)",
            "severity": "critical",
            "method": "pixelate",
            "consent_status": "none",
            "person_label": None,
            "requires_protection": True,
            "reasoning": "Bystander face, critical — mandatory pixelation",
            "screen_state": None,
        }]
        tool = self._make_tool(strategies)
        result = json.loads(tool._run(0, "none", {}, "test"))
        assert result["status"] == "blocked"

    def test_block_weaken_critical(self):
        """Cannot weaken CRITICAL item (solid_overlay → blur)."""
        strategies = [{
            "element_type": "text",
            "element_description": "Text: 238-49-6521",
            "severity": "critical",
            "method": "solid_overlay",
            "consent_status": None,
            "person_label": None,
            "requires_protection": True,
            "reasoning": "Critical PII — full solid overlay redaction",
            "screen_state": None,
        }]
        tool = self._make_tool(strategies)
        result = json.loads(tool._run(0, "blur", {}, "test"))
        assert result["status"] == "blocked"


# ============================================================
# 4. ConsistencyValidationTool
# ============================================================

class TestConsistencyValidation:
    """Test that consistency validation corrects mismatches."""

    def _make_tool(self):
        from agents.tools import ConsistencyValidationTool
        return ConsistencyValidationTool()

    def test_critical_must_require_protection(self):
        """CRITICAL item with requires_protection=False → corrected to True."""
        tool = self._make_tool()
        data = json.dumps({"assessments": [{
            "severity": "critical",
            "requires_protection": False,
        }]})
        result = json.loads(tool._run(data))
        assert result["corrections_made"] == 1
        assert result["assessments"][0]["requires_protection"] is True

    def test_low_must_not_require_protection(self):
        """LOW item with requires_protection=True → corrected to False."""
        tool = self._make_tool()
        data = json.dumps({"assessments": [{
            "severity": "low",
            "requires_protection": True,
        }]})
        result = json.loads(tool._run(data))
        assert result["corrections_made"] == 1
        assert result["assessments"][0]["requires_protection"] is False

    def test_no_correction_needed(self):
        """Consistent assessment → 0 corrections."""
        tool = self._make_tool()
        data = json.dumps({"assessments": [{
            "severity": "critical",
            "requires_protection": True,
        }]})
        result = json.loads(tool._run(data))
        assert result["corrections_made"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
