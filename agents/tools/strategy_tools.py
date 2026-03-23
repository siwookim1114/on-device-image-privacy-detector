## Strategy Agent Tools (Phase 2)

import json
from typing import Any, List, Dict, Optional, Type, Union

from langchain.tools import BaseTool
from pydantic import BaseModel
from utils.models import (
    ModifyStrategyInput,
    ModifyStrategyItem,
    BatchModifyStrategiesInput,
)
from agents.tools.common import (
    get_risk_color,
    VALID_METHODS,
    METHOD_STRENGTH,
    SCREEN_KEYWORDS,
)


class ModifyStrategyTool(BaseTool):
    """Modify the obfuscation method/params for a single strategy."""
    name: str = "modify_strategy"
    description: str = (
        "Modify the obfuscation method for a specific element. "
        "Provide index, new method (blur/pixelate/solid_overlay/inpaint/avatar_replace/none), "
        "optional parameters dict, and reasoning."
    )
    args_schema: Type = ModifyStrategyInput
    handle_tool_error: bool = True
    strategies: Any = None
    allowed_methods: Any = None
    challenges_issued: Any = None  # shared dict() for challenge-confirm: {(index, type): first_reasoning}

    def _check_challenge(self, index: int, s: dict, method_lower: str, reasoning: str = "") -> Optional[str]:
        """
        Challenge-confirm pattern: return a warning on first attempt to add
        protection to items that likely don't need it. If the VLM calls again
        for the same item, allow through (VLM confirmed its decision).
        """
        if self.challenges_issued is None:
            self.challenges_issued = {}

        # Only challenge when ADDING protection (current=none, new!=none)
        if s["method"] != "none" or method_lower == "none":
            return None

        challenge_type = None
        challenge_msg = None

        # Challenge 1: consent=explicit face getting protection
        if (s["element_type"] == "face"
                and s.get("consent_status") == "explicit"):
            challenge_type = "consent_explicit_face"
            person = s.get("person_label", "unknown")
            challenge_msg = (
                f"CHALLENGE: This face (index={index}, label='{person}') has "
                f"consent_status='explicit', meaning the face-matching database "
                f"confirmed this person registered and granted explicit consent "
                f"to appear unprotected. Phase 1 correctly set method='none'. "
                f"Adding '{method_lower}' would override a verified consent decision. "
                f"If you believe this is a MISIDENTIFICATION (wrong person matched), "
                f"call modify_strategy again for index={index} with specific visual "
                f"evidence explaining why the face match is incorrect."
            )

        # Challenge 2: LOW text label getting protection
        elif (s["element_type"] == "text"
                and s["severity"] == "low"):
            challenge_type = "low_text_label"
            challenge_msg = (
                f"CHALLENGE: This text (index={index}, '{s['element_description']}') "
                f"has severity=LOW and method='none', indicating Phase 1 classified "
                f"it as a LABEL or descriptor (e.g., 'Password:', 'Bank Account:'). "
                f"Labels describe what data is nearby but do NOT themselves contain "
                f"sensitive information — only VALUES need protection. "
                f"If you visually confirm this text contains ACTUAL SENSITIVE DATA "
                f"(digits, passwords, account numbers — not just a label word), "
                f"call modify_strategy again for index={index} explaining what "
                f"specific sensitive data you see."
            )

        # Challenge 3: verified-OFF screen device getting protection
        elif (s["element_type"] == "object"
                and any(kw in s.get("element_description", "").lower()
                        for kw in SCREEN_KEYWORDS)
                and s.get("screen_state") == "verified_off"):
            challenge_type = "verified_off_screen"
            challenge_msg = (
                f"CHALLENGE: This screen device (index={index}, "
                f"'{s['element_description']}') has screen_state='verified_off'. "
                f"Phase 1.5a VLM verification already examined a CROPPED close-up "
                f"view of this specific device and determined: the screen is OFF, "
                f"facing AWAY from the camera, or showing the device's BACK/LID. "
                f"The screen content is NOT visible to the camera. "
                f"Adding '{method_lower}' would protect a non-visible screen. "
                f"If you believe the screen IS actually facing the camera AND shows "
                f"readable sensitive content, call modify_strategy again for "
                f"index={index} describing what specific content you see."
            )

        if challenge_type is None:
            return None

        key = (index, challenge_type)
        if key in self.challenges_issued:
            previous_reasoning = self.challenges_issued[key]
            # Second call — only allow if VLM provided DIFFERENT reasoning
            if reasoning.strip() == previous_reasoning.strip():
                # Same reasoning — re-issue challenge, do NOT consume
                return json.dumps({
                    "status": "challenge",
                    "index": index,
                    "challenge_type": challenge_type,
                    "message": (
                        f"{challenge_msg} "
                        f"NOTE: Your reasoning is identical to your previous attempt. "
                        f"Please provide DIFFERENT reasoning explaining what specific "
                        f"visual evidence you see that justifies this override."
                    ),
                })
            # Different reasoning — VLM confirmed with new justification, allow through
            del self.challenges_issued[key]
            return None
        else:
            # First call — issue challenge, store the reasoning used
            self.challenges_issued[key] = reasoning
            return json.dumps({
                "status": "challenge",
                "index": index,
                "challenge_type": challenge_type,
                "message": challenge_msg,
            })

    def _run(
        self,
        index: int,
        method: str,
        parameters: Dict[str, Any] = None,
        reasoning: str = "VLM strategy review",
    ) -> str:
        if parameters is None:
            parameters = {}

        if index < 0 or index >= len(self.strategies):
            return json.dumps({"status": "error", "message": f"Invalid index {index}, valid range: 0-{len(self.strategies)-1}"})

        method_lower = method.lower().strip()
        if method_lower not in VALID_METHODS:
            return json.dumps({"status": "error", "message": f"Invalid method '{method}'. Valid: {sorted(VALID_METHODS)}"})

        # Ethical mode guard
        if self.allowed_methods and method_lower not in self.allowed_methods and method_lower != "none":
            return json.dumps({"status": "blocked", "message": f"Method '{method}' not allowed in current ethical mode. Allowed: {self.allowed_methods}"})

        s = self.strategies[index]

        # Challenge-confirm: soft guard with VLM override on second call
        challenge_result = self._check_challenge(index, s, method_lower, reasoning)
        if challenge_result is not None:
            return challenge_result

        # Guard: cannot weaken CRITICAL items
        if s["severity"] == "critical":
            old_strength = METHOD_STRENGTH.get(s["method"], 0)
            new_strength = METHOD_STRENGTH.get(method_lower, 0)
            if new_strength < old_strength:
                return json.dumps({"status": "blocked", "message": f"Cannot weaken CRITICAL item from {s['method']} to {method_lower}"})

        # Guard: cannot weaken HIGH severity text items
        if s["severity"] == "high" and s["element_type"] == "text":
            old_strength = METHOD_STRENGTH.get(s["method"], 0)
            new_strength = METHOD_STRENGTH.get(method_lower, 0)
            if new_strength < old_strength:
                return json.dumps({"status": "blocked", "message": f"Cannot weaken HIGH severity text from {s['method']} to {method_lower}"})

        # Guard: cannot remove protection for bystander faces
        if s["element_type"] == "face" and s.get("consent_status") == "none" and method_lower == "none":
            return json.dumps({"status": "blocked", "message": "Cannot remove protection for bystander face (consent=none)"})

        # Guard: cannot set NONE for items that require protection
        if s["requires_protection"] and method_lower == "none":
            return json.dumps({"status": "blocked", "message": f"Cannot remove protection for item that requires_protection=True"})

        old_method = s["method"]
        s["method"] = method_lower
        s["parameters"] = parameters
        s["reasoning"] = f"{s['reasoning']} -> VLM: {reasoning}"
        s["vlm_modified"] = True

        return json.dumps({
            "status": "modified",
            "index": index,
            "element": s["element_description"],
            "old_method": old_method,
            "new_method": method_lower,
            "parameters": parameters,
        })


class BatchModifyStrategiesTool(BaseTool):
    """Modify multiple strategies at once."""
    name: str = "batch_modify_strategies"
    description: str = (
        "Modify obfuscation methods for multiple elements at once. "
        "Provide a list of modifications, each with index, method, optional parameters, and reasoning."
    )
    args_schema: Type = BatchModifyStrategiesInput
    handle_tool_error: bool = True
    strategies: Any = None
    allowed_methods: Any = None
    challenges_issued: Any = None  # shared dict() for challenge-confirm: {(index, type): first_reasoning}

    def _run(self, modifications: List[Dict]) -> str:
        single_tool = ModifyStrategyTool(
            strategies=self.strategies,
            allowed_methods=self.allowed_methods,
            challenges_issued=self.challenges_issued,
        )
        results = []
        for mod in modifications:
            if isinstance(mod, dict):
                idx = mod.get("index", 0)
                method = mod.get("method", "blur")
                params = mod.get("parameters", {})
                reason = mod.get("reasoning", "VLM batch review")
            else:
                idx = mod.index
                method = mod.method
                params = mod.parameters
                reason = mod.reasoning
            result = json.loads(single_tool._run(idx, method, params, reason))
            results.append(result)
        return json.dumps({"status": "batch_complete", "results": results})


class GetCurrentStrategiesTool(BaseTool):
    """View current strategy state."""
    name: str = "get_current_strategies"
    description: str = "View the current state of all proposed strategies. No arguments needed."
    handle_tool_error: bool = True
    strategies: Any = None

    def _run(self, tool_input: str = "") -> str:
        lines = []
        for i, s in enumerate(self.strategies):
            screen_info = ""
            if s.get("screen_state"):
                screen_info = f" | screen={s['screen_state']}"
            lines.append(
                f"[{i}] {s['element_type']} | {s['element_description'][:40]} | "
                f"severity={s['severity']} | method={s['method']}{screen_info} | "
                f"params={json.dumps(s.get('parameters', {}))}"
            )
        return "\n".join(lines)


class FinalizeStrategiesTool(BaseTool):
    """Finalize all strategies."""
    name: str = "finalize_strategies"
    description: str = "Finalize all strategies after review. Call this when done reviewing. No arguments needed."
    handle_tool_error: bool = True
    strategies: Any = None
    already_finalized: bool = False

    def _run(self, tool_input: str = "") -> str:
        if self.already_finalized:
            return json.dumps({"status": "already_finalized", "message": "Strategies already finalized"})

        self.already_finalized = True
        method_counts = {}
        for s in self.strategies:
            m = s["method"]
            method_counts[m] = method_counts.get(m, 0) + 1

        return json.dumps({
            "status": "finalized",
            "total_strategies": len(self.strategies),
            "method_breakdown": method_counts,
            "message": "All strategies finalized. Review complete.",
        })
