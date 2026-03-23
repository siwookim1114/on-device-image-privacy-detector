"""
Agent 3: Strategy Agent (VLM-Powered Obfuscation Recommender)

Two-phase architecture:
  Phase 1: Deterministic rule-based defaults (fast, no LLM needed)
  Phase 2: VLM agent with tool calling to review and modify defaults (Qwen3-VL)

Maps each RiskAssessment to a ProtectionStrategy with:
  - Recommended obfuscation method + parameters
  - Alternative methods (scored)
  - Processing mode flags (requires_user_decision, user_can_override)
  - Ethical compliance checks

Stateless run() returns clean StrategyRecommendations model.
"""

import json
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langgraph.errors import GraphRecursionError
from langchain_core.messages import HumanMessage

from utils.models import (
    RiskAnalysisResult,
    RiskAssessment,
    RiskLevel,
    ProtectionStrategy,
    StrategyRecommendations,
    AlternativeMethod,
    ObfuscationMethod,
    PrivacyProfile,
)
from agents.local_wrapper import VisionLLM
from agents.agent_factory import create_vlm, resize_for_vlm, build_vlm_agent
from agents.tools import (
    ModifyStrategyTool,
    BatchModifyStrategiesTool,
    GetCurrentStrategiesTool,
    FinalizeStrategiesTool,
)

class StrategyAgent:
    """
    Agent 3: VLM-Powered Strategy Agent

    Phase 1: Deterministic rule-based defaults (fast, <100ms)
    Phase 2: VLM agent with tool calling to review and modify (Qwen3-VL)
    """

    def __init__(
        self,
        config,
        privacy_profile: Optional[PrivacyProfile] = None,
        vlm_backend: str = "llama-cpp",
    ):
        self.config = config
        self.privacy_profile = privacy_profile or PrivacyProfile()
        self.vlm_backend = vlm_backend

        # Parse ethical and processing modes from profile
        self.ethical_mode = self.privacy_profile.ethical_mode
        self.processing_mode = self.privacy_profile.default_mode

        # Cache allowed methods from config
        self.allowed_methods = self._get_allowed_methods()

        # VLM backend config (shared factory)
        self.vlm, self.vlm_model = create_vlm(vlm_backend)

        print(f"\n[StrategyAgent] Initialized")
        print(f"  Phase 1: Deterministic rule-based defaults")
        print(f"  Phase 2: VLM agent review (model: {self.vlm_model}, backend: {self.vlm_backend})")
        print(f"  Ethical mode: {self.ethical_mode}")
        print(f"  Processing mode: {self.processing_mode}")
        print(f"  Allowed methods: {self.allowed_methods}")

    # ==================== Public API ====================

    def run(
        self,
        risk_result: RiskAnalysisResult,
        image_path: str,
        annotated_image: Optional[Image.Image] = None,
    ) -> StrategyRecommendations:
        """
        Generate obfuscation strategies for all risk assessments.

        Args:
            risk_result: RiskAnalysisResult from Agent 2 + 2.5
            image_path: Path to original image
            annotated_image: Annotated image with bounding boxes (None = Phase 1 only)

        Returns:
            StrategyRecommendations with one ProtectionStrategy per assessment
        """
        start = time.time()

        # Phase 1: Deterministic defaults
        print(f"\n  Strategy Phase 1: Rule-based defaults...")
        strategies = self._rule_based_defaults(risk_result)
        print(f"  Phase 1: {len(strategies)} strategies generated")

        # Phase 2: VLM review (skip if no image provided = fallback-only mode)
        if annotated_image is not None:
            print(f"\n  Strategy Phase 2: VLM agent review...")
            strategies = self._vlm_review(strategies, annotated_image)
        else:
            print(f"  Phase 2 skipped (fallback-only mode)")

        # Apply processing mode flags
        for s in strategies:
            self._apply_processing_mode(s)

        # Build final result
        result = self._build_result(strategies, image_path, start)
        self._print_summary(result)
        return result

    # ==================== Phase 1: Deterministic Defaults ====================

    def _rule_based_defaults(self, risk_result: RiskAnalysisResult) -> List[Dict]:
        """Generate rule-based default strategy for each assessment."""
        strategies = []
        for assessment in risk_result.risk_assessments:
            strategy = self._generate_default_strategy(assessment)
            strategies.append(strategy)
        return strategies

    def _generate_default_strategy(self, a: RiskAssessment) -> Dict:
        """Map a single assessment to a default strategy dict."""
        severity = a.severity.value if hasattr(a.severity, "value") else a.severity

        if a.element_type == "face":
            method, params, reasoning = self._default_for_face(a, severity)
        elif a.element_type == "text":
            method, params, reasoning = self._default_for_text(a, severity)
        else:
            method, params, reasoning = self._default_for_object(a, severity)

        # Enforce ethical mode
        method = self._enforce_ethical_mode(method)

        # Generate alternatives
        alternatives = self._generate_alternatives(method, a.element_type, severity)

        return {
            "detection_id": a.detection_id,
            "element_type": a.element_type,
            "element_description": a.element_description,
            "severity": severity,
            "method": method,
            "parameters": params,
            "reasoning": reasoning,
            "alternatives": alternatives,
            "consent_status": a.consent_status.value if hasattr(a.consent_status, "value") else a.consent_status,
            "person_label": a.person_label,
            "requires_protection": a.requires_protection,
            "screen_state": a.screen_state,
            "vlm_modified": False,
        }

    def _default_for_face(self, a: RiskAssessment, severity: str) -> Tuple[str, Dict, str]:
        """Rule-based default for face elements."""
        consent = a.consent_status
        if hasattr(consent, "value"):
            consent = consent.value

        # User's own face — no protection
        if consent == "explicit":
            return "none", {}, "User's own face (explicit consent)"

        # Known contact
        if consent == "assumed":
            if severity == "low":
                return "none", {}, "Known contact, low risk context"
            elif severity == "medium":
                return "blur", {"kernel_size": 15}, "Known contact, moderate risk — light blur"
            elif severity == "high":
                return "blur", {"kernel_size": 25}, "Known contact, high risk context"
            else:  # critical
                return "blur", {"kernel_size": 35}, "Known contact, critical risk context"

        # Unclear consent
        if consent == "unclear":
            return "blur", {"kernel_size": 25}, "Unclear consent — default blur protection"

        # Bystander (consent=none or null)
        if severity in ("low", "medium"):
            return "blur", {"kernel_size": 25}, "Bystander face — blur protection"
        elif severity == "high":
            return "blur", {"kernel_size": 35}, "Bystander face, high risk — strong blur"
        else:  # critical
            return "pixelate", {"block_size": 16}, "Bystander face, critical — mandatory pixelation"

    def _default_for_text(self, a: RiskAssessment, severity: str) -> Tuple[str, Dict, str]:
        """Rule-based default for text elements."""
        if severity == "critical":
            return "solid_overlay", {"color": "#000000"}, "Critical PII — full solid overlay redaction"
        elif severity == "high":
            return "solid_overlay", {"color": "#000000"}, "Sensitive PII — solid overlay redaction"
        elif severity == "medium":
            return "blur", {"kernel_size": 25}, "Moderate sensitivity text — blur"
        else:  # low
            return "none", {}, "Low risk text (label or benign)"

    def _default_for_object(self, a: RiskAssessment, severity: str) -> Tuple[str, Dict, str]:
        """Rule-based default for object elements."""
        if severity == "low":
            return "none", {}, "Low risk object — no protection needed"
        elif severity == "medium":
            return "blur", {"kernel_size": 25}, "Medium risk object — blur"
        else:  # high or critical
            return "blur", {"kernel_size": 35}, "High/critical risk object — strong blur"

    # ==================== Phase 2: VLM Agent Review ====================

    def _vlm_review(
        self,
        strategies: List[Dict],
        annotated_image: Image.Image,
    ) -> List[Dict]:
        """
        Phase 2: VLM agent reviews Phase 1 defaults with visual context.

        Uses create_agent with tools so the VLM can modify strategies
        through structured, guarded tool calls. Middleware prevents
        context overflow and caps iterations.
        """
        print(f"\n{'-'*60}")
        print(f"Strategy Phase 2: VLM Agent Review")
        print(f"{'-'*60}")

        try:
            # Resize image for VLM
            annotated_image = resize_for_vlm(annotated_image, max_dim=1024)

            image_b64 = self.vlm._image_to_base64(annotated_image)

            # Build agent
            agent, max_iters = self._build_vlm_agent(strategies)

            # Build strategy summary for VLM
            strategy_summary = self._format_strategy_summary(strategies)

            input_message = HumanMessage(content=[
                {"type": "text", "text": f"Proposed obfuscation strategies to review:\n{strategy_summary}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ])

            print(f"  Starting VLM strategy review ({len(strategies)} strategies)...")

            result = agent.invoke(
                {"messages": [input_message]},
                config={"recursion_limit": 2 * max_iters + 5},
            )

            # Extract tool calls from message history
            tool_calls = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(tc["name"])

            print(f"  VLM made {len(tool_calls)} tool calls: {tool_calls}")
            print(f"  Phase 2 complete: {len(strategies)} strategies after review")

            return strategies  # Modified in-place by tools

        except GraphRecursionError:
            print(f"  VLM agent hit max iterations — returning strategies as modified so far")
            return strategies

        except Exception as e:
            print(f"  VLM strategy review failed: {e}")
            traceback.print_exc()
            print(f"  Keeping Phase 1 defaults unchanged")
            return strategies

    def _build_vlm_agent(self, strategies: List[Dict]):
        """
        Build Phase 2 agent with tools and middleware.

        Tools modify strategies in-place with safety guards.
        MessageTrimMiddleware keeps context within VLM limits.
        ModelCallLimitMiddleware caps iterations.
        """
        shared_challenges = {}

        phase2_tools = [
            ModifyStrategyTool(
                strategies=strategies,
                allowed_methods=self.allowed_methods,
                challenges_issued=shared_challenges,
            ),
            BatchModifyStrategiesTool(
                strategies=strategies,
                allowed_methods=self.allowed_methods,
                challenges_issued=shared_challenges,
            ),
            GetCurrentStrategiesTool(strategies=strategies),
            FinalizeStrategiesTool(strategies=strategies),
        ]

        n = len(strategies)
        max_iters = max(10, min(n + 8, 30))

        system_prompt = (
            "You are a privacy protection strategy advisor. You act as a SAFETY NET — the previous agents "
            "(detection + risk assessment) may have made mistakes. Your job is to visually verify each "
            "proposed strategy against the actual image and correct any errors.\n\n"
            f"Ethical mode: {self.ethical_mode} (allowed methods: {', '.join(self.allowed_methods)})\n\n"
            "REVIEW GUIDELINES BY ELEMENT TYPE:\n\n"
            "FACES:\n"
            "- ALWAYS trust the consent status from the data (consent=none, consent=explicit, etc.) — "
            "it was determined by a face-matching agent using a registered consent database. "
            "Do NOT override consent based on visual appearance.\n"
            "- consent=none (bystander): blur or pixelate — NEVER set to 'none'\n"
            "- consent=explicit (user): should always be 'none'\n"
            "- Your role for faces: verify the obfuscation METHOD is appropriate (blur vs pixelate vs solid_overlay) "
            "based on face size and visibility — not re-judge identity or consent.\n\n"
            "TEXT — LABELS vs VALUES (critical distinction):\n"
            "- LABELS are descriptors like 'Social Security #:', 'Bank Account:', 'PIN:', 'Password:'\n"
            "  Labels do NOT contain sensitive data themselves — they should typically be 'none'\n"
            "- VALUES are the actual sensitive data like '238-49-6521', 'magicl231', '4821'\n"
            "  Values MUST be protected — critical PII values should use solid_overlay\n"
            "- If you see a text item with severity=low and method=none, it is likely a label — "
            "do NOT add protection unless you visually confirm it contains actual sensitive data\n\n"
            "OBJECTS (screen devices):\n"
            "- Each screen device has a 'screen_state' field from Phase 1.5a crop verification.\n"
            "- screen=verified_off → screen is OFF, CLOSED, or showing BACK/LID (facing away). "
            "Do NOT add protection.\n"
            "- screen=verified_on → screen faces camera and is ON. BUT 'on' does NOT mean 'sensitive'. "
            "A screen can be ON and still show nothing private. YOU MUST visually evaluate the CONTENT:\n"
            "  * PROTECT (blur/solid_overlay) ONLY IF you can see readable sensitive text: "
            "personal names, addresses, emails, passwords, financial data, medical records, login forms\n"
            "  * Set method='none' if the screen shows: a solid color, desktop wallpaper, generic UI, "
            "a lock screen, a blue/black screen, app windows with no readable PII, or content too "
            "small/blurry to read at this resolution\n"
            "  * The DEFAULT for verified_on screens with method=blur is ALREADY set — only KEEP it "
            "if you see specific sensitive content. Otherwise DOWNGRADE to 'none'.\n"
            "- Low-risk objects (mouse, keyboard): typically need no protection\n\n"
            "HARD RULES (cannot be overridden):\n"
            "- NEVER weaken CRITICAL severity items (e.g., solid_overlay -> blur)\n"
            "- NEVER weaken HIGH severity text items\n"
            "- NEVER remove protection for bystander faces (consent=none)\n"
            "- Items marked [LOCKED] require protection — cannot set to 'none'\n\n"
            "SOFT RULES (use judgment):\n"
            "- LOW severity items (labels, low-risk objects) usually stay at 'none' — only add protection "
            "if you visually confirm the item was misclassified and actually contains sensitive content\n"
            "- MEDIUM screen devices default to blur — downgrade to 'none' if you see NO readable "
            "sensitive content on the screen. Only keep blur or strengthen to solid_overlay if you can "
            "identify specific sensitive text/data on the screen.\n"
            "- Only use methods allowed by the current ethical mode\n\n"
            "WORKFLOW:\n"
            "1. Review all strategies against the image\n"
            "2. ONLY modify strategies that need CHANGES — do NOT call modify_strategy to confirm "
            "a default that is already correct. If a CRITICAL item is already solid_overlay, skip it. "
            "If a LOW label is already 'none', skip it.\n"
            "3. Use batch_modify_strategies when changing multiple items (more efficient than calling "
            "modify_strategy repeatedly)\n"
            "4. Call finalize_strategies when done — if all defaults look correct, call it immediately\n\n"
            "IMPORTANT:\n"
            "- ALWAYS call a tool in EVERY response. No explanations without tool calls.\n"
            "- Go directly to tool calls. No preamble or reasoning text.\n"
            "- Do NOT use <think> tags or internal reasoning. Act immediately."
        )

        # Middleware: trim old messages to prevent context overflow.
        # Keeps the first message (image + strategies) and last 10 messages
        # (5 tool call/result pairs).
        agent = build_vlm_agent(
            vlm=self.vlm,
            tools=phase2_tools,
            system_prompt=system_prompt,
            max_iters=max_iters,
            trim_threshold=12,
            trim_keep=10,
        )

        print(f"  Phase 2 agent built:")
        print(f"    LLM: {self.vlm_model}")
        print(f"    Max iterations: {max_iters}")
        print(f"    Tools: {[t.name for t in phase2_tools]}")

        return agent, max_iters

    # ==================== Shared Helpers ====================

    def _format_strategy_summary(self, strategies: List[Dict]) -> str:
        """Format strategies as text for VLM prompt."""
        lines = []
        for i, s in enumerate(strategies):
            consent_info = ""
            if s["element_type"] == "face":
                consent = s.get("consent_status", "unknown")
                label = s.get("person_label", "")
                consent_info = f" | consent={consent}"
                if label:
                    consent_info += f" ({label})"

            params_str = ""
            if s["parameters"]:
                params_str = f" | params={json.dumps(s['parameters'])}"

            # Show screen state for screen devices
            screen_info = ""
            if s.get("screen_state"):
                screen_info = f" | screen={s['screen_state']}"

            # Show LOCKED marker for items requiring protection
            lock_tag = " [LOCKED]" if s.get("requires_protection") else ""

            lines.append(
                f"[{i}] {s['element_type']} | {s['element_description'][:45]} | "
                f"severity={s['severity']}{lock_tag} | method={s['method']}{params_str}{consent_info}{screen_info} | "
                f"reason: {s['reasoning'][:60]}"
            )
        return "\n".join(lines)

    def _get_allowed_methods(self) -> List[str]:
        """Get allowed obfuscation methods from ethical mode config."""
        try:
            ethical_config = self.config.ethical_modes
            mode_config = getattr(ethical_config, self.ethical_mode, None)
            if mode_config is None:
                return ["blur", "pixelate", "solid_overlay", "avatar_replace", "inpaint"]

            allowed = mode_config.allowed if hasattr(mode_config, "allowed") else mode_config.get("allowed", "all")
            if allowed == "all":
                return ["blur", "pixelate", "solid_overlay", "avatar_replace", "inpaint", "generative_replace"]
            if isinstance(allowed, list):
                return allowed
            return ["blur", "pixelate", "solid_overlay"]
        except Exception:
            return ["blur", "pixelate", "solid_overlay", "avatar_replace", "inpaint"]

    def _enforce_ethical_mode(self, method: str) -> str:
        """Enforce ethical mode constraints on method selection."""
        if method == "none":
            return method
        if method in self.allowed_methods:
            return method
        for fallback in ["blur", "pixelate", "solid_overlay"]:
            if fallback in self.allowed_methods:
                return fallback
        return "none"

    def _generate_alternatives(self, primary: str, element_type: str, severity: str) -> List[Dict]:
        """Generate element-type aware alternative methods with default parameters."""
        if primary == "none":
            return []

        # Default parameters by method
        default_params = {
            "blur": {"kernel_size": 25 if severity in ("low", "medium") else 35},
            "pixelate": {"block_size": 12 if severity in ("low", "medium") else 16},
            "solid_overlay": {"color": "#000000"},
            "inpaint": {},
            "avatar_replace": {},
            "generative_replace": {},
        }

        # Element-type specific alternative maps
        if element_type == "face":
            alt_map = {
                "blur": [
                    ("pixelate", 8, "Pixelation preserves less detail"),
                    ("solid_overlay", 7, "Full concealment"),
                    ("avatar_replace", 6, "Replace with synthetic avatar"),
                ],
                "pixelate": [
                    ("blur", 8, "Gaussian blur is softer"),
                    ("solid_overlay", 7, "Full concealment"),
                    ("avatar_replace", 6, "Replace with synthetic avatar"),
                ],
                "solid_overlay": [
                    ("blur", 7, "Less aggressive but may leak identity"),
                    ("pixelate", 6, "Block-based concealment"),
                    ("avatar_replace", 5, "Replace with synthetic avatar"),
                ],
                "avatar_replace": [
                    ("blur", 8, "Simple blur alternative"),
                    ("pixelate", 7, "Pixelation alternative"),
                ],
            }
        elif element_type == "text":
            alt_map = {
                "blur": [
                    ("pixelate", 8, "Pixelation for text concealment"),
                    ("solid_overlay", 7, "Full text redaction"),
                ],
                "pixelate": [
                    ("solid_overlay", 8, "Full text redaction"),
                    ("blur", 7, "Gaussian blur alternative"),
                ],
                "solid_overlay": [
                    ("pixelate", 6, "Block-based concealment — may leak partial text"),
                    ("blur", 5, "Less aggressive — risk of OCR recovery"),
                ],
            }
        else:  # object
            alt_map = {
                "blur": [
                    ("pixelate", 8, "Pixelation alternative"),
                    ("solid_overlay", 7, "Full concealment"),
                    ("inpaint", 5, "AI-fill the region"),
                ],
                "pixelate": [
                    ("blur", 8, "Gaussian blur is softer"),
                    ("solid_overlay", 7, "Full concealment"),
                ],
                "solid_overlay": [
                    ("blur", 7, "Less aggressive blur"),
                    ("inpaint", 5, "AI-fill the region"),
                ],
                "inpaint": [
                    ("solid_overlay", 8, "Simpler full concealment"),
                    ("blur", 7, "Gaussian blur alternative"),
                ],
            }

        alternatives = []
        for alt_method, score, reasoning in alt_map.get(primary, []):
            if alt_method in self.allowed_methods:
                alternatives.append({
                    "method": alt_method,
                    "parameters": default_params.get(alt_method, {}),
                    "score": score,
                    "reasoning": reasoning,
                })
        return alternatives

    def _apply_processing_mode(self, strategy: Dict):
        """Apply processing mode flags to a strategy."""
        severity = strategy["severity"]
        method = strategy["method"]

        # NONE method never needs user decision
        if method == "none":
            strategy["requires_user_decision"] = False
            strategy["user_can_override"] = True
            strategy["optional"] = True
            return

        if self.processing_mode == "auto":
            strategy["requires_user_decision"] = False
        elif self.processing_mode == "hybrid":
            # CRITICAL auto-applies, everything else needs user review
            strategy["requires_user_decision"] = severity != "critical"
        else:  # manual
            strategy["requires_user_decision"] = True

        # user_can_override from config
        try:
            level_config = getattr(self.config.risk_levels, severity, None)
            strategy["user_can_override"] = level_config.user_can_override if level_config else True
        except Exception:
            strategy["user_can_override"] = severity != "critical"

        # Optional: LOW severity items
        strategy["optional"] = severity == "low"

    def _get_execution_priority(self, severity: str) -> int:
        """Map severity to execution priority (1=highest, 5=lowest)."""
        return {"critical": 1, "high": 2, "medium": 3, "low": 4}.get(severity, 5)

    def _build_result(
        self,
        strategies: List[Dict],
        image_path: str,
        start_time: float,
    ) -> StrategyRecommendations:
        """Convert strategy dicts to StrategyRecommendations."""
        protection_strategies = []

        for s in strategies:
            # Build alternative method objects
            alt_objects = []
            for alt in s.get("alternatives", []):
                alt_objects.append(AlternativeMethod(
                    method=ObfuscationMethod(alt["method"]),
                    parameters=alt.get("parameters", {}),
                    reasoning=alt["reasoning"],
                    score=alt["score"],
                ))

            # Map method string to ObfuscationMethod enum
            method_enum = ObfuscationMethod(s["method"])

            ps = ProtectionStrategy(
                detection_id=s["detection_id"],
                element=s["element_description"],
                severity=RiskLevel(s["severity"]),
                recommended_action="Protect" if s["method"] != "none" else "None",
                recommended_method=method_enum,
                parameters=s.get("parameters", {}),
                reasoning=s["reasoning"],
                alternative_options=alt_objects,
                ethical_compliance="COMPLIANT",
                execution_priority=self._get_execution_priority(s["severity"]),
                optional=s.get("optional", False),
                requires_user_decision=s.get("requires_user_decision", False),
                user_can_override=s.get("user_can_override", True),
            )
            protection_strategies.append(ps)

        # Sort by execution priority
        protection_strategies.sort(key=lambda p: p.execution_priority)

        total_protections = sum(
            1 for p in protection_strategies
            if p.recommended_method != ObfuscationMethod.NONE
        )
        requires_confirmation = sum(1 for p in protection_strategies if p.requires_user_decision)

        elapsed = (time.time() - start_time) * 1000

        return StrategyRecommendations(
            image_path=image_path,
            strategies=protection_strategies,
            total_protections_recommended=total_protections,
            requires_user_confirmation=requires_confirmation,
            estimated_processing_time_ms=elapsed,
        )

    def _print_summary(self, result: StrategyRecommendations):
        """Print strategy results summary."""
        print(f"\n  Strategy Recommendations Summary:")
        print(f"  ├─ Total strategies: {len(result.strategies)}")
        print(f"  ├─ Protections recommended: {result.total_protections_recommended}")
        print(f"  ├─ Requires user confirmation: {result.requires_user_confirmation}")
        print(f"  └─ Processing time: {result.estimated_processing_time_ms:.2f}ms")

        # Method breakdown
        method_counts = {}
        for s in result.strategies:
            m = s.recommended_method.value if s.recommended_method else "none"
            method_counts[m] = method_counts.get(m, 0) + 1

        print(f"\n  Method Breakdown:")
        for method, count in sorted(method_counts.items()):
            print(f"    {method}: {count}")

        # Count VLM modifications (reasoning contains " -> VLM: " when modified)
        vlm_modified = sum(1 for s in result.strategies if " -> VLM: " in s.reasoning)
        if vlm_modified:
            print(f"\n  VLM Modifications: {vlm_modified}/{len(result.strategies)} strategies adjusted")

        print(f"\n  Strategy Details:")
        for i, s in enumerate(result.strategies, 1):
            method_name = s.recommended_method.value if s.recommended_method else "none"
            action = "PROTECT" if method_name != "none" else "skip"
            vlm_tag = " *VLM*" if " -> VLM: " in s.reasoning else ""
            user_flag = " [USER REVIEW]" if s.requires_user_decision else ""
            print(f"    {i}. [{s.severity.value.upper():>8}] {s.element[:40]} -> {method_name}{vlm_tag}{user_flag}")
            print(f"       {action} | Priority: {s.execution_priority} | {s.reasoning[:70]}")
            print()
