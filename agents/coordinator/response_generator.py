"""
Response Generator for the Coordinator Agent.

Builds natural-language replies from pipeline state and action context.

Two generation paths:
  1. LLM path   -- TextLLM.call() produces a fluent, context-aware response.
  2. Fallback   -- deterministic string returned when TextLLM is None or fails.

Also provides:
  - generate_suggestions()    -- deterministic follow-up action suggestions
  - compress_pipeline_state() -- slim summary dict for prompt injection
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agents.coordinator.tools import _all_assessments, _all_strategies

if TYPE_CHECKING:
    from agents.text_llm import TextLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for the response-generation LLM call
# ---------------------------------------------------------------------------

RESPONSE_SYSTEM_PROMPT = """\
You are the conversational interface for an on-device image privacy protection system.
The system has just completed an action on the user's image. Your job is to summarize
what happened in plain, friendly English so the user understands the outcome.

Rules:
1. Be concise: 1-3 sentences maximum.
2. Use plain language. Never mention model names (MTCNN, YOLO, FaceNet, SAM, etc.).
3. Do NOT reveal the raw content of redacted text (privacy). Say "a phone number"
   or "a credit card number" instead of the actual digits.
4. When protection was applied, mention the method used (blur, pixelate, etc.)
   and which elements were affected.
5. When a safety guard blocked or challenged an action, explain WHY in user terms
   (e.g. "bystander faces must stay protected").
6. End with a brief note on what the user can do next, only if it is non-obvious.
7. Return ONLY plain text. No JSON, no markdown headers, no bullet points.
8. When given specific details about what changed, you MUST incorporate them in
   your response. Be specific: "I changed the face protection from avatar to blur"
   is better than "I made changes to the protection settings".
"""


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def _build_user_message(
    action_taken: str,
    pipeline_summary: Dict[str, Any],
    conversation_tail: List[Dict[str, str]],
    intent_action: str,
    details: str,
) -> str:
    """
    Assemble the user-side prompt that gives the LLM all the context it needs
    to produce a single response message.
    """
    # Last 3 conversation turns for continuity
    tail_lines = []
    for turn in conversation_tail[-3:]:
        role = turn.get("role", "?")
        content = (turn.get("content", "") or "")[:200]
        tail_lines.append(f"  {role}: {content}")
    conversation_block = "\n".join(tail_lines) if tail_lines else "  (none)"

    total = pipeline_summary.get("total_elements", 0)
    protected = pipeline_summary.get("protected_count", 0)
    skipped = pipeline_summary.get("skipped_count", 0)
    critical = pipeline_summary.get("critical_count", 0)
    severity_dist = pipeline_summary.get("severity_distribution", {})

    # Include specific action details from the handler (e.g. modification specifics)
    action_details = pipeline_summary.get("action_details", "")
    pipeline_action = pipeline_summary.get("action_taken", "")

    specific_section = ""
    if action_details:
        specific_section = (
            f"\nWhat specifically happened:\n"
            f"  {action_details}\n"
            f"\nYour response MUST incorporate the specific details above. "
            f"Do not just give generic counts.\n"
        )

    return (
        f"Action taken: {action_taken}\n"
        f"Intent: {intent_action}\n"
        f"Details: {details}\n"
        f"{specific_section}"
        f"\n"
        f"Pipeline summary:\n"
        f"  Total elements detected : {total}\n"
        f"  Elements protected      : {protected}\n"
        f"  Elements skipped (none) : {skipped}\n"
        f"  Critical elements       : {critical}\n"
        f"  Severity distribution   : {severity_dist}\n"
        f"\n"
        f"Recent conversation:\n"
        f"{conversation_block}\n"
        f"\n"
        f"Generate a concise response (1-3 sentences) for the user."
    )


# ---------------------------------------------------------------------------
# ResponseGenerator class
# ---------------------------------------------------------------------------

class ResponseGenerator:
    """
    Generates natural-language coordinator responses.

    When a TextLLM is available, responses are fluent and context-aware.
    When it is None (tests, fallback-only mode), responses are deterministic
    strings built from the ``details`` parameter.
    """

    def __init__(self, text_llm: Optional["TextLLM"] = None) -> None:
        self.text_llm = text_llm

    # ------------------------------------------------------------------
    # Primary: generate a response string
    # ------------------------------------------------------------------

    def generate(
        self,
        action_taken: str,
        pipeline_summary: Dict[str, Any],
        conversation_tail: List[Dict[str, str]],
        intent_action: str,
        details: str = "",
    ) -> str:
        """
        Build and return a user-facing response string.

        Args:
            action_taken:      Short tag describing what happened, e.g.
                               ``"strategy_modified:det_001"`` or ``"pipeline_complete"``.
            pipeline_summary:  Compressed state dict from ``compress_pipeline_state()``.
            conversation_tail: Recent conversation turns (list of role/content dicts).
            intent_action:     The IntentAction value string (e.g. ``"modify_strategy"``).
            details:           Free-form detail string used as the deterministic
                               fallback response when the LLM is unavailable.

        Returns:
            A plain-text response string (1-3 sentences).
        """
        if self.text_llm is None:
            return details or f"Action completed: {action_taken}."

        user_message = _build_user_message(
            action_taken=action_taken,
            pipeline_summary=pipeline_summary,
            conversation_tail=conversation_tail,
            intent_action=intent_action,
            details=details,
        )

        try:
            response = self.text_llm.call(
                system_prompt=RESPONSE_SYSTEM_PROMPT,
                user_message=user_message,
                max_tokens=256,
                temperature=0.0,
            )
            text = (response or "").strip()
            if text:
                return text
        except Exception as exc:
            logger.warning("TextLLM response generation failed: %s", exc)

        # Fallback on LLM failure
        return details or f"Action completed: {action_taken}."

    # ------------------------------------------------------------------
    # Deterministic follow-up suggestions
    # ------------------------------------------------------------------

    def generate_suggestions(
        self,
        pipeline_summary: Dict[str, Any],
        action_taken: str,
    ) -> List[str]:
        """
        Return a list of 0-3 deterministic follow-up suggestions for the UI.

        Suggestions are context-dependent: they reflect what the user is most
        likely to want to do next given the current pipeline state and the
        action that was just performed.
        """
        suggestions: List[str] = []
        protected = pipeline_summary.get("protected_count", 0)
        skipped = pipeline_summary.get("skipped_count", 0)
        critical = pipeline_summary.get("critical_count", 0)
        has_faces = pipeline_summary.get("has_faces", False)
        has_screens = pipeline_summary.get("has_screens", False)
        has_text = pipeline_summary.get("has_text", False)

        # After full pipeline completion
        if "pipeline_complete" in action_taken or "export_complete" in action_taken:
            if protected > 0:
                suggestions.append("Review protected elements and adjust methods if needed.")
            if skipped > 0:
                suggestions.append(
                    f"{skipped} element(s) were skipped. "
                    "Say 'strengthen' to add protection."
                )
            if has_faces and skipped > 0:
                suggestions.append(
                    "You can enroll recognized faces in your consent database "
                    "to skip them automatically next time."
                )
            return suggestions[:3]

        # After a strategy modification
        if "strategy_modified" in action_taken:
            suggestions.append("Say 'approve' to accept and re-export the image.")
            if protected > 1:
                suggestions.append("You can modify other elements too.")
            return suggestions[:3]

        # After an ignore action
        if "ignored" in action_taken:
            suggestions.append("Say 'undo' to restore protection if needed.")
            suggestions.append("Say 'approve' when you are satisfied.")
            return suggestions[:2]

        # After strengthen
        if "strengthened" in action_taken:
            suggestions.append("Say 'approve' to finalize the changes.")
            return suggestions[:2]

        # After undo
        if "undo" in action_taken:
            suggestions.append("The previous state has been restored.")
            return suggestions[:1]

        # Generic suggestions based on state
        if critical > 0:
            suggestions.append(
                f"{critical} critical element(s) detected. "
                "Review before sharing."
            )
        if has_screens:
            suggestions.append("Screens were detected. Check if they need protection.")
        if has_text:
            suggestions.append("Ask 'why was this text redacted?' for an explanation.")

        return suggestions[:3]

    # ------------------------------------------------------------------
    # Pipeline state compression (static)
    # ------------------------------------------------------------------

    @staticmethod
    def compress_pipeline_state(pipeline_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compress the full pipeline state into a slim summary dict suitable
        for prompt injection and suggestion generation.

        Uses ``_all_assessments`` and ``_all_strategies`` from the coordinator
        tools module to extract data from both dict and model-object
        representations of pipeline state.

        Returns a dict with keys:
          total_elements, protected_count, skipped_count, critical_count,
          severity_distribution, has_faces, has_text, has_screens,
          element_types, methods_used
        """
        if pipeline_state is None:
            return {
                "total_elements": 0,
                "protected_count": 0,
                "skipped_count": 0,
                "critical_count": 0,
                "severity_distribution": {},
                "has_faces": False,
                "has_text": False,
                "has_screens": False,
                "element_types": [],
                "methods_used": [],
            }

        assessments = _all_assessments(pipeline_state)
        strategies = _all_strategies(pipeline_state)

        # Severity distribution
        severity_dist: Dict[str, int] = {}
        element_types_set: set = set()
        for a in assessments:
            sev = (a.get("severity") or "unknown").lower()
            severity_dist[sev] = severity_dist.get(sev, 0) + 1
            etype = (a.get("element_type") or "").lower()
            if etype:
                element_types_set.add(etype)

        # Strategy counts
        methods_used: set = set()
        protected_count = 0
        skipped_count = 0
        for s in strategies:
            method = s.get("recommended_method") or s.get("method")
            if hasattr(method, "value"):
                method = method.value
            if method and method != "none":
                protected_count += 1
                methods_used.add(method)
            else:
                skipped_count += 1

        critical_count = severity_dist.get("critical", 0)

        return {
            "total_elements": len(assessments),
            "protected_count": protected_count,
            "skipped_count": skipped_count,
            "critical_count": critical_count,
            "severity_distribution": severity_dist,
            "has_faces": "face" in element_types_set,
            "has_text": "text" in element_types_set,
            "has_screens": "screen" in element_types_set,
            "element_types": sorted(element_types_set),
            "methods_used": sorted(methods_used),
        }
