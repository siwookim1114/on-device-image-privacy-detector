"""
Intent Classifier — Hybrid LLM-first + Regex system.

Level 0: Regex fast-path for trivial 1-2 word commands (0ms).
Level 1: LLM structured classification (PRIMARY for multi-word, ~50ms).
Level 2: Regex fallback when LLM unavailable.
Level 3: VLM server fallback (≤1s).
Level 4: Safe default (QUERY, confidence=0.3).

Each public function is independently testable with no external dependencies
so that regression tests can run without llama-server.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agents.text_llm import TextLLM

logger = logging.getLogger(__name__)
# Enumerations (mirrors COORDINATOR_BLUEPRINT.md Section 4)

class IntentAction(str, Enum):
    PROCESS          = "process"          # Full pipeline
    MODIFY_STRATEGY  = "modify_strategy"  # Change protection method / params
    IGNORE           = "ignore"           # Mark element as no-protection-needed
    STRENGTHEN       = "strengthen"       # Increase protection
    QUERY            = "query"            # Read-only explanation
    UNDO             = "undo"             # Revert to last checkpoint
    APPROVE          = "approve"          # Accept current state
    REJECT           = "reject"           # Decline current state
# ParsedIntent (concrete, not the TypedDict skeleton in blueprint)

@dataclass
class ParsedIntent:
    action: IntentAction
    target_stage: str                        # "detect", "risk", "strategy", "execution"
    target_elements: Optional[List[str]]     # detection IDs or None → all
    target_element_types: Optional[List[str]]# ["face", "text", "screen"] or None → all
    confidence: float                        # 0-1; 1.0 if regex-matched
    method_specified: Optional[str]          # "blur", "pixelate", "solid_overlay", "avatar"
    strength_parameter: Optional[float]      # 0-1 for obfuscation strength
    natural_language: str                    # Original user query
    extracted_constraints: Dict              # {"apply_to": "all_faces", ...}
    requires_safety_check: bool              # True if modifying THIRD_PARTY or CRITICAL
    requires_checkpoint: bool                # True if confidence < 0.8 or involves override
    multi_intents: List["ParsedIntent"] = field(default_factory=list)  # decomposed intents


# ── IntentClassification Pydantic model (for TextLLM structured output) ────
class IntentClassification(BaseModel):
    """
    Pydantic model for structured LLM intent classification output.

    Used with TextLLM.call_structured() to get schema-constrained JSON
    from the in-process small LLM. Fields mirror ParsedIntent but use
    Pydantic types for validation and JSON schema generation.
    """
    action: str = Field(
        ...,
        description="One of: process, modify_strategy, ignore, strengthen, query, undo, approve, reject",
    )
    target_stage: str = Field(
        default="",
        description="Pipeline stage: detect, risk, strategy, execution, or empty",
    )
    target_element_types: Optional[List[str]] = Field(
        default=None,
        description="Element types: face, text, screen, object, or null for all",
    )
    method_specified: Optional[str] = Field(
        default=None,
        description="Obfuscation method: blur, pixelate, solid_overlay, avatar_replace, inpaint, or null",
    )
    strength_parameter: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Obfuscation strength 0.0-1.0, or null",
    )
    extracted_constraints: Dict = Field(
        default_factory=dict,
        description="Additional constraints extracted from the query",
    )
    requires_safety_check: bool = Field(
        default=False,
        description="True if action modifies THIRD_PARTY or CRITICAL elements",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification confidence 0.0-1.0",
    )
    reasoning: str = Field(
        default="",
        description="One-sentence reasoning for debugging",
    )

    def to_parsed_intent(self, original_query: str) -> "ParsedIntent":
        """
        Convert this structured LLM output into a ParsedIntent dataclass.

        Handles invalid action strings gracefully by falling back to QUERY.
        Sets requires_checkpoint based on confidence threshold.

        Confidence boosting: Small models often omit the confidence field,
        causing it to default to 0.5. When the LLM explicitly chose a
        non-query action, we boost confidence to 0.85 (the model made a
        deliberate classification choice). This prevents the clarification
        gate from blocking valid LLM classifications.
        """
        try:
            action = IntentAction(self.action.lower())
        except ValueError:
            action = IntentAction.QUERY

        confidence = max(0.0, min(1.0, self.confidence))

        # Boost default confidence for non-query LLM classifications
        # The model chose this action deliberately — trust it
        if confidence == 0.5 and action != IntentAction.QUERY:
            confidence = 0.85

        # Extract element types from query if LLM didn't provide them
        target_types = self.target_element_types
        if not target_types:
            target_types = _extract_element_types(original_query)

        # Extract method from query if LLM didn't provide it, and normalize
        # LLM aliases (e.g. "avatar" → "avatar_replace", "black box" → "solid_overlay")
        method = self.method_specified
        if method:
            method = _normalize_method(method) or method
        if not method:
            method = _extract_method(original_query)

        # Extract current_method constraint if LLM didn't provide it
        # Detects when user references a current method ("black box", "blurred", "avatar")
        constraints = dict(self.extracted_constraints)
        if "current_method" not in constraints:
            current = _extract_current_method(original_query)
            if current:
                constraints["current_method"] = current

        return ParsedIntent(
            action=action,
            target_stage=self.target_stage or "",
            target_elements=None,
            target_element_types=target_types,
            confidence=confidence,
            method_specified=method,
            strength_parameter=self.strength_parameter,
            natural_language=original_query,
            extracted_constraints=constraints,
            requires_safety_check=self.requires_safety_check,
            requires_checkpoint=(confidence < 0.8),
        )


# ── LLM intent prompt builder (for TextLLM) ───────────────────────────────

_LLM_INTENT_SYSTEM_PROMPT = """\
Classify user commands for image privacy. Return JSON with ALL fields filled.

action: process | modify_strategy | ignore | strengthen | query | undo | approve | reject
target_element_types: REQUIRED array - ["face"], ["text"], ["screen"], ["object"]. "values"/"sensitive"/"black box"/"numbers" = ["text"]. "person"/"people" = ["face"]. "laptop"/"monitor"/"computer" = ["screen"]. null ONLY for query/approve/reject/undo.
method_specified: blur | pixelate | solid_overlay | avatar_replace | inpaint | none | null. "black box" = solid_overlay. "avatar"/"cartoon" = avatar_replace. For "from X to Y" or "X instead of Y", method = Y (the NEW one).
confidence: 0.85-1.0 for clear commands, 0.6-0.8 for ambiguous

Rules:
- "No, ..." followed by instructions = refinement (modify_strategy), NOT reject. Only "no"/"nope"/"reject" alone = reject.
- "from X to Y" or "instead of X" -> method_specified = the NEW target method, not the old one.
- When user references a CURRENT protection ("black box"/"blurred"/"avatar face"), set extracted_constraints.current_method to the method name (solid_overlay/blur/avatar_replace). This filters to only elements using that method.
- "it"/"them"/"those" refer to elements from the last action. Check "Last action" in context.
- "back"/"revert"/"change back" = modify_strategy. method_specified should be the PREVIOUS method. Set extracted_constraints.current_method to the current method.

Examples:
Q: "blur the face" -> {"action":"modify_strategy","target_element_types":["face"],"method_specified":"blur","confidence":0.95}
Q: "change black boxes to blur" -> {"action":"modify_strategy","target_element_types":["text"],"method_specified":"blur","extracted_constraints":{"current_method":"solid_overlay"},"confidence":0.9}
Q: "remove laptop protection" -> {"action":"ignore","target_element_types":["screen"],"method_specified":null,"confidence":0.9}
Q: "what risks were found?" -> {"action":"query","target_element_types":null,"method_specified":null,"confidence":0.9}
Q: "change face from avatar to blur" -> {"action":"modify_strategy","target_element_types":["face"],"method_specified":"blur","confidence":0.95}
Q: "no, only apply to the actual values not labels" -> {"action":"modify_strategy","target_element_types":["text"],"method_specified":null,"confidence":0.85}
Q: "pixelate all text" -> {"action":"modify_strategy","target_element_types":["text"],"method_specified":"pixelate","confidence":0.95}
Q: (Last action: Changed 4 text from solid_overlay to blur) "change it back to black box" -> {"action":"modify_strategy","target_element_types":["text"],"method_specified":"solid_overlay","extracted_constraints":{"current_method":"blur"},"confidence":0.95}"""

_LLM_INTENT_USER_TEMPLATE = """\
Pipeline context:
  stage: {current_stage} | elements: {element_count} | critical: {has_critical} | pending_hitl: {pending_hitl}

Recent conversation:
{conversation_tail}

Last action: {last_action_summary}

User message: "{user_query}"
"""


def build_intent_llm_prompt(
    user_query: str,
    current_stage: str = "unknown",
    element_count: int = 0,
    has_critical: bool = False,
    pending_hitl: bool = False,
    recent_actions: Optional[List[str]] = None,
    conversation_tail: str = "",
    last_action_summary: str = "none",
) -> Tuple[str, str]:
    """
    Build (system_prompt, user_message) pair for the in-process TextLLM
    intent classifier.

    Returns a tuple suitable for TextLLM.call_structured().
    """
    user_msg = _LLM_INTENT_USER_TEMPLATE.format(
        current_stage=current_stage,
        element_count=element_count,
        has_critical=str(has_critical).lower(),
        pending_hitl=str(pending_hitl).lower(),
        conversation_tail=conversation_tail or "none",
        last_action_summary=last_action_summary or "none",
        user_query=user_query,
    )
    return _LLM_INTENT_SYSTEM_PROMPT, user_msg


# 1.1  Complete Regex Rule Set
#
# Design notes:
#  - Patterns are ordered: more specific first, general last.
#  - Each pattern is a (compiled_regex, IntentAction, handler_fn) triple.
#  - handler_fn enriches the ParsedIntent after action is assigned.
#  - Negative lookbehind guards prevent "don't blur" matching MODIFY_STRATEGY.

# ── Element-type extractor (shared) ────────────────────────────────────────
_ELEMENT_TYPE_PATTERN = re.compile(
    r"\b(face|faces|person|people|text|texts?|screen|screens?|"
    r"laptop|laptops?|monitor|monitors?|computer|computers?|"
    r"phone|phones?|tv|television|televisions?|"
    r"label|labels?|number|numbers?|object|objects?)\b",
    re.IGNORECASE,
)
_ELEMENT_TYPE_MAP = {
    "face": "face", "faces": "face", "person": "face", "people": "face",
    "text": "text", "texts": "text",
    "screen": "screen", "screens": "screen",
    "laptop": "screen", "laptops": "screen",
    "monitor": "screen", "monitors": "screen",
    "computer": "screen", "computers": "screen",
    "phone": "screen", "phones": "screen",
    "tv": "screen", "television": "screen", "televisions": "screen",
    "label": "text", "labels": "text",
    "number": "text", "numbers": "text",
    "object": "object", "objects": "object",
}

# ── Method extractor (shared) ───────────────────────────────────────────────
_METHOD_PATTERN = re.compile(
    r"\b(blur|pixelate|pixelation|solid[\s_]?overlay|black[\s_]?box|"
    r"avatar|emoji|cartoon|silhouette|inpaint|generative)\b",
    re.IGNORECASE,
)
_METHOD_MAP = {
    "blur": "blur", "blurred": "blur", "blurring": "blur",
    "pixelate": "pixelate", "pixelation": "pixelate", "pixelated": "pixelate",
    "solid overlay": "solid_overlay", "solidoverlay": "solid_overlay",
    "solid_overlay": "solid_overlay", "overlay": "solid_overlay",
    "black box": "solid_overlay", "black_box": "solid_overlay", "blackbox": "solid_overlay",
    "avatar": "avatar_replace", "avatar_replace": "avatar_replace",
    "emoji": "avatar_replace", "cartoon": "avatar_replace",
    "silhouette": "silhouette",
    "inpaint": "inpaint",
    "generative": "generative_replace", "generative_replace": "generative_replace",
}

# ── Strength extractor (shared) ─────────────────────────────────────────────
_STRENGTH_PATTERN = re.compile(
    r"\b(strong(?:er)?|heavy|maximum|max|light(?:er)?|soft|minimal|medium|moderate)\b",
    re.IGNORECASE,
)
_STRENGTH_MAP = {
    "stronger": 0.9, "strong": 0.8, "heavy": 0.9, "maximum": 1.0, "max": 1.0,
    "lighter": 0.3, "light": 0.3, "soft": 0.3, "minimal": 0.2,
    "medium": 0.5, "moderate": 0.5,
}


def _extract_element_types(text: str) -> Optional[List[str]]:
    hits = _ELEMENT_TYPE_PATTERN.findall(text)
    types = list(dict.fromkeys(
        _ELEMENT_TYPE_MAP[h.lower()] for h in hits if h.lower() in _ELEMENT_TYPE_MAP
    ))
    return types if types else None


_FROM_TO_METHOD_PATTERN = re.compile(
    r"(?:from\s+)?"
    r"(?:blur|pixelate|pixelation|solid[\s_]?overlay|black[\s_]?box|"
    r"avatar|emoji|cartoon|silhouette|inpaint|generative)"
    r"\s+(?:to|into|with)\s+"
    r"(blur|pixelate|pixelation|solid[\s_]?overlay|black[\s_]?box|"
    r"avatar|emoji|cartoon|silhouette|inpaint|generative)",
    re.IGNORECASE,
)


def _normalize_method(raw: str) -> Optional[str]:
    """Normalize a raw method string to the canonical method name."""
    key = raw.lower().replace(" ", "_")
    return _METHOD_MAP.get(key, _METHOD_MAP.get(raw.lower()))


def _extract_method(text: str) -> Optional[str]:
    # First try "from X to Y" or "X to Y" pattern — return Y (the target)
    from_to = _FROM_TO_METHOD_PATTERN.search(text)
    if from_to:
        return _normalize_method(from_to.group(1))

    # Fallback: single method extraction (first match)
    m = _METHOD_PATTERN.search(text)
    if not m:
        return None
    raw = m.group(0).lower().replace(" ", "_")
    return _METHOD_MAP.get(raw, _METHOD_MAP.get(m.group(0).lower()))


def _extract_strength(text: str) -> Optional[float]:
    m = _STRENGTH_PATTERN.search(text)
    if not m:
        return None
    return _STRENGTH_MAP.get(m.group(0).lower())


def _extract_current_method(text: str) -> Optional[str]:
    """Extract reference to a CURRENT protection method from the query.

    Detects when user refers to existing protection by visual appearance:
    "black box" -> solid_overlay, "blurred"/"blur" in "from blur" -> blur, etc.

    Only extracts when there's a "from...to" or "change [method]" pattern,
    not when the method is the TARGET (which is handled by _extract_method).
    """
    # Pattern: "change/from [current_method] ... to [new_method]"
    # Allow up to 3 extra words between method and "to" (e.g., "blurred face to")
    from_to = re.search(
        r"(?:change|from|replace)\s+(?:the\s+)?"
        r"(blur(?:red)?|pixelat(?:e|ed|ion)|solid[\s_]?overlay|black[\s_]?box(?:es)?|"
        r"avatar(?:\s+face)?|cartoon|emoji|silhouette)"
        r"(?:\s+\w+){0,3}\s+(?:to|into|with|for)\b",
        text, re.IGNORECASE,
    )
    if from_to:
        raw = from_to.group(1).lower().strip()
        # Normalize to canonical method name
        if "black" in raw or "solid" in raw:
            return "solid_overlay"
        if "blur" in raw:
            return "blur"
        if "pixelat" in raw:
            return "pixelate"
        if "silhouette" in raw:
            return "silhouette"
        if "avatar" in raw or "cartoon" in raw or "emoji" in raw:
            return "avatar_replace"

    # Pattern: "black box(es) protecting..." (reference to current solid_overlay)
    # But NOT when "black box" appears after "to"/"back to"/"with"/"into" (that's the TARGET method)
    bb_match = re.search(r"\bblack[\s_]?box(?:es)?\b", text, re.IGNORECASE)
    if bb_match:
        before = text[:bb_match.start()]
        # If "to", "back to", "with", or "into" appears before the match, it's the target
        if not re.search(r"\b(?:to|with|into)\b", before, re.IGNORECASE):
            return "solid_overlay"

    return None


def _target_stage_for_method_change() -> str:
    return "strategy"


def _target_stage_for_strengthen() -> str:
    return "execution"


# ── Pattern table ────────────────────────────────────────────────────────────
#
# Format: (pattern_str, action, target_stage, notes)
# Patterns are tried in ORDER; first match wins.
# Use word boundaries (\b) and anchors where needed to prevent false matches.

_RAW_PATTERNS: List[Tuple[str, IntentAction, str, str]] = [
    # ── REJECT / APPROVE (before IGNORE to prevent "no" matching both) ────
    (
        r"^(?:no|nope|reject|cancel|decline)\b(?!\s*[,.]?\s*\w{2,})"
        r"|^don['']t(?:\s+do\s+(?:it|that))?\b(?!\s*protect)"
        r"|^revert\s+(?:changes?|that)\b",
        IntentAction.REJECT, "", "Negative confirmation"
    ),
    (
        r"^(?:yes|yeah|yep|ok(?:ay)?|sure|approve|confirm|accept|"
        r"looks?\s+good|proceed|go\s+ahead|apply\s+(?:it|changes?|that))\b",
        IntentAction.APPROVE, "", "Positive confirmation"
    ),

    # ── UNDO ──────────────────────────────────────────────────────────────
    (
        r"\b(?:undo|revert|go\s+back|previous\s+state|restore\s+(?:original|previous)|"
        r"roll\s*back)\b",
        IntentAction.UNDO, "", "Undo / revert"
    ),

    # ── QUERY ─────────────────────────────────────────────────────────────
    (
        r"\b(?:why|explain|how\s+(?:did|does|come)|what\s+(?:is|was|are|were)|"
        r"reason|rationale|tell\s+me|show\s+me\s+(?:the\s+)?(?:reason|why|how)|"
        r"what\s+(?:protection|method|strategy)\s+(?:is|was|are))\b",
        IntentAction.QUERY, "", "Explanation query"
    ),

    # ── IGNORE (before MODIFY to prevent "remove protection" matching both) ─
    # Note: "don't X" is matched here ONLY when followed by "protect".
    # Plain "don't" / "don't do it" is already captured by REJECT above.
    (
        r"\b(?:ignore|skip|no\s+protection|"
        r"(?:remove|clear|delete)\s+(?:the\s+)?protection|"
        r"leave\s+(?:it\s+)?(?:as[\s-]?is|unprotected|alone)|"
        r"unprotect|exclude)\b"
        r"|don['']t\s+protect",
        IntentAction.IGNORE, "strategy", "Remove / ignore protection"
    ),

    # ── STRENGTHEN ────────────────────────────────────────────────────────
    (
        r"\b(?:stronger|more\s+protection|increase\s+(?:protection|strength|blur)|"
        r"max(?:imize)?\s+(?:protection|privacy)|heavier\s+(?:blur|protection)|"
        r"expand\s+(?:the\s+)?(?:mask|region|area|protection)|"
        r"reinforce|double\s+protect)\b",
        IntentAction.STRENGTHEN, "execution", "Strengthen existing protection"
    ),

    # ── MODIFY_STRATEGY — specific method mentioned ──────────────────────
    (
        r"(?:(?:make|set|change|switch|use|apply)\s+(?:it\s+|that\s+|the\s+)?)"
        r"(?:to\s+)?(?:blur|pixelate|solid[\s_]?overlay|black\s+box|avatar|emoji)",
        IntentAction.MODIFY_STRATEGY, "strategy", "Method change with explicit method"
    ),
    (
        r"(?:blur|pixelate|solid[\s_]?overlay|avatar)\s+"
        r"(?:the\s+)?(?:face|faces?|person|people|text|screen|object)",
        IntentAction.MODIFY_STRATEGY, "strategy", "Verb-method + element type"
    ),

    # ── MODIFY_STRATEGY — "change [element] to [method]" (element-first) ──
    # Also handles "change [element] back to [method]"
    (
        r"(?:change|modify|switch|replace|convert)\s+"
        r"(?:the\s+)?(?:face|faces?|person|people|text|screen|object|label|labels?)"
        r"\s+(?:back\s+)?(?:to|with|into)\s+"
        r"(?:blur|pixelate|pixelation|solid[\s_]?overlay|black[\s_]?box|avatar|emoji|cartoon|silhouette|inpaint|none)",
        IntentAction.MODIFY_STRATEGY, "strategy", "Change element to method"
    ),
    # ── MODIFY_STRATEGY — "[method] instead of [method]" ──
    (
        r"(?:blur|pixelate|solid[\s_]?overlay|avatar|emoji|cartoon|silhouette|inpaint)"
        r"\s+(?:instead\s+of|rather\s+than|not)\s+"
        r"(?:blur|pixelate|solid[\s_]?overlay|avatar|emoji|cartoon|silhouette|inpaint)",
        IntentAction.MODIFY_STRATEGY, "strategy", "Method swap"
    ),
    # ── MODIFY_STRATEGY — "use [method] for/on [element]" ──
    (
        r"(?:use|apply|try)\s+"
        r"(?:blur|pixelate|solid[\s_]?overlay|avatar|emoji|cartoon|silhouette|inpaint)"
        r"\s+(?:for|on|to)\s+(?:the\s+)?(?:face|faces?|person|people|text|screen|object)",
        IntentAction.MODIFY_STRATEGY, "strategy", "Use method on element"
    ),

    # ── PROCESS ──────────────────────────────────────────────────────────
    (
        r"^(?:process|analyze|analyse|detect|run|start|scan|go|protect|"
        r"begin|execute|(?:run\s+)?(?:the\s+)?(?:full\s+)?pipeline)\b",
        IntentAction.PROCESS, "detect", "Full pipeline trigger"
    ),
    (
        r"\b(?:process\s+(?:the\s+)?image|analyze\s+(?:the\s+)?(?:photo|image|picture)|"
        r"scan\s+(?:the\s+)?image|find\s+(?:all\s+)?(?:faces|text|sensitive))\b",
        IntentAction.PROCESS, "detect", "Process image (phrase form)"
    ),

    # ── MODIFY_STRATEGY — generic ─────────────────────────────────────────
    (
        r"\b(?:change|modify|update|adjust|tweak|alter)\s+"
        r"(?:the\s+)?(?:protection|method|strategy|approach|technique)\b",
        IntentAction.MODIFY_STRATEGY, "strategy", "Generic strategy change"
    ),
]

# Compile all patterns once at module load
_COMPILED_PATTERNS: List[Tuple[re.Pattern, IntentAction, str]] = [
    (re.compile(pat, re.IGNORECASE), action, stage)
    for pat, action, stage, _ in _RAW_PATTERNS
]
# 1.2  Multi-intent splitter
#
# Splits compound queries on conjunctions before classifying each part.
# "Blur the face AND make text stronger" → two ParsedIntents.

_CONJUNCTION_SPLIT = re.compile(
    r"\s*(?:\band\b|;|,\s*(?:also|and|additionally|then))\s*",
    re.IGNORECASE,
)


def _split_multi_intent(query: str) -> List[str]:
    """Split compound NL queries into individual intent strings."""
    parts = _CONJUNCTION_SPLIT.split(query.strip())
    # Only return multiple parts if each part looks like a standalone command
    # (has a verb or keyword); otherwise keep original to avoid over-splitting.
    cleaned = [p.strip() for p in parts if p.strip()]
    if len(cleaned) < 2:
        return [query]
    # Reject split if any part is a single word (likely false conjunction split)
    if any(len(p.split()) < 2 for p in cleaned):
        return [query]
    return cleaned
# 1.3  Regex classifier — Level 1

def _regex_classify(query: str) -> Optional[ParsedIntent]:
    """
    Attempt classification via regex.
    Returns ParsedIntent with confidence=1.0 if matched, else None.
    """
    q = query.strip()

    for compiled, action, target_stage in _COMPILED_PATTERNS:
        if compiled.search(q):
            extracted_constraints = {}
            # Extract current_method for strategy-modifying intents
            if action in {IntentAction.MODIFY_STRATEGY, IntentAction.IGNORE, IntentAction.STRENGTHEN}:
                current = _extract_current_method(query)
                if current:
                    extracted_constraints["current_method"] = current
            return ParsedIntent(
                action=action,
                target_stage=target_stage,
                target_elements=None,
                target_element_types=_extract_element_types(q),
                confidence=1.0,
                method_specified=_extract_method(q),
                strength_parameter=_extract_strength(q),
                natural_language=query,
                extracted_constraints=extracted_constraints,
                requires_safety_check=(action in {
                    IntentAction.IGNORE, IntentAction.MODIFY_STRATEGY
                }),
                requires_checkpoint=False,
            )
    return None
# 1.4  VLM Classification — Level 2
#
# Used only when regex returns None (ambiguous query).
# Designed for Qwen3-VL-30B via llama-server (OpenAI-compatible /v1/chat).
# Returns structured JSON that is parsed into ParsedIntent.

# ── Prompt template ─────────────────────────────────────────────────────────
VLM_INTENT_SYSTEM_PROMPT = """\
You are an intent classifier for an on-device image privacy protection system.
The user interacts with an active pipeline session that has already processed an image.

Current pipeline state context will be provided in each call.

You must classify the user's message into exactly one primary intent action
and produce a JSON object — nothing else. No markdown, no explanation outside JSON.

Intent action definitions:
- process       : User wants to run the full pipeline on a new image
- modify_strategy: User wants to change which protection method is used
                   (blur / pixelate / solid_overlay / avatar_replace / inpaint)
- ignore        : User wants to remove or skip protection for a specific element
- strengthen    : User wants to make existing protection stronger or expand coverage
- query         : User wants an explanation of a past decision (read-only)
- undo          : User wants to revert to a previous pipeline state
- approve       : User accepts the current output and wants to proceed / commit
- reject        : User declines the current output and wants to cancel changes

Output schema (strict JSON):
{
  "action": "<one of the 8 actions above>",
  "target_stage": "<detect|risk|strategy|execution|none>",
  "target_element_types": ["face", "text", "screen"] or null,
  "method_specified": "<blur|pixelate|solid_overlay|avatar_replace|inpaint|null>",
  "strength_parameter": <0.0-1.0 or null>,
  "extracted_constraints": {},
  "requires_safety_check": <true|false>,
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence, used only for debugging>"
}

Confidence scoring rules:
- 0.9-1.0 : One clearly dominant action; unambiguous phrasing
- 0.7-0.89: Action is clear but element target is vague
- 0.5-0.69: Two plausible actions; context needed
- 0.0-0.49: Highly ambiguous; system will ask user to clarify

IMPORTANT disambiguation rules:
- A sentence starting with "No" or "No," followed by instructions is NOT a reject — it is a \
refinement (modify_strategy or ignore). Only classify as reject when "no" is the entire intent \
(e.g., "No", "Nope", "Reject this").
- "from X to Y" or "X to Y" where X and Y are methods → modify_strategy with method_specified = Y (the target)
- "change [element] of/from [current_method] to [new_method]" → modify_strategy with method_specified = new_method
- "only apply to X not Y" or "not labels" → modify_strategy with extracted_constraints about scope

If the query is genuinely ambiguous between two actions, set confidence ≤ 0.5
and choose the SAFER action (the one that does less, e.g. query over modify_strategy).
"""

VLM_INTENT_USER_TEMPLATE = """\
Current pipeline context:
  stage         : {current_stage}
  element_count : {element_count}
  has_critical  : {has_critical}
  pending_hitl  : {pending_hitl}

Recent conversation:
{conversation_tail}

Last action: {last_action_summary}

User message: "{user_query}"

Classify the intent and return ONLY the JSON object defined in the system prompt.
"""


def build_vlm_intent_prompt(
    user_query: str,
    current_stage: str = "unknown",
    element_count: int = 0,
    has_critical: bool = False,
    pending_hitl: bool = False,
    recent_actions: Optional[List[str]] = None,
    conversation_tail: str = "",
    last_action_summary: str = "none",
) -> Tuple[str, str]:
    """
    Build (system_prompt, user_message) pair for the VLM intent classifier.

    Returns a tuple suitable for ChatOpenAI with system + user messages.
    Caller is responsible for assembling the LangChain message list.
    """
    user_msg = VLM_INTENT_USER_TEMPLATE.format(
        current_stage=current_stage,
        element_count=element_count,
        has_critical=str(has_critical).lower(),
        pending_hitl=str(pending_hitl).lower(),
        conversation_tail=conversation_tail or "none",
        last_action_summary=last_action_summary or "none",
        user_query=user_query,
    )
    return VLM_INTENT_SYSTEM_PROMPT, user_msg


def parse_vlm_intent_response(
    raw_response: str,
    original_query: str,
) -> ParsedIntent:
    """
    Parse the VLM JSON response into a ParsedIntent.

    Applies fallback defaults if the JSON is malformed so callers always
    receive a valid ParsedIntent (never raises on bad VLM output).
    """
    try:
        # Strip potential markdown code fence
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        data = json.loads(cleaned)

        action_str = data.get("action", "query").lower()
        try:
            action = IntentAction(action_str)
        except ValueError:
            action = IntentAction.QUERY  # Safest fallback

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        # Extract element types from query if VLM didn't provide them
        target_types = data.get("target_element_types")
        if not target_types:
            target_types = _extract_element_types(original_query)

        # Extract method from query if VLM didn't provide it, and normalize
        # VLM aliases (e.g. "avatar" → "avatar_replace", "black box" → "solid_overlay")
        method = data.get("method_specified")
        if method:
            method = _normalize_method(method) or method
        if not method:
            method = _extract_method(original_query)

        # Extract current_method constraint if VLM didn't provide it
        constraints = data.get("extracted_constraints", {})
        if isinstance(constraints, dict) and "current_method" not in constraints:
            current = _extract_current_method(original_query)
            if current:
                constraints["current_method"] = current

        return ParsedIntent(
            action=action,
            target_stage=data.get("target_stage", "strategy"),
            target_elements=None,
            target_element_types=target_types,
            confidence=confidence,
            method_specified=method,
            strength_parameter=data.get("strength_parameter"),
            natural_language=original_query,
            extracted_constraints=constraints,
            requires_safety_check=bool(data.get("requires_safety_check", False)),
            requires_checkpoint=(confidence < 0.8),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        # VLM produced non-JSON: default to QUERY (safest read-only action)
        return ParsedIntent(
            action=IntentAction.QUERY,
            target_stage="",
            target_elements=None,
            target_element_types=None,
            confidence=0.3,
            method_specified=None,
            strength_parameter=None,
            natural_language=original_query,
            extracted_constraints={},
            requires_safety_check=False,
            requires_checkpoint=True,
        )
# 1.5  Multi-intent decomposition

def decompose_multi_intent(
    query: str,
    classify_fn,  # callable: str -> ParsedIntent
) -> ParsedIntent:
    """
    Detect compound queries and decompose them into a primary ParsedIntent
    whose `multi_intents` list holds the individual sub-intents.

    classify_fn is the full hybrid_classify function (injected to avoid
    circular dependency).

    Examples:
      "blur the face and make text stronger"
      → primary = modify_strategy (face), multi_intents = [strengthen (text)]

      "yes, but skip the screen"
      → primary = approve, multi_intents = [ignore (screen)]
    """
    parts = _split_multi_intent(query)
    if len(parts) == 1:
        return classify_fn(query)

    sub_intents = [classify_fn(part) for part in parts]
    if not sub_intents:
        return classify_fn(query)

    # Primary intent = highest-priority action among sub-intents
    # Priority: REJECT > APPROVE > STRENGTHEN > IGNORE > MODIFY_STRATEGY > PROCESS > QUERY > UNDO
    _ACTION_PRIORITY = {
        IntentAction.REJECT: 8,
        IntentAction.APPROVE: 7,
        IntentAction.STRENGTHEN: 6,
        IntentAction.IGNORE: 5,
        IntentAction.MODIFY_STRATEGY: 4,
        IntentAction.PROCESS: 3,
        IntentAction.QUERY: 2,
        IntentAction.UNDO: 1,
    }
    primary = max(sub_intents, key=lambda si: _ACTION_PRIORITY.get(si.action, 0))
    secondary = [si for si in sub_intents if si is not primary]

    # Aggregate element types across all sub-intents
    all_types: List[str] = []
    for si in sub_intents:
        if si.target_element_types:
            all_types.extend(si.target_element_types)

    primary.multi_intents = secondary
    if all_types:
        primary.target_element_types = list(dict.fromkeys(all_types))

    # Composite confidence: geometric mean of sub-intent confidences
    confidences = [si.confidence for si in sub_intents]
    composite = 1.0
    for c in confidences:
        composite *= c
    primary.confidence = composite ** (1.0 / len(confidences))

    return primary
# 1.6  Confidence calibration — when to ask user to clarify

# Threshold below which the coordinator should ask for clarification
# instead of acting on the classification.
CLARIFICATION_THRESHOLD = 0.50

# Threshold below which requires_checkpoint is forced True
CHECKPOINT_CONFIDENCE_THRESHOLD = 0.80

# Action-specific confidence floors:
# Some actions are inherently higher-stakes and need higher confidence.
_ACTION_CONFIDENCE_FLOORS: Dict[IntentAction, float] = {
    IntentAction.IGNORE: 0.70,       # Removing protection needs clearer signal
    IntentAction.REJECT: 0.65,       # Reverting changes is recoverable but notable
    IntentAction.MODIFY_STRATEGY: 0.55,
    IntentAction.STRENGTHEN: 0.55,
    IntentAction.APPROVE: 0.60,
    IntentAction.PROCESS: 0.55,
    IntentAction.QUERY: 0.40,        # Read-only; lower bar acceptable
    IntentAction.UNDO: 0.60,
}


def needs_clarification(intent: ParsedIntent) -> bool:
    """
    Returns True if the coordinator should ask the user to clarify
    their query rather than acting on it.

    Uses per-action confidence floors so that high-stakes actions
    require stronger classification signal.
    """
    floor = _ACTION_CONFIDENCE_FLOORS.get(intent.action, CLARIFICATION_THRESHOLD)
    effective_threshold = max(CLARIFICATION_THRESHOLD, floor)
    return intent.confidence < effective_threshold


def generate_clarification_prompt(intent: ParsedIntent) -> str:
    """
    Generate a natural-language clarification request when confidence is low.
    Suggests what the system thinks the user meant so the user can confirm.
    """
    action_labels = {
        IntentAction.PROCESS: "run the full pipeline",
        IntentAction.MODIFY_STRATEGY: "change the protection method",
        IntentAction.IGNORE: "remove protection from an element",
        IntentAction.STRENGTHEN: "make protection stronger",
        IntentAction.QUERY: "get an explanation",
        IntentAction.UNDO: "undo recent changes",
        IntentAction.APPROVE: "approve and proceed",
        IntentAction.REJECT: "reject and revert",
    }
    best_guess = action_labels.get(intent.action, intent.action.value)
    return (
        f"I'm not sure I understood — did you want to {best_guess}? "
        f"(confidence: {intent.confidence:.0%}). "
        f"Please confirm or rephrase, e.g. '{intent.natural_language}'"
    )
# 1.7  Full hybrid classifier (public API)

def hybrid_classify(
    query: str,
    vlm_call_fn=None,               # Optional callable: (system, user) -> str
    context: Optional[Dict] = None,  # Pipeline context for VLM prompt
    intent_llm: Optional["TextLLM"] = None,  # In-process small LLM (preferred over vlm_call_fn)
) -> ParsedIntent:
    """
    LLM-first hybrid classifier.

    For short (1-2 word) inputs, regex handles them instantly and unambiguously.
    For longer (3+ word) inputs, the LLM is ALWAYS tried first because regex
    patterns can match too aggressively (e.g., "No, only apply blur to values"
    starts with "No" which triggers the REJECT regex, misclassifying a
    refinement command as a rejection).

    Classification levels:
      Level 0: Regex fast-path for trivial 1-2 word commands ("yes", "undo").
      Level 1: LLM structured classification (PRIMARY for 3+ words).
      Level 2: Regex fallback (when LLM is unavailable).
      Level 3: VLM server fallback.
      Level 4: Safe default (QUERY, confidence=0.3).

    Args:
        query:       Raw user natural language string.
        vlm_call_fn: Callable(system_prompt: str, user_msg: str) -> str.
                     Must return the raw VLM response text.
                     If None and intent_llm is None, falls back to QUERY with confidence=0.3.
        context:     Dict with keys: current_stage, element_count,
                     has_critical, pending_hitl, recent_actions.
        intent_llm:  In-process TextLLM instance for structured classification.
                     When provided, used as the primary classifier for multi-word inputs.

    Returns ParsedIntent. Never raises.
    """
    ctx = context or {}

    def _classify_single(q: str) -> ParsedIntent:
        word_count = len(q.split())

        # Level 0: Trivial 1-2 word commands — regex only
        # Unambiguous short inputs like "approve", "undo", "yes", "no", "reject"
        if word_count <= 2:
            result = _regex_classify(q)
            if result is not None:
                return result

        # Level 1: LLM structured classification (PRIMARY for multi-word)
        if intent_llm is not None:
            system_prompt, user_msg = build_intent_llm_prompt(
                user_query=q,
                current_stage=ctx.get("current_stage", "unknown"),
                element_count=ctx.get("element_count", 0),
                has_critical=ctx.get("has_critical", False),
                pending_hitl=ctx.get("pending_hitl", False),
                recent_actions=ctx.get("recent_actions"),
                conversation_tail=ctx.get("conversation_tail", ""),
                last_action_summary=ctx.get("last_action_summary", "none"),
            )
            try:
                classification = intent_llm.call_structured(
                    system_prompt=system_prompt,
                    user_message=user_msg,
                    response_model=IntentClassification,
                    max_tokens=256,
                )
                return classification.to_parsed_intent(q)
            except Exception as exc:
                logger.warning("Intent LLM failed: %s", exc)

        # Level 2: Regex fallback (for when LLM is unavailable)
        result = _regex_classify(q)
        if result is not None:
            return result

        # Level 3: VLM server fallback
        if vlm_call_fn is not None:
            system_prompt, user_msg = build_vlm_intent_prompt(
                user_query=q,
                current_stage=ctx.get("current_stage", "unknown"),
                element_count=ctx.get("element_count", 0),
                has_critical=ctx.get("has_critical", False),
                pending_hitl=ctx.get("pending_hitl", False),
                recent_actions=ctx.get("recent_actions"),
                conversation_tail=ctx.get("conversation_tail", ""),
                last_action_summary=ctx.get("last_action_summary", "none"),
            )
            try:
                raw = vlm_call_fn(system_prompt, user_msg)
                return parse_vlm_intent_response(raw, q)
            except Exception as exc:
                logger.warning("VLM intent fallback failed: %s", exc)

        # Level 4: No LLM available — safe default
        return ParsedIntent(
            action=IntentAction.QUERY,
            target_stage="",
            target_elements=None,
            target_element_types=None,
            confidence=0.3,
            method_specified=None,
            strength_parameter=None,
            natural_language=q,
            extracted_constraints={},
            requires_safety_check=False,
            requires_checkpoint=True,
        )

    # Decompose multi-intent before classifying
    result = decompose_multi_intent(query, _classify_single)

    # Force requires_checkpoint for low-confidence results
    if result.confidence < CHECKPOINT_CONFIDENCE_THRESHOLD:
        result.requires_checkpoint = True

    return result
