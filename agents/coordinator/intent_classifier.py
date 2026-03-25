"""
Intent Classifier — Hybrid Regex + VLM two-level system.

Level 1: Regex fast-path (0ms, handles ~80% of queries at confidence=1.0).
Level 2: VLM classification (≤1s, handles ambiguous/complex queries).

Each public function is independently testable with no external dependencies
so that regression tests can run without llama-server.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
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
    r"label|labels?|number|numbers?|object|objects?)\b",
    re.IGNORECASE,
)
_ELEMENT_TYPE_MAP = {
    "face": "face", "faces": "face", "person": "face", "people": "face",
    "text": "text", "texts": "text",
    "screen": "screen", "screens": "screen",
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
    "blur": "blur",
    "pixelate": "pixelate", "pixelation": "pixelate",
    "solid overlay": "solid_overlay", "solidoverlay": "solid_overlay",
    "solid_overlay": "solid_overlay",
    "black box": "solid_overlay", "black_box": "solid_overlay",
    "avatar": "avatar_replace",
    "emoji": "avatar_replace", "cartoon": "avatar_replace",
    "silhouette": "avatar_replace",
    "inpaint": "inpaint",
    "generative": "generative_replace",
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


def _extract_method(text: str) -> Optional[str]:
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
        r"^(?:no|nope|reject|cancel|decline)\b"
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
            return ParsedIntent(
                action=action,
                target_stage=target_stage,
                target_elements=None,
                target_element_types=_extract_element_types(q),
                confidence=1.0,
                method_specified=_extract_method(q),
                strength_parameter=_extract_strength(q),
                natural_language=query,
                extracted_constraints={},
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

If the query is genuinely ambiguous between two actions, set confidence ≤ 0.5
and choose the SAFER action (the one that does less, e.g. query over modify_strategy).
"""

VLM_INTENT_USER_TEMPLATE = """\
Current pipeline context:
  stage         : {current_stage}
  element_count : {element_count}
  has_critical  : {has_critical}
  pending_hitl  : {pending_hitl}
  recent_actions: {recent_actions}

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
        recent_actions=", ".join(recent_actions or []) or "none",
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

        return ParsedIntent(
            action=action,
            target_stage=data.get("target_stage", "strategy"),
            target_elements=None,
            target_element_types=data.get("target_element_types"),
            confidence=confidence,
            method_specified=data.get("method_specified"),
            strength_parameter=data.get("strength_parameter"),
            natural_language=original_query,
            extracted_constraints=data.get("extracted_constraints", {}),
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
) -> ParsedIntent:
    """
    Two-level hybrid classifier.

    1. Try regex fast-path (0ms).
    2. If no match, call VLM (≤1s).
    3. Decompose multi-intents.
    4. Attach requires_checkpoint based on confidence thresholds.

    Args:
        query:       Raw user natural language string.
        vlm_call_fn: Callable(system_prompt: str, user_msg: str) -> str.
                     Must return the raw VLM response text.
                     If None, falls back to QUERY with confidence=0.3.
        context:     Dict with keys: current_stage, element_count,
                     has_critical, pending_hitl, recent_actions.

    Returns ParsedIntent. Never raises.
    """
    ctx = context or {}

    def _classify_single(q: str) -> ParsedIntent:
        # Level 1: regex
        result = _regex_classify(q)
        if result is not None:
            return result

        # Level 2: VLM
        if vlm_call_fn is None:
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

        system_prompt, user_msg = build_vlm_intent_prompt(
            user_query=q,
            current_stage=ctx.get("current_stage", "unknown"),
            element_count=ctx.get("element_count", 0),
            has_critical=ctx.get("has_critical", False),
            pending_hitl=ctx.get("pending_hitl", False),
            recent_actions=ctx.get("recent_actions"),
        )
        try:
            raw = vlm_call_fn(system_prompt, user_msg)
            return parse_vlm_intent_response(raw, q)
        except Exception:
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
