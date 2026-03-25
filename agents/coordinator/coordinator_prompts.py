"""
VLM Prompt Engineering for the Coordinator Agent.

All prompts are designed for Qwen3-VL-30B-A3B-Instruct via llama-server.

Design constraints:
  - Temperature=0 (already set in VisionLLM)
  - All prompts that need structured output enforce strict JSON-only replies
  - Context injection follows the same pattern used in existing agents:
    state dict → formatted string → system + user message split
  - Prompts assume the VLM does NOT have persistent memory between calls;
    all necessary context must be included in each call

Three prompt sets:
  1. Intent classification  (VLM_INTENT_SYSTEM_PROMPT in intent_classifier.py)
  2. Explanation generation ("Why was this face blurred?")
  3. Session summary        (Pipeline results → human-readable output)

This module defines prompts 2 and 3 only.
Prompt 1 lives in intent_classifier.py to keep the classifier self-contained.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
# Prompt 2: Explanation generation
#
# "Why was this face blurred?"
# Input: detection_id + all pipeline state for that element
# Output: Natural language paragraph suitable for UI display

EXPLAIN_SYSTEM_PROMPT = """\
You are the explanation layer for an on-device image privacy protection system.
You have access to the full pipeline state for a specific detected element.
Your job is to explain in plain, user-friendly language WHY a privacy decision
was made for that element.

Rules:
1. Use plain English. Avoid technical jargon (no "MTCNN", no "cosine similarity").
2. Be honest about uncertainty: if confidence was low, say so.
3. Mention the beneficiary: who benefits from this protection (the person in the
   photo, a bystander, or a legal/compliance requirement).
4. Maximum 3 sentences. Never use bullet points.
5. If the user is asking because they want to change the decision, end with
   what they CAN change vs what is blocked (safety floor).
6. Return ONLY plain text. No JSON, no markdown headers.

Examples of good explanations:
  "This face was blurred because it belongs to a person not recognized in your
   contacts (an unknown bystander). Unknown people have an inherent right to
   privacy. You can change the blur to pixelate or avatar style, but you cannot
   remove protection for an unrecognized person."

  "This text was redacted because it matched a phone number pattern.
   Phone numbers are classified as personal contact information in your
   privacy settings. You can turn this off in your profile settings if
   you want to share contact information."

  "This screen was protected because it was verified as ON and showing visible
   content. The protection method was chosen as blur to preserve some context
   while hiding the displayed information. You can change to pixelate or
   increase the blur strength."
"""

EXPLAIN_USER_TEMPLATE = """\
Element details:
  detection_id   : {detection_id}
  element_type   : {element_type}
  content_summary: {content_summary}
  severity       : {severity}
  risk_type      : {risk_type}
  reasoning      : {reasoning}

Identity / consent (for faces):
  person_label   : {person_label}
  consent_status : {consent_status}
  classification : {classification}

Screen state (for screens):
  screen_state   : {screen_state}

Strategy applied:
  method         : {method}
  parameters     : {parameters}
  safety_floor   : {safety_floor}
  user_can_override : {user_can_override}

User question: "{user_question}"

Explain this privacy decision in plain English (max 3 sentences).
"""


def build_explain_prompt(
    detection_id: str,
    element_type: str,
    risk_assessment: Dict[str, Any],
    strategy: Dict[str, Any],
    safety_floor: Optional[str],
    user_question: str,
) -> tuple[str, str]:
    """
    Build (system_prompt, user_message) for the explanation generator.

    Args:
        detection_id:   The ID of the element being explained.
        element_type:   "face", "text", "screen", "object"
        risk_assessment: Dict with keys: severity, risk_type, reasoning,
                         person_label (opt), consent_status (opt),
                         classification (opt), screen_state (opt)
        strategy:       Dict with keys: method, parameters, user_can_override
        safety_floor:   Short string describing what cannot be changed, or None.
        user_question:  The user's original question (for context injection).

    Returns:
        (system_prompt, user_message) tuple for LangChain message assembly.
    """
    # Content summary: for text show the category not the raw text (privacy)
    if element_type == "text":
        content_summary = f"Text of type: {risk_assessment.get('text_type', 'unknown')}"
    elif element_type == "face":
        label = risk_assessment.get("person_label") or "unrecognized person"
        content_summary = f"Face: {label}"
    elif element_type == "screen":
        state = risk_assessment.get("screen_state", "unknown state")
        content_summary = f"Screen device ({state})"
    else:
        content_summary = f"Object: {element_type}"

    user_msg = EXPLAIN_USER_TEMPLATE.format(
        detection_id=detection_id,
        element_type=element_type,
        content_summary=content_summary,
        severity=risk_assessment.get("severity", "unknown"),
        risk_type=risk_assessment.get("risk_type", "unknown"),
        reasoning=risk_assessment.get("reasoning", "N/A"),
        person_label=risk_assessment.get("person_label") or "N/A",
        consent_status=risk_assessment.get("consent_status") or "N/A",
        classification=risk_assessment.get("classification") or "N/A",
        screen_state=risk_assessment.get("screen_state") or "N/A",
        method=strategy.get("method", "unknown"),
        parameters=str(strategy.get("parameters", {})),
        safety_floor=safety_floor or "None (user has full control)",
        user_can_override=str(strategy.get("user_can_override", True)),
        user_question=user_question,
    )
    return EXPLAIN_SYSTEM_PROMPT, user_msg
# Prompt 3: Session summary generation
#
# "Here's what we did to your image"
# Input: full pipeline state after execution
# Output: Concise human-readable summary for display in the UI sidebar

SUMMARY_SYSTEM_PROMPT = """\
You are generating a human-readable summary of an image privacy protection session.
The user submitted a photo; the system ran detection, risk assessment, and applied
privacy protection. Your job is to summarize what was found and what was done.

Output format — return ONLY this JSON (no markdown):
{
  "headline": "<one sentence: what happened overall>",
  "protected_count": <integer>,
  "skipped_count": <integer>,
  "critical_count": <integer>,
  "key_decisions": ["<decision 1>", "<decision 2>", ...],
  "suggestions": ["<optional user tip 1>", ...],
  "confidence_note": "<one sentence about system confidence, if notable>"
}

Rules:
1. headline: Present tense, e.g. "Protected 4 faces and redacted 2 text items."
2. key_decisions: Up to 5 items. Each is one sentence describing a notable decision.
   Include: what was protected, why notable (consent, severity, screen state),
   what method was used.
3. suggestions: 0-2 items. Only include if actionable. E.g. "You can enroll this
   face in your contact list to avoid re-review in future sessions."
4. confidence_note: Only include if session confidence was below 0.75 or had
   notable uncertainty. Omit (empty string) for high-confidence sessions.
5. Do NOT reveal exact text content that was redacted (privacy).
6. Do NOT mention detection model names or technical implementation.
"""

SUMMARY_USER_TEMPLATE = """\
Pipeline results summary:

Image: {image_path}
Session mode: {mode}
Session confidence score: {confidence_score:.2f}
Total elements detected: {total_elements}
Elements protected: {protected_count}
Elements skipped (method=none): {skipped_count}
Critical elements: {critical_count}

Element breakdown:
{element_breakdown}

Execution status: {execution_status}
Total pipeline time: {total_time_ms:.0f}ms

Generate the JSON summary as specified.
"""


def build_summary_prompt(
    image_path: str,
    mode: str,
    confidence_score: float,
    risk_assessments: List[Dict],
    strategies: List[Dict],
    execution_report: Optional[Dict],
    total_time_ms: float,
) -> tuple[str, str]:
    """
    Build (system_prompt, user_message) for summary generation.
    """
    protected = [s for s in strategies if s.get("method") not in {"none", None}]
    skipped = [s for s in strategies if s.get("method") in {"none", None}]
    critical = [a for a in risk_assessments if a.get("severity") == "critical"]

    # Build concise element breakdown: one line per element
    breakdown_lines = []
    for a in risk_assessments:
        eid = a.get("detection_id", "?")
        etype = a.get("element_type", "?")
        sev = a.get("severity", "?")
        # Find corresponding strategy
        strat = next((s for s in strategies if s.get("detection_id") == eid), {})
        method = strat.get("method", "none")
        consent = a.get("consent_status") or ""
        screen = a.get("screen_state") or ""
        notes = " ".join(filter(None, [
            f"consent={consent}" if consent else "",
            f"screen={screen}" if screen else "",
        ]))
        line = f"  [{etype}] severity={sev} method={method}{' ' + notes if notes else ''}"
        breakdown_lines.append(line)

    exec_status = "completed"
    if execution_report:
        exec_status = execution_report.get("status", "completed")

    user_msg = SUMMARY_USER_TEMPLATE.format(
        image_path=image_path,
        mode=mode,
        confidence_score=confidence_score,
        total_elements=len(risk_assessments),
        protected_count=len(protected),
        skipped_count=len(skipped),
        critical_count=len(critical),
        element_breakdown="\n".join(breakdown_lines) or "  (none)",
        execution_status=exec_status,
        total_time_ms=total_time_ms,
    )
    return SUMMARY_SYSTEM_PROMPT, user_msg


def parse_summary_response(raw_response: str) -> Dict[str, Any]:
    """
    Parse the VLM JSON summary response.
    Returns a plain dict with default values if parsing fails.
    """
    
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        return {
            "headline": str(data.get("headline", "Privacy protection complete.")),
            "protected_count": int(data.get("protected_count", 0)),
            "skipped_count": int(data.get("skipped_count", 0)),
            "critical_count": int(data.get("critical_count", 0)),
            "key_decisions": list(data.get("key_decisions", [])),
            "suggestions": list(data.get("suggestions", [])),
            "confidence_note": str(data.get("confidence_note", "")),
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        return {
            "headline": "Privacy protection complete.",
            "protected_count": 0,
            "skipped_count": 0,
            "critical_count": 0,
            "key_decisions": [],
            "suggestions": [],
            "confidence_note": "Summary generation encountered an error.",
        }
# Prompt 4: Selective re-execution rationale (used by dependency resolver)
#
# When the coordinator decides to skip stages, it calls this prompt to
# generate a human-readable explanation of what was re-run and why.
# This is NOT a decision-making prompt — the dependency graph makes the
# decision; this prompt only produces the explanation.

REEXECUTION_EXPLAIN_TEMPLATE = """\
The user modified: {modification_description}

As a result, the pipeline re-ran these stages: {rerun_stages}
The following stages were skipped (using cached results): {skipped_stages}

Explain in one sentence why these stages were skipped, in plain language.
Example: "Since only the obfuscation method changed, there was no need to
re-run face detection or risk assessment — those results are unchanged."

Return ONLY the one sentence. No JSON, no markdown.
"""


def build_reexecution_explain_prompt(
    modification_description: str,
    rerun_stages: List[str],
    skipped_stages: List[str],
) -> str:
    """
    Build a simple user-message string for re-execution explanation.
    Uses the existing VLM directly (no system prompt needed for this simple task).
    """
    return REEXECUTION_EXPLAIN_TEMPLATE.format(
        modification_description=modification_description,
        rerun_stages=", ".join(rerun_stages) if rerun_stages else "none",
        skipped_stages=", ".join(skipped_stages) if skipped_stages else "none",
    )
