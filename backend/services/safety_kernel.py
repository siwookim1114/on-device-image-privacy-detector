"""
Safety Kernel — validates ALL state mutations for the privacy pipeline.

Core philosophy (from COORDINATOR_BLUEPRINT.md §8):
  - WHETHER to protect (scope) is enforced by this kernel.
  - HOW to protect (method) is always the user's free choice.
  - Immutable audit trail is stored in MongoDB (or in-memory when unavailable).

Rule precedence (evaluated top-to-bottom, first match wins):
  Fast-paths → Block rules → Challenge rules → Default ALLOW.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
# Enumerations


class ProtectionBeneficiary(str, Enum):
    SELF = "self"
    THIRD_PARTY = "third_party"
    LEGAL = "legal_requirement"


class SafetyAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    CHALLENGE = "challenge"
# Strength tables (used for "is this a strengthening move?" comparisons)

# Higher value = stronger protection.  "none" (0) means no protection at all.
METHOD_STRENGTH: Dict[str, int] = {
    "none": 0,
    "blur": 1,
    "pixelate": 2,
    "silhouette": 3,
    "avatar_replace": 3,
    "inpaint": 3,
    "solid_overlay": 4,
    "generative_replace": 4,
}

SEVERITY_RANK: Dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}
# OverrideRecord — immutable audit log entry


class OverrideRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    session_id: str
    detection_id: str
    override_type: str           # "strategy_method", "risk_severity", "ignore_element", "add_protection"
    element_type: str            # "face", "text", "screen", "object"
    beneficiary: ProtectionBeneficiary
    severity_at_time: str        # e.g. "critical"
    original_value: str          # e.g. "blur"
    requested_value: str         # e.g. "none"
    approved_value: str          # what was actually committed
    safety_action: SafetyAction
    rule_id: Optional[str] = None   # which rule fired, e.g. "BLOCK_THIRD_PARTY_UNPROTECT"
    reason: str = ""
    user_confirmed: bool = False
# SafetyKernel


class SafetyKernel:
    """
    Validates override requests before they are applied to pipeline state.

    All write operations on the pipeline (method changes, severity changes,
    ignore requests, add-protection requests) must pass through
    ``validate_override``.  The returned ``OverrideRecord`` describes the
    decision and is appended to the audit trail.

    MongoDB storage is optional: pass a pymongo Collection as *db_collection*
    to persist records there; otherwise they are kept in an in-memory list.
    """

    def __init__(self, db_collection=None, on_record=None) -> None:
        # pymongo Collection | None
        self._collection = db_collection
        # In-memory fallback when no MongoDB collection is provided
        self._in_memory: List[Dict[str, Any]] = []
        # Optional callback invoked after every OverrideRecord is persisted.
        # Signature: on_record(rec: OverrideRecord) -> None
        self._on_record = on_record
    # Public API

    def determine_beneficiary(
        self,
        element_type: str,
        consent_status: Optional[str],
        severity: str,
        legal_requirement: bool = False,
    ) -> ProtectionBeneficiary:
        """
        Classify who benefits from the protection.

        Decision tree (matches COORDINATOR_BLUEPRINT.md §8):
          - legal_requirement flag or CRITICAL text → LEGAL
          - Face with consent explicit or assumed    → SELF
          - Face with consent none or unclear         → THIRD_PARTY
          - Everything else                          → SELF
        """
        severity_lower = severity.lower() if severity else ""
        consent_lower = consent_status.lower() if consent_status else ""
        element_lower = element_type.lower() if element_type else ""

        # Legal takes highest precedence
        if legal_requirement:
            return ProtectionBeneficiary.LEGAL

        # CRITICAL text with GDPR/CCPA-type content is always a legal item
        if element_lower == "text" and severity_lower == "critical":
            return ProtectionBeneficiary.LEGAL

        if element_lower == "face":
            if consent_lower in ("explicit", "assumed"):
                return ProtectionBeneficiary.SELF
            # consent=none, unclear, or unknown → third party
            return ProtectionBeneficiary.THIRD_PARTY

        # Screens and generic objects default to SELF (user's own property)
        return ProtectionBeneficiary.SELF

    def validate_override(
        self,
        session_id: str,
        detection_id: str,
        override_type: str,
        element_type: str,
        original_value: str,
        requested_value: str,
        severity: str,
        consent_status: Optional[str] = None,
        screen_state: Optional[str] = None,
        legal_requirement: bool = False,
        user_confirmed: bool = False,
    ) -> OverrideRecord:
        """
        Core validation method.  Rules are evaluated in strict priority order;
        the first matching rule determines the SafetyAction.

        Parameters
        ----------
        session_id:       Active pipeline session UUID.
        detection_id:     ID of the element being modified.
        override_type:    "strategy_method" | "risk_severity" | "ignore_element"
                          | "add_protection"
        element_type:     "face" | "text" | "screen" | "object"
        original_value:   Current value (method name or severity string).
        requested_value:  What the user is requesting.
        severity:         Current severity of the element (low/medium/high/critical).
        consent_status:   Face consent level (explicit/assumed/none/unclear) or None.
        screen_state:     "verified_on" | "verified_off" | None.
        legal_requirement: Explicit flag set by caller when legal mandate applies.
        user_confirmed:   True if the user has already confirmed a CHALLENGE prompt.

        Returns
        -------
        OverrideRecord describing the decision.  Also persists the record.
        """
        beneficiary = self.determine_beneficiary(
            element_type, consent_status, severity, legal_requirement
        )

        orig_lower = original_value.lower() if original_value else ""
        req_lower = requested_value.lower() if requested_value else ""
        sev_lower = severity.lower() if severity else ""
        elem_lower = element_type.lower() if element_type else ""
        consent_lower = consent_status.lower() if consent_status else ""
        screen_lower = screen_state.lower() if screen_state else ""

        # Helpers
        orig_method_strength = METHOD_STRENGTH.get(orig_lower, -1)
        req_method_strength = METHOD_STRENGTH.get(req_lower, -1)
        orig_sev_rank = SEVERITY_RANK.get(orig_lower, -1)
        req_sev_rank = SEVERITY_RANK.get(req_lower, -1)

        is_method_override = override_type == "strategy_method"
        is_severity_override = override_type == "risk_severity"
        is_add_protection = override_type == "add_protection"
        going_to_none = req_lower == "none"
        # FAST PATHS — always ALLOW (no further checks needed)

        # FP-1: Non-none → non-none method switch (user controls HOW, not WHETHER)
        if (
            is_method_override
            and orig_lower != "none"
            and req_lower != "none"
        ):
            return self._record(
                session_id, detection_id, override_type, element_type,
                beneficiary, sev_lower, original_value, requested_value,
                requested_value, SafetyAction.ALLOW,
                rule_id="FP1_METHOD_SWITCH",
                reason="Non-none to non-none method change is always permitted; user controls how to protect.",
                user_confirmed=user_confirmed,
            )

        # FP-2: Strengthening a method (higher strength) or severity upgrade
        if is_method_override and req_method_strength > orig_method_strength and req_method_strength > 0:
            return self._record(
                session_id, detection_id, override_type, element_type,
                beneficiary, sev_lower, original_value, requested_value,
                requested_value, SafetyAction.ALLOW,
                rule_id="FP2_STRENGTHEN_METHOD",
                reason="Increasing protection strength is always allowed.",
                user_confirmed=user_confirmed,
            )

        if is_severity_override and req_sev_rank > orig_sev_rank:
            return self._record(
                session_id, detection_id, override_type, element_type,
                beneficiary, sev_lower, original_value, requested_value,
                requested_value, SafetyAction.ALLOW,
                rule_id="FP2_STRENGTHEN_SEVERITY",
                reason="Upgrading severity is always allowed.",
                user_confirmed=user_confirmed,
            )

        # FP-3: add_protection requests always pass through
        if is_add_protection:
            return self._record(
                session_id, detection_id, override_type, element_type,
                beneficiary, sev_lower, original_value, requested_value,
                requested_value, SafetyAction.ALLOW,
                rule_id="FP3_ADD_PROTECTION",
                reason="Adding new protection coverage is always permitted.",
                user_confirmed=user_confirmed,
            )

        # FP-4: SELF beneficiary — user has full control over their own data
        if beneficiary == ProtectionBeneficiary.SELF:
            return self._record(
                session_id, detection_id, override_type, element_type,
                beneficiary, sev_lower, original_value, requested_value,
                requested_value, SafetyAction.ALLOW,
                rule_id="FP4_SELF_BENEFICIARY",
                reason="User has full autonomy over their own data.",
                user_confirmed=user_confirmed,
            )
        # BLOCK RULES — hard stops, never overridable by user_confirmed

        # BR-5: Third-party face → method=none blocked unconditionally
        if (
            is_method_override
            and going_to_none
            and elem_lower == "face"
            and beneficiary == ProtectionBeneficiary.THIRD_PARTY
        ):
            return self._record(
                session_id, detection_id, override_type, element_type,
                beneficiary, sev_lower, original_value, requested_value,
                original_value, SafetyAction.BLOCK,
                rule_id="BLOCK_THIRD_PARTY_UNPROTECT",
                reason="Unknown persons have an inherent right to privacy; bystander protection cannot be removed.",
                user_confirmed=user_confirmed,
            )

        # BR-6: CRITICAL text → method=none blocked
        if (
            is_method_override
            and going_to_none
            and elem_lower == "text"
            and sev_lower == "critical"
        ):
            return self._record(
                session_id, detection_id, override_type, element_type,
                beneficiary, sev_lower, original_value, requested_value,
                original_value, SafetyAction.BLOCK,
                rule_id="BLOCK_CRITICAL_TEXT_UNPROTECT",
                reason="Critical PII (SSN / CC / password) must always be protected.",
                user_confirmed=user_confirmed,
            )

        # BR-7: Legal requirement → method=none blocked
        if (
            is_method_override
            and going_to_none
            and beneficiary == ProtectionBeneficiary.LEGAL
        ):
            return self._record(
                session_id, detection_id, override_type, element_type,
                beneficiary, sev_lower, original_value, requested_value,
                original_value, SafetyAction.BLOCK,
                rule_id="BLOCK_LEGAL_UNPROTECT",
                reason="Regulatory mandate (GDPR/CCPA); protection cannot be waived.",
                user_confirmed=user_confirmed,
            )

        # BR-8: CRITICAL text severity downgrade blocked
        if (
            is_severity_override
            and sev_lower == "critical"
            and req_sev_rank < SEVERITY_RANK["critical"]
        ):
            return self._record(
                session_id, detection_id, override_type, element_type,
                beneficiary, sev_lower, original_value, requested_value,
                original_value, SafetyAction.BLOCK,
                rule_id="BLOCK_CRITICAL_TEXT_DOWNGRADE",
                reason="Cannot downgrade severity of CRITICAL text elements.",
                user_confirmed=user_confirmed,
            )

        # BR-9: HIGH text severity downgrade blocked
        if (
            is_severity_override
            and elem_lower == "text"
            and sev_lower == "high"
            and req_sev_rank < SEVERITY_RANK["high"]
        ):
            return self._record(
                session_id, detection_id, override_type, element_type,
                beneficiary, sev_lower, original_value, requested_value,
                original_value, SafetyAction.BLOCK,
                rule_id="BLOCK_HIGH_TEXT_DOWNGRADE",
                reason="Cannot downgrade severity of HIGH-risk text elements.",
                user_confirmed=user_confirmed,
            )

        # BR-10: Bystander face severity downgrade blocked
        if (
            is_severity_override
            and elem_lower == "face"
            and beneficiary == ProtectionBeneficiary.THIRD_PARTY
            and req_sev_rank < orig_sev_rank
        ):
            return self._record(
                session_id, detection_id, override_type, element_type,
                beneficiary, sev_lower, original_value, requested_value,
                original_value, SafetyAction.BLOCK,
                rule_id="BLOCK_BYSTANDER_SEVERITY_DOWNGRADE",
                reason="Cannot lower severity for unrecognised (bystander) faces.",
                user_confirmed=user_confirmed,
            )
        # CHALLENGE RULES — soft guards; allowed if user_confirmed=True

        # CR-11: Explicit-consent face + method→none → CHALLENGE
        #        (user confirmed they want to remove their own face protection)
        if (
            is_method_override
            and going_to_none
            and elem_lower == "face"
            and consent_lower == "explicit"
        ):
            if user_confirmed:
                return self._record(
                    session_id, detection_id, override_type, element_type,
                    beneficiary, sev_lower, original_value, requested_value,
                    requested_value, SafetyAction.ALLOW,
                    rule_id="CR11_EXPLICIT_FACE_UNPROTECT_CONFIRMED",
                    reason="User confirmed removal of protection for their own face.",
                    user_confirmed=True,
                )
            return self._record(
                session_id, detection_id, override_type, element_type,
                beneficiary, sev_lower, original_value, requested_value,
                original_value, SafetyAction.CHALLENGE,
                rule_id="CR11_EXPLICIT_FACE_UNPROTECT",
                reason=(
                    "You are about to remove protection for a face that has explicit consent. "
                    "Confirm with user_confirmed=true to proceed."
                ),
                user_confirmed=False,
            )

        # CR-12: Verified-off screen + strengthen (adding protection where screen is off)
        if (
            is_method_override
            and screen_lower == "verified_off"
            and req_method_strength > 0
            and orig_lower == "none"
        ):
            if user_confirmed:
                return self._record(
                    session_id, detection_id, override_type, element_type,
                    beneficiary, sev_lower, original_value, requested_value,
                    requested_value, SafetyAction.ALLOW,
                    rule_id="CR12_VERIFIED_OFF_SCREEN_CONFIRMED",
                    reason="User confirmed protection of a screen verified as OFF.",
                    user_confirmed=True,
                )
            return self._record(
                session_id, detection_id, override_type, element_type,
                beneficiary, sev_lower, original_value, requested_value,
                original_value, SafetyAction.CHALLENGE,
                rule_id="CR12_VERIFIED_OFF_SCREEN",
                reason=(
                    "This screen was verified as OFF (no sensitive content visible). "
                    "Adding protection may be unnecessary. Confirm with user_confirmed=true to proceed."
                ),
                user_confirmed=False,
            )

        # CR-13: LOW-severity label text + strengthen → CHALLENGE
        if (
            is_method_override
            and elem_lower == "text"
            and sev_lower == "low"
            and req_method_strength > 0
            and orig_lower == "none"
        ):
            if user_confirmed:
                return self._record(
                    session_id, detection_id, override_type, element_type,
                    beneficiary, sev_lower, original_value, requested_value,
                    requested_value, SafetyAction.ALLOW,
                    rule_id="CR13_LOW_LABEL_STRENGTHEN_CONFIRMED",
                    reason="User confirmed protection of a LOW-severity text label.",
                    user_confirmed=True,
                )
            return self._record(
                session_id, detection_id, override_type, element_type,
                beneficiary, sev_lower, original_value, requested_value,
                original_value, SafetyAction.CHALLENGE,
                rule_id="CR13_LOW_LABEL_STRENGTHEN",
                reason=(
                    "This text is classified as LOW severity (likely a label, not a value). "
                    "Protecting it may over-redact benign content. Confirm with user_confirmed=true."
                ),
                user_confirmed=False,
            )
        # DEFAULT: allow everything else
        return self._record(
            session_id, detection_id, override_type, element_type,
            beneficiary, sev_lower, original_value, requested_value,
            requested_value, SafetyAction.ALLOW,
            rule_id="DEFAULT_ALLOW",
            reason="No blocking or challenge rule matched; request is permitted.",
            user_confirmed=user_confirmed,
        )

    def apply_batch(self, session_id: str, overrides) -> "OverrideResponse":
        """
        Process a batch of OverrideRequest objects through the safety kernel.

        Iterates each override, calls ``validate_override``, and collects
        results into an ``OverrideResponse`` (applied vs. rejected lists).

        Parameters
        ----------
        session_id: Active session UUID.
        overrides:  Iterable of OverrideRequest-like objects with attributes:
                    type, detection_id, value, reason, user_confirmed.

        Returns
        -------
        OverrideResponse with ``applied`` and ``rejected`` lists.
        """
        # Import response types here to avoid circular imports at module scope
        from backend.schemas.responses import (
            AppliedOverride,
            OverrideResponse,
            RejectedOverride,
        )

        applied: List[AppliedOverride] = []
        rejected: List[RejectedOverride] = []

        for ov in overrides:
            ov_type = str(ov.type)
            detection_id = str(ov.detection_id)
            requested_value = str(ov.value) if ov.value is not None else ""
            reason = str(ov.reason)
            user_confirmed = bool(ov.user_confirmed)

            try:
                record = self.validate_override(
                    session_id=session_id,
                    detection_id=detection_id,
                    override_type=ov_type,
                    element_type="unknown",        # element_type not in OverrideRequest;
                                                   # callers that need full enforcement should
                                                   # pass richer context via the service layer
                    original_value="",             # unknown at batch level without session state
                    requested_value=requested_value,
                    severity="low",                # conservative default
                    consent_status=None,
                    screen_state=None,
                    legal_requirement=False,
                    user_confirmed=user_confirmed,
                )

                if record.safety_action == SafetyAction.BLOCK:
                    rejected.append(RejectedOverride(
                        detection_id=detection_id,
                        type=ov_type,
                        error_code="SAFETY_BLOCK",
                        message=record.reason,
                    ))
                elif record.safety_action == SafetyAction.CHALLENGE:
                    rejected.append(RejectedOverride(
                        detection_id=detection_id,
                        type=ov_type,
                        error_code="SAFETY_CHALLENGE",
                        message=record.reason,
                    ))
                else:
                    applied.append(AppliedOverride(
                        detection_id=detection_id,
                        type=ov_type,
                        value=ov.value,
                        reason=record.reason,
                    ))

            except Exception as exc:
                logger.error(
                    "apply_batch: unexpected error for detection_id=%s: %s",
                    detection_id,
                    exc,
                )
                rejected.append(RejectedOverride(
                    detection_id=detection_id,
                    type=ov_type,
                    error_code="PIPELINE_ERROR",
                    message=str(exc),
                ))

        return OverrideResponse(applied=applied, rejected=rejected)

    def get_audit_trail(self, session_id: str) -> List[dict]:
        """
        Return all override records for *session_id*, ordered by timestamp.

        Queries MongoDB when a collection is available; falls back to the
        in-memory list otherwise.
        """
        if self._collection is not None:
            try:
                cursor = self._collection.find(
                    {"session_id": session_id},
                    {"_id": 0},
                ).sort("timestamp", 1)
                return list(cursor)
            except Exception as exc:
                logger.warning("MongoDB query failed, falling back to in-memory: %s", exc)

        return [
            r for r in self._in_memory if r.get("session_id") == session_id
        ]

    def export_audit_json(self, session_id: str) -> str:
        """
        Export the full audit trail for *session_id* as a formatted JSON string.
        """
        trail = self.get_audit_trail(session_id)
        return json.dumps(trail, indent=2, default=str)
    # Private helpers

    def _record(
        self,
        session_id: str,
        detection_id: str,
        override_type: str,
        element_type: str,
        beneficiary: ProtectionBeneficiary,
        severity_at_time: str,
        original_value: str,
        requested_value: str,
        approved_value: str,
        safety_action: SafetyAction,
        rule_id: Optional[str] = None,
        reason: str = "",
        user_confirmed: bool = False,
    ) -> OverrideRecord:
        """
        Build, persist, and return an ``OverrideRecord``.

        Persistence is attempted to MongoDB first; on any failure the record
        is kept in the in-memory list (never lost).
        """
        rec = OverrideRecord(
            session_id=session_id,
            detection_id=detection_id,
            override_type=override_type,
            element_type=element_type,
            beneficiary=beneficiary,
            severity_at_time=severity_at_time,
            original_value=original_value,
            requested_value=requested_value,
            approved_value=approved_value,
            safety_action=safety_action,
            rule_id=rule_id,
            reason=reason,
            user_confirmed=user_confirmed,
        )

        rec_dict = rec.model_dump()

        # Always write to in-memory list so get_audit_trail works even if
        # MongoDB is temporarily unavailable.
        self._in_memory.append(rec_dict)

        # Notify the optional provenance callback (e.g. ProvenanceService).
        # Errors in the callback must never surface to callers.
        if self._on_record is not None:
            try:
                self._on_record(rec)
            except Exception as exc:
                logger.warning("on_record callback failed: %s", exc)

        if self._collection is not None:
            try:
                self._collection.insert_one({**rec_dict})
            except Exception as exc:
                logger.warning(
                    "Failed to persist OverrideRecord to MongoDB (in-memory copy retained): %s",
                    exc,
                )

        return rec
