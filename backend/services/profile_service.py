"""
profile_service.py — Business logic for Privacy Profile CRUD.

Responsibilities:
  - Create, read, update, and delete PrivacyProfile documents in MongoDB.
  - Enforce BLOCK-guarded fields (SSN, credit-card, password protection MUST
    remain True; disabling them is a regulatory violation).
  - On delete, cascade the removal to all FaceDatabase person entries owned
    by this user (self_person_id + known_contacts[*].person_id).
  - Provide get_default_profile() for guest / pre-onboarding mode.

MongoDB collection: privacy_profiles
Primary key: user_id (str UUID, indexed unique)

All public methods are async (non-blocking interface contract), but the
underlying pymongo calls are synchronous — acceptable for single-machine,
low-concurrency deployment.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
# Project-root path injection — same pattern as consent_service.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.models import (
    FaceSensitivitySettings,
    ObjectSensitivitySettings,
    PrivacyProfile,
    ScreenSensitivitySettings,
    TextSensitivitySettings,
)

logger = logging.getLogger(__name__)
# Constants

# Fields in text_settings that MUST remain True — regulatory / safety requirement.
LOCKED_TEXT_FIELDS: tuple[str, ...] = (
    "protect_ssn",
    "protect_credit_card",
    "protect_passwords",
)

COLLECTION_NAME = "privacy_profiles"
# ProfileService


class ProfileService:
    """
    Service layer for Privacy Profile management.

    ``db`` is a PyMongo database instance (or None for unit testing without
    MongoDB). When ``db`` is None, all write operations raise RuntimeError
    with a clear message.
    """

    def __init__(self, db: Optional[Any] = None) -> None:
        self._db = db
        self._collection = db[COLLECTION_NAME] if db is not None else None
    # Public API

    async def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the privacy profile for *user_id*.

        Returns the profile as a plain dict (MongoDB document with ``_id``
        stripped), or None if no profile exists yet.
        """
        if self._collection is None:
            logger.warning("ProfileService: MongoDB not connected — get_profile returning None")
            return None

        doc = self._collection.find_one({"user_id": user_id}, {"_id": 0})
        return doc  # None if not found

    async def create_profile(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new privacy profile for *user_id*.

        Raises:
            RuntimeError: If MongoDB is not connected.
            ValueError: If *user_id* already has a profile (HTTP 409 semantics).
            ValueError: If locked text fields are set to False.
        """
        self._require_db("create_profile")

        # Conflict check
        existing = self._collection.find_one({"user_id": user_id}, {"_id": 0})
        if existing is not None:
            raise ValueError(f"Profile already exists for user_id='{user_id}'")

        # Locked field validation
        self._validate_locked_fields(data)

        # Build the full profile from defaults + caller data
        profile = self._build_profile(user_id, data)

        # Persist
        self._collection.insert_one({**profile, "user_id": user_id})
        logger.info("ProfileService: created profile for user_id=%s", user_id)
        return profile

    async def update_profile(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Partial update of an existing privacy profile.

        Uses upsert semantics: if no profile exists yet (e.g. guest updating
        before completing onboarding) a new document is created.

        Raises:
            RuntimeError: If MongoDB is not connected.
            ValueError: If locked text fields are set to False.
        """
        self._require_db("update_profile")

        # Locked field validation applies to partial updates too
        self._validate_locked_fields(data)

        now = datetime.now(tz=timezone.utc).isoformat()

        # Flatten nested sub-model dicts into MongoDB dotted-key updates
        update_doc: Dict[str, Any] = {"updated_at": now}
        updated_fields: List[str] = ["updated_at"]

        # Scalar top-level fields
        _SCALAR_FIELDS = {
            "display_name", "self_person_id", "face_enrollment_count",
            "preferred_face_method", "preferred_text_method",
            "preferred_screen_method", "preferred_object_method",
            "default_mode", "ethical_mode",
            "auto_advance_threshold", "pause_on_critical", "pause_on_new_faces",
            "require_confirm_on_bystander_unprotect",
            "threshold_overrides", "onboarding_complete",
            "known_contacts",
        }

        for field in _SCALAR_FIELDS:
            if field in data and data[field] is not None:
                update_doc[field] = data[field]
                updated_fields.append(field)

        # Nested settings — merge at the sub-document level
        for sub_key in ("face_settings", "text_settings", "screen_settings", "object_settings"):
            if sub_key in data and data[sub_key]:
                sub_data = data[sub_key]
                # Only write non-None values from the sub-dict
                for k, v in sub_data.items():
                    if v is not None:
                        update_doc[f"{sub_key}.{k}"] = v
                updated_fields.append(sub_key)

        result = self._collection.update_one(
            {"user_id": user_id},
            {"$set": update_doc},
            upsert=True,
        )

        logger.info(
            "ProfileService: updated profile for user_id=%s (matched=%d, modified=%d, upserted=%s)",
            user_id,
            result.matched_count,
            result.modified_count,
            result.upserted_id,
        )

        updated_doc = self._collection.find_one({"user_id": user_id}, {"_id": 0})
        return updated_doc or {}

    async def delete_profile(self, user_id: str) -> bool:
        """
        Delete the profile for *user_id* and cascade face database entries.

        Returns True if a document was deleted, False if none existed.

        Cascade: Attempts to remove person entries from the face database
        (FaceDatabase) for self_person_id and all known_contacts[*].person_id.
        Cascade errors are logged but do NOT abort the profile deletion.
        """
        self._require_db("delete_profile")

        # Load existing doc to collect person_ids for cascade
        doc = self._collection.find_one({"user_id": user_id}, {"_id": 0})
        if doc is None:
            return False

        # Delete the profile document
        result = self._collection.delete_one({"user_id": user_id})
        deleted = result.deleted_count > 0

        if deleted:
            logger.info("ProfileService: deleted profile for user_id=%s", user_id)
            # Cascade to face database
            self._cascade_delete_faces(doc)

        return deleted

    def get_default_profile(self) -> Dict[str, Any]:
        """
        Return a default PrivacyProfile as a plain dict.

        Used for guest mode or pre-onboarding pipeline runs — the pipeline
        falls back to this when no persisted profile is found for the session.
        """
        profile = PrivacyProfile()
        return profile.model_dump()
    # Validation helpers

    def _validate_locked_fields(self, data: Dict[str, Any]) -> None:
        """
        Raise ValueError if any BLOCK-guarded text field is set to False.

        Protected by SafetyKernel rules BR-5 and BR-6. The UI renders these
        as non-interactive locked checkboxes, but the service enforces this
        server-side so no API call can bypass the constraint.
        """
        text = data.get("text_settings", {}) or {}
        # Handle both nested dict and Pydantic-model-like objects
        if hasattr(text, "model_dump"):
            text = text.model_dump(exclude_none=True)

        for field in LOCKED_TEXT_FIELDS:
            value = text.get(field)
            if value is False:
                raise ValueError(
                    f"Cannot disable '{field}' — this field is a regulatory "
                    f"requirement and cannot be set to False. "
                    f"(SafetyKernel BLOCK rules BR-5/BR-6)"
                )

    def _validate_method_ethical_mode(
        self, data: Dict[str, Any], ethical_mode: str
    ) -> None:
        """
        Raise ValueError if a preferred method conflicts with the ethical mode.

        strict mode: only blur, pixelate, solid_overlay are permitted.
        balanced / creative: all methods permitted (creative outputs are watermarked
        at pipeline time).
        """
        if ethical_mode == "strict":
            strict_allowed = {"blur", "pixelate", "solid_overlay", "none"}
            method_fields = [
                "preferred_face_method",
                "preferred_text_method",
                "preferred_screen_method",
                "preferred_object_method",
            ]
            for mf in method_fields:
                method = data.get(mf)
                if method and method not in strict_allowed:
                    raise ValueError(
                        f"'{method}' is not allowed in strict ethical_mode. "
                        f"Permitted methods: {sorted(strict_allowed)}"
                    )
    # Internal helpers

    def _require_db(self, operation: str) -> None:
        if self._collection is None:
            raise RuntimeError(
                f"ProfileService.{operation}: MongoDB is not connected. "
                "Ensure the database is initialised before calling this method."
            )

    def _build_profile(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge caller-supplied *data* onto a default PrivacyProfile.

        Returns a plain dict ready for MongoDB insertion.
        """
        # Start from defaults
        profile = PrivacyProfile(user_id=user_id)
        profile_dict = profile.model_dump()

        now = datetime.now(tz=timezone.utc).isoformat()
        profile_dict["created_at"] = now
        profile_dict["updated_at"] = now

        # Apply top-level scalar overrides
        _SCALAR_FIELDS = {
            "display_name", "self_person_id", "face_enrollment_count",
            "preferred_face_method", "preferred_text_method",
            "preferred_screen_method", "preferred_object_method",
            "default_mode", "ethical_mode",
            "auto_advance_threshold", "pause_on_critical", "pause_on_new_faces",
            "require_confirm_on_bystander_unprotect",
            "threshold_overrides", "onboarding_complete", "known_contacts",
        }
        for field in _SCALAR_FIELDS:
            if field in data and data[field] is not None:
                profile_dict[field] = data[field]

        # Apply nested settings
        for sub_key, sub_cls in (
            ("face_settings", FaceSensitivitySettings),
            ("text_settings", TextSensitivitySettings),
            ("screen_settings", ScreenSensitivitySettings),
            ("object_settings", ObjectSensitivitySettings),
        ):
            if sub_key in data and data[sub_key]:
                sub_data = data[sub_key]
                if hasattr(sub_data, "model_dump"):
                    sub_data = sub_data.model_dump(exclude_none=True)
                # Merge over the default sub-dict
                profile_dict[sub_key].update(
                    {k: v for k, v in sub_data.items() if v is not None}
                )

        return profile_dict

    def _cascade_delete_faces(self, profile_doc: Dict[str, Any]) -> None:
        """
        Attempt to remove FaceDatabase entries for persons linked to this profile.

        This is a best-effort operation — errors are logged but not re-raised
        so the main profile deletion always succeeds.
        """
        person_ids: List[str] = []

        self_id = profile_doc.get("self_person_id")
        if self_id:
            person_ids.append(self_id)

        for contact in profile_doc.get("known_contacts", []):
            pid = contact.get("person_id") if isinstance(contact, dict) else getattr(contact, "person_id", None)
            if pid:
                person_ids.append(pid)

        if not person_ids:
            return

        try:
            from utils.storage import FaceDatabase  # type: ignore[import]

            face_db = FaceDatabase()
            for pid in person_ids:
                try:
                    face_db.delete_person(pid)
                    logger.info("ProfileService: cascade-deleted person_id=%s from FaceDatabase", pid)
                except Exception as exc:
                    logger.warning(
                        "ProfileService: could not delete person_id=%s from FaceDatabase: %s",
                        pid,
                        exc,
                    )
        except ImportError:
            logger.warning(
                "ProfileService: FaceDatabase not importable — skipping face cascade for %s",
                person_ids,
            )
        except Exception as exc:
            logger.error("ProfileService: unexpected error during face cascade: %s", exc)
