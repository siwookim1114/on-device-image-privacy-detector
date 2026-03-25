"""
profile.py — FastAPI router for the Privacy Profile API.

Endpoints:
  GET    /api/v1/profile                — Retrieve current user's profile
  POST   /api/v1/profile                — Create profile (onboarding completion)
  PUT    /api/v1/profile                — Partial update of existing profile
  DELETE /api/v1/profile                — Delete profile + cascade face DB entries
  GET    /api/v1/profile/questionnaire  — Serve static questionnaire structure
  POST   /api/v1/profile/enroll-face    — Enroll a face photo for the current user

All write endpoints enforce BLOCK-guarded fields (SSN, CC, passwords must stay True).
Face enrollment delegates to ConsentService when available; falls back gracefully
if the consent service has not been initialized.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from backend.middleware.auth_middleware import require_auth
from backend.schemas.profile_schemas import (
    ContactEntryResponse,
    CreateProfileRequest,
    FaceSensitivityResponse,
    ObjectSensitivityResponse,
    ProfileCreatedResponse,
    ProfileDeletedResponse,
    ProfileResponse,
    ProfileUpdatedResponse,
    QuestionnaireResponse,
    ScreenSensitivityResponse,
    TextSensitivityResponse,
    UpdateProfileRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/profile", tags=["profile"])

# Absolute path to the static questionnaire YAML
_QUESTIONNAIRE_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "questionnaire.yaml"

# Allowed MIME types for face photo uploads
_ALLOWED_IMAGE_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)

# Maximum face photo upload size: 10 MB
_MAX_FACE_PHOTO_BYTES = 10 * 1024 * 1024
# Dependency helpers

def _get_profile_service(request: Request):
    """Extract ProfileService from app.state; raise 503 if not initialised."""
    svc = getattr(request.app.state, "profile_service", None)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "code": "PIPELINE_ERROR",
                    "message": "ProfileService is not available yet.",
                    "details": {},
                }
            },
        )
    return svc


def _user_id_from_session(session: object) -> str:
    """Extract user_id (== session_id) from the session object returned by require_auth."""
    # SessionRecord stores user identity as session_id (UUID token owner).
    user_id = getattr(session, "session_id", None)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": "Could not determine user identity from session.",
                    "details": {},
                }
            },
        )
    return user_id


def _profile_doc_to_response(doc: Dict[str, Any]) -> ProfileResponse:
    """Convert a MongoDB document dict to a ProfileResponse model."""

    def _dt_str(val: Any) -> str:
        """Return ISO-8601 string for datetime or passthrough if already a str."""
        if val is None:
            return ""
        if hasattr(val, "isoformat"):
            return val.isoformat()
        return str(val)

    face_raw = doc.get("face_settings", {}) or {}
    text_raw = doc.get("text_settings", {}) or {}
    screen_raw = doc.get("screen_settings", {}) or {}
    obj_raw = doc.get("object_settings", {}) or {}

    contacts: List[ContactEntryResponse] = [
        ContactEntryResponse(
            person_id=c.get("person_id", ""),
            display_name=c.get("display_name", ""),
            relationship=c.get("relationship", "friend"),
            consent_level=c.get("consent_level", "assumed"),
        )
        for c in (doc.get("known_contacts") or [])
    ]

    return ProfileResponse(
        user_id=doc.get("user_id", ""),
        profile_version=int(doc.get("profile_version", 1)),
        onboarding_complete=bool(doc.get("onboarding_complete", False)),
        display_name=doc.get("display_name"),
        self_person_id=doc.get("self_person_id"),
        face_enrollment_count=int(doc.get("face_enrollment_count", 0)),
        known_contacts=contacts,
        face_settings=FaceSensitivityResponse(
            bystander_sensitivity=face_raw.get("bystander_sensitivity", "critical"),
            known_contact_sensitivity=face_raw.get("known_contact_sensitivity", "high"),
            self_sensitivity=face_raw.get("self_sensitivity", "medium"),
            min_face_size_px=int(face_raw.get("min_face_size_px", 30)),
        ),
        text_settings=TextSensitivityResponse(
            protect_ssn=bool(text_raw.get("protect_ssn", True)),
            protect_credit_card=bool(text_raw.get("protect_credit_card", True)),
            protect_passwords=bool(text_raw.get("protect_passwords", True)),
            protect_phone_numbers=bool(text_raw.get("protect_phone_numbers", True)),
            protect_email_addresses=bool(text_raw.get("protect_email_addresses", True)),
            protect_addresses=bool(text_raw.get("protect_addresses", False)),
            protect_names=bool(text_raw.get("protect_names", False)),
            protect_numeric_fragments=bool(text_raw.get("protect_numeric_fragments", True)),
            protect_generic_text=bool(text_raw.get("protect_generic_text", False)),
        ),
        screen_settings=ScreenSensitivityResponse(
            protect_screens_when_on=bool(screen_raw.get("protect_screens_when_on", True)),
            protect_screens_when_off=bool(screen_raw.get("protect_screens_when_off", False)),
            own_devices_unprotected=bool(screen_raw.get("own_devices_unprotected", True)),
        ),
        object_settings=ObjectSensitivityResponse(
            protect_license_plates=bool(obj_raw.get("protect_license_plates", True)),
            protect_personal_documents=bool(obj_raw.get("protect_personal_documents", True)),
            protect_other_objects=bool(obj_raw.get("protect_other_objects", False)),
        ),
        preferred_face_method=doc.get("preferred_face_method", "blur"),
        preferred_text_method=doc.get("preferred_text_method", "solid_overlay"),
        preferred_screen_method=doc.get("preferred_screen_method", "blur"),
        preferred_object_method=doc.get("preferred_object_method", "blur"),
        default_mode=doc.get("default_mode", "hybrid"),
        ethical_mode=doc.get("ethical_mode", "balanced"),
        auto_advance_threshold=doc.get("auto_advance_threshold", "medium"),
        pause_on_critical=bool(doc.get("pause_on_critical", True)),
        pause_on_new_faces=bool(doc.get("pause_on_new_faces", True)),
        require_confirm_on_bystander_unprotect=bool(
            doc.get("require_confirm_on_bystander_unprotect", True)
        ),
        threshold_overrides=doc.get("threshold_overrides") or {},
        created_at=_dt_str(doc.get("created_at")),
        updated_at=_dt_str(doc.get("updated_at")),
    )
# GET /profile

@router.get(
    "",
    response_model=ProfileResponse,
    summary="Get current user's privacy profile.",
    responses={
        200: {"description": "Profile found."},
        404: {"description": "No profile exists yet for this user."},
    },
)
async def get_profile(
    request: Request,
    session=Depends(require_auth),
) -> ProfileResponse:
    """
    Return the privacy profile for the authenticated user.

    If no profile exists yet (user has not completed onboarding), returns HTTP 404
    so the frontend knows to redirect to the questionnaire flow.
    """
    user_id = _user_id_from_session(session)
    svc = _get_profile_service(request)

    doc = await svc.get_profile(user_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "PROFILE_NOT_FOUND",
                    "message": f"No privacy profile found for this session. "
                               f"Complete onboarding at POST /api/v1/profile.",
                    "details": {},
                }
            },
        )

    return _profile_doc_to_response(doc)
# POST /profile

@router.post(
    "",
    response_model=ProfileCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create privacy profile (onboarding completion).",
    responses={
        201: {"description": "Profile created."},
        409: {"description": "Profile already exists for this user."},
        422: {"description": "Validation error (locked fields or invalid values)."},
    },
)
async def create_profile(
    data: CreateProfileRequest,
    request: Request,
    session=Depends(require_auth),
) -> ProfileCreatedResponse:
    """
    Persist a new privacy profile for the authenticated user.

    Called at the final step of the onboarding questionnaire. Returns HTTP 409
    if a profile already exists — use PUT to update.

    BLOCK-guarded: ``text_settings.protect_ssn``, ``protect_credit_card``, and
    ``protect_passwords`` cannot be set to ``False``.
    """
    user_id = _user_id_from_session(session)
    svc = _get_profile_service(request)

    try:
        await svc.create_profile(user_id, data.model_dump(exclude_none=True))
    except ValueError as exc:
        msg = str(exc)
        http_status = (
            status.HTTP_409_CONFLICT
            if "already exists" in msg
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        raise HTTPException(
            status_code=http_status,
            detail={
                "error": {
                    "code": "PROFILE_CONFLICT" if "already exists" in msg else "VALIDATION_ERROR",
                    "message": msg,
                    "details": {},
                }
            },
        )

    return ProfileCreatedResponse(
        user_id=user_id,
        onboarding_complete=data.onboarding_complete,
    )
# PUT /profile

@router.put(
    "",
    response_model=ProfileUpdatedResponse,
    summary="Partial update of the privacy profile.",
    responses={
        200: {"description": "Profile updated."},
        422: {"description": "Validation error (locked fields or invalid values)."},
    },
)
async def update_profile(
    data: UpdateProfileRequest,
    request: Request,
    session=Depends(require_auth),
) -> ProfileUpdatedResponse:
    """
    Apply a partial update to the privacy profile.

    Only provided (non-null) fields are written; omitted fields are left
    unchanged. Uses upsert semantics — creates the profile if it does not yet
    exist (e.g. guest updating before completing the questionnaire).
    """
    user_id = _user_id_from_session(session)
    svc = _get_profile_service(request)

    payload = data.model_dump(exclude_none=True)

    try:
        updated_doc = await svc.update_profile(user_id, payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": str(exc),
                    "details": {},
                }
            },
        )

    # Derive the list of top-level fields that were touched
    updated_fields: List[str] = []
    for key in payload:
        updated_fields.append(key)

    return ProfileUpdatedResponse(
        user_id=user_id,
        updated_fields=updated_fields,
    )
# DELETE /profile

@router.delete(
    "",
    response_model=ProfileDeletedResponse,
    summary="Delete profile and cascade face database entries.",
    responses={
        200: {"description": "Profile deleted."},
        404: {"description": "No profile found for this user."},
    },
)
async def delete_profile(
    request: Request,
    session=Depends(require_auth),
) -> ProfileDeletedResponse:
    """
    Delete the privacy profile and remove all linked face embeddings from the
    FaceDatabase (self + all known contacts).
    """
    user_id = _user_id_from_session(session)
    svc = _get_profile_service(request)

    deleted = await svc.delete_profile(user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "PROFILE_NOT_FOUND",
                    "message": "No profile found to delete for this session.",
                    "details": {},
                }
            },
        )

    return ProfileDeletedResponse(user_id=user_id)
# GET /profile/questionnaire

@router.get(
    "/questionnaire",
    response_model=QuestionnaireResponse,
    summary="Get static questionnaire structure for frontend rendering.",
    responses={
        200: {"description": "Questionnaire structure returned."},
        500: {"description": "Questionnaire YAML could not be loaded."},
    },
)
async def get_questionnaire() -> QuestionnaireResponse:
    """
    Return the static questionnaire definition from ``configs/questionnaire.yaml``.

    The frontend uses this to render the onboarding flow dynamically without
    hard-coding field names or step order. No authentication required.
    """
    try:
        raw = _QUESTIONNAIRE_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        return QuestionnaireResponse(**data)
    except FileNotFoundError:
        logger.error("Questionnaire YAML not found at %s", _QUESTIONNAIRE_PATH)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "PIPELINE_ERROR",
                    "message": "Questionnaire configuration file not found.",
                    "details": {"path": str(_QUESTIONNAIRE_PATH)},
                }
            },
        )
    except Exception as exc:
        logger.exception("Failed to load questionnaire YAML")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "PIPELINE_ERROR",
                    "message": f"Could not parse questionnaire: {exc}",
                    "details": {},
                }
            },
        )
# POST /profile/enroll-face

@router.post(
    "/enroll-face",
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a face photo for the current user.",
    responses={
        201: {"description": "Face enrolled successfully."},
        400: {"description": "No face detected in the uploaded image."},
        413: {"description": "Image file too large (max 10 MB)."},
        422: {"description": "Unsupported file type."},
        503: {"description": "ConsentService or FaceDatabase not available."},
    },
)
async def enroll_face(
    file: UploadFile,
    request: Request,
    session=Depends(require_auth),
) -> Dict[str, Any]:
    """
    Enroll a single face photo for the authenticated user.

    Workflow:
      1. Validate MIME type and file size.
      2. Read image bytes.
      3. Delegate to ConsentService.enroll_face() which runs MTCNN + FaceNet
         and writes the embedding to FaceDatabase.
      4. Update ``self_person_id`` and ``face_enrollment_count`` on the profile.

    Returns the ``person_id`` and updated ``face_enrollment_count`` on success.
    Returns HTTP 400 if MTCNN finds no face in the uploaded image.
    """
    user_id = _user_id_from_session(session)

    # -- MIME type validation --------------------------------------------------
    content_type = file.content_type or ""
    if content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": (
                        f"Unsupported file type '{content_type}'. "
                        f"Accepted: {sorted(_ALLOWED_IMAGE_TYPES)}"
                    ),
                    "details": {},
                }
            },
        )

    # -- File size validation --------------------------------------------------
    image_bytes = await file.read()
    if len(image_bytes) > _MAX_FACE_PHOTO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": f"Image too large ({len(image_bytes)} bytes). Maximum: {_MAX_FACE_PHOTO_BYTES} bytes.",
                    "details": {},
                }
            },
        )

    # -- Retrieve ConsentService -----------------------------------------------
    consent_svc = getattr(request.app.state, "consent_service", None)
    if consent_svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "code": "PIPELINE_ERROR",
                    "message": "ConsentService is not available. Ensure MongoDB is running.",
                    "details": {},
                }
            },
        )

    # -- Delegate enrollment ---------------------------------------------------
    try:
        result = await consent_svc.enroll_face(
            user_id=user_id,
            image_bytes=image_bytes,
            label=f"self_{user_id[:8]}",
            relationship="self",
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "no face" in msg or "no faces" in msg or "not detected" in msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "No face detected in the uploaded image. Please use a clear, well-lit portrait.",
                        "details": {"original_error": str(exc)},
                    }
                },
            )
        logger.exception("Unexpected error during face enrollment for user_id=%s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "PIPELINE_ERROR",
                    "message": f"Face enrollment failed: {exc}",
                    "details": {},
                }
            },
        )

    # -- Update profile with self_person_id and enrollment count ---------------
    svc = _get_profile_service(request)
    person_id: str = result.get("person_id", "")
    enrollment_count: int = result.get("embedding_count", 1)

    try:
        await svc.update_profile(
            user_id,
            {
                "self_person_id": person_id,
                "face_enrollment_count": enrollment_count,
            },
        )
    except Exception as exc:
        # Enrollment succeeded; profile update failure is non-fatal but should be surfaced
        logger.error(
            "ProfileService: failed to update face_enrollment_count for user_id=%s: %s",
            user_id,
            exc,
        )

    return {
        "person_id": person_id,
        "face_enrollment_count": enrollment_count,
        "message": "Face enrolled successfully.",
    }
