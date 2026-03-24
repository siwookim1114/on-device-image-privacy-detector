"""
consent.py — FastAPI router for consent / face-identity management.

Endpoints:
  GET    /consent/persons               — List all registered persons
  POST   /consent/persons               — Register a new person (multipart)
  PUT    /consent/persons/{person_id}   — Update person metadata
  DELETE /consent/persons/{person_id}   — Delete person and embeddings
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from backend.middleware.auth_middleware import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/consent", tags=["consent"])

# Allowed MIME types for face photo uploads
_ALLOWED_IMAGE_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)

# Maximum upload size: 10 MB
_MAX_PHOTO_BYTES = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _get_consent_service(request: Request):
    """Extract ConsentService from app.state; raise 503 if not initialised."""
    svc = getattr(request.app.state, "consent_service", None)
    if svc is None:
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
    return svc


# ---------------------------------------------------------------------------
# GET /consent/persons
# ---------------------------------------------------------------------------

@router.get(
    "/persons",
    summary="List all registered persons in the consent database.",
    responses={
        200: {"description": "Person list returned."},
        503: {"description": "Consent service unavailable."},
    },
)
async def list_persons(
    request: Request,
    session=Depends(require_auth),
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Return all persons in the face consent database (embeddings stripped).

    Supports basic pagination via ``limit`` and ``offset`` query parameters.
    """
    svc = _get_consent_service(request)
    result = await svc.get_persons(limit=limit, offset=offset)

    if "error" in result and result.get("total", 0) == 0:
        # Non-fatal: still return the envelope so the frontend can render
        # an empty state rather than crashing on a 5xx.
        logger.warning("get_persons returned error: %s", result["error"])

    return result


# ---------------------------------------------------------------------------
# POST /consent/persons
# ---------------------------------------------------------------------------

@router.post(
    "/persons",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new person (multipart: images + JSON metadata).",
    responses={
        201: {"description": "Person registered successfully."},
        400: {"description": "No face detected in any of the provided images."},
        422: {"description": "Validation error (unsupported file type or bad JSON)."},
        503: {"description": "Consent service unavailable."},
    },
)
async def register_person(
    request: Request,
    session=Depends(require_auth),
    images: List[UploadFile] = File(..., description="One or more face photos."),
    data: Optional[str] = Form(
        default=None,
        description=(
            "JSON string with fields: label (str), relationship (str), "
            "consent_status (str), notes (str, optional)."
        ),
    ),
) -> Dict[str, Any]:
    """
    Register a new person in the consent database.

    Accepts one or more face photos (JPEG/PNG/WebP) and a JSON metadata
    blob.  MTCNN + FaceNet extract 512-D embeddings from each image; images
    where no face is detected are silently skipped (a warning is logged).

    Returns the redacted person document on success.
    """
    svc = _get_consent_service(request)

    # -- Parse metadata JSON ---------------------------------------------------
    meta: Dict[str, Any] = {}
    if data:
        try:
            meta = json.loads(data)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": f"Invalid JSON in 'data' field: {exc}",
                        "details": {},
                    }
                },
            ) from exc

    label: str = meta.get("label", "unknown")
    relationship: str = meta.get("relationship", "unknown")
    consent_status: str = meta.get("consent_status", "assumed")
    notes: Optional[str] = meta.get("notes")

    tmp_paths: List[str] = []
    try:
        for upload in images:
            content_type = upload.content_type or ""
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

            img_bytes = await upload.read()
            if len(img_bytes) > _MAX_PHOTO_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail={
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": (
                                f"Image '{upload.filename}' is too large "
                                f"({len(img_bytes)} bytes). Maximum: {_MAX_PHOTO_BYTES} bytes."
                            ),
                            "details": {},
                        }
                    },
                )

            # Write to a temp file so _extract_embeddings_from_paths can open it
            suffix = os.path.splitext(upload.filename or ".jpg")[1] or ".jpg"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as fh:
                fh.write(img_bytes)
            tmp_paths.append(tmp_path)

        # -- Delegate to ConsentService ----------------------------------------
        result = await svc.register_person(
            label=label,
            relationship=relationship,
            consent_status=consent_status,
            image_paths=tmp_paths,
            notes=notes,
        )

    finally:
        # Always remove temp files regardless of outcome
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    if "error" in result:
        error_msg = result["error"]
        if "no face" in error_msg.lower() or "not detected" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": error_msg,
                        "details": {},
                    }
                },
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "PIPELINE_ERROR",
                    "message": error_msg,
                    "details": {},
                }
            },
        )

    return result


# ---------------------------------------------------------------------------
# PUT /consent/persons/{person_id}
# ---------------------------------------------------------------------------

@router.put(
    "/persons/{person_id}",
    summary="Update metadata for an existing person.",
    responses={
        200: {"description": "Person updated."},
        404: {"description": "Person not found."},
        503: {"description": "Consent service unavailable."},
    },
)
async def update_person(
    person_id: str,
    request: Request,
    session=Depends(require_auth),
    updates: Dict[str, Any] = None,  # type: ignore[assignment]
) -> Dict[str, Any]:
    """
    Update label, relationship, consent_status, or notes for a person.

    Embedding changes require re-registration (POST /consent/persons).

    Accepts JSON body with any subset of: ``label``, ``relationship``,
    ``consent_status``, ``notes``.
    """
    svc = _get_consent_service(request)

    # Read JSON body manually to support partial updates cleanly
    try:
        body_bytes = await request.body()
        updates = json.loads(body_bytes) if body_bytes else {}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": f"Invalid JSON body: {exc}",
                    "details": {},
                }
            },
        ) from exc

    result = await svc.update_person(person_id=person_id, updates=updates)

    if "error" in result:
        error_msg = result["error"]
        http_status = (
            status.HTTP_404_NOT_FOUND
            if "not found" in error_msg.lower()
            else status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        raise HTTPException(
            status_code=http_status,
            detail={
                "error": {
                    "code": "PERSON_NOT_FOUND" if http_status == 404 else "PIPELINE_ERROR",
                    "message": error_msg,
                    "details": {"person_id": person_id},
                }
            },
        )

    return result


# ---------------------------------------------------------------------------
# DELETE /consent/persons/{person_id}
# ---------------------------------------------------------------------------

@router.delete(
    "/persons/{person_id}",
    summary="Delete a person and all associated face embeddings.",
    responses={
        200: {"description": "Person deleted."},
        404: {"description": "Person not found."},
        503: {"description": "Consent service unavailable."},
    },
)
async def delete_person(
    person_id: str,
    request: Request,
    session=Depends(require_auth),
) -> Dict[str, Any]:
    """
    Remove a person and all their face embeddings from the database.

    Returns HTTP 404 if no person with the given ``person_id`` exists.
    """
    svc = _get_consent_service(request)
    deleted = await svc.delete_person(person_id=person_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "PERSON_NOT_FOUND",
                    "message": f"No person found with id '{person_id}'.",
                    "details": {"person_id": person_id},
                }
            },
        )

    return {"person_id": person_id, "deleted": True}
