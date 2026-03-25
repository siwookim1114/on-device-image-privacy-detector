from __future__ import annotations

import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse

from backend.middleware.auth_middleware import require_auth

router = APIRouter(prefix="/pipeline", tags=["images"])

# Minimum pipeline status required before each image type is available
_MIME: dict[str, str] = {
    "original": "image/jpeg",
    "protected": "image/jpeg",
    "risk-map": "image/png",
}
# GET /pipeline/{session_id}/image/original

@router.get(
    "/{session_id}/image/original",
    response_class=FileResponse,
    summary="Serve the original uploaded image.",
)
async def serve_original(
    session_id: str,
    request: Request,
    session: Annotated[object, Depends(require_auth)],
) -> FileResponse:
    _assert_session_ownership(session, session_id)
    return _serve_image(request, session_id, "original")
# GET /pipeline/{session_id}/image/protected

@router.get(
    "/{session_id}/image/protected",
    response_class=FileResponse,
    summary="Serve the protected (obfuscated) output image.",
)
async def serve_protected(
    session_id: str,
    request: Request,
    session: Annotated[object, Depends(require_auth)],
) -> FileResponse:
    _assert_session_ownership(session, session_id)
    return _serve_image(request, session_id, "protected")
# GET /pipeline/{session_id}/image/risk-map

@router.get(
    "/{session_id}/image/risk-map",
    response_class=FileResponse,
    summary="Serve the risk map visualisation.",
)
async def serve_risk_map(
    session_id: str,
    request: Request,
    session: Annotated[object, Depends(require_auth)],
) -> FileResponse:
    _assert_session_ownership(session, session_id)
    return _serve_image(request, session_id, "risk-map")
# Shared image resolution helper

def _serve_image(request: Request, session_id: str, image_type: str) -> FileResponse:
    """Resolve the file path for the requested image type and return it.

    File path conventions (matching pipeline output structure):
    - original:   <upload_dir>/<session_id>_*   (any extension)
    - protected:  <results_dir>/<session_id>_protected.png
    - risk-map:   <results_dir>/<session_id>_risk_map.png
    """
    settings = request.app.state.settings

    if image_type == "original":
        path = _find_original(settings.upload_dir, session_id)
    elif image_type == "protected":
        path = os.path.join(settings.results_dir, f"{session_id}_protected.png")
    elif image_type == "risk-map":
        path = os.path.join(settings.results_dir, f"{session_id}_risk_map.png")
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "PIPELINE_NOT_FOUND",
                    "message": f"Unknown image type '{image_type}'.",
                    "details": {},
                }
            },
        )

    if path is None or not os.path.isfile(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "PIPELINE_NOT_FOUND",
                    "message": (
                        f"Image '{image_type}' is not yet available for "
                        f"session '{session_id}'."
                    ),
                    "details": {
                        "session_id": session_id,
                        "image_type": image_type,
                    },
                }
            },
        )

    return FileResponse(
        path=path,
        media_type=_MIME[image_type],
        filename=os.path.basename(path),
    )


def _find_original(upload_dir: str, session_id: str) -> str | None:
    """Scan the upload directory for a file whose name starts with session_id."""
    if not os.path.isdir(upload_dir):
        return None
    prefix = f"{session_id}_"
    for entry in os.scandir(upload_dir):
        if entry.is_file() and entry.name.startswith(prefix):
            return entry.path
    return None


def _assert_session_ownership(session: object, session_id: str) -> None:
    if getattr(session, "session_id", None) != session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "PIPELINE_NOT_FOUND",
                    "message": f"No pipeline found for session '{session_id}'.",
                    "details": {"session_id": session_id},
                }
            },
        )
