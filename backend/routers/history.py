from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse

from backend.middleware.auth_middleware import require_auth
from backend.schemas.responses import HistoryResponse, SessionSummary

router = APIRouter(prefix="/history", tags=["history"])

ArtifactType = Literal["protected", "risk_json", "provenance"]


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=HistoryResponse,
    summary="List all sessions for the authenticated token (paginated).",
)
async def list_history(
    request: Request,
    session: Annotated[object, Depends(require_auth)],
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page."),
) -> HistoryResponse:
    """Return a paginated list of past pipeline sessions.

    The SessionManager (or a dedicated HistoryStore wired by another agent)
    is queried.  Falls back to an empty list when services are not yet wired.
    """
    session_manager = request.app.state.session_manager

    all_sessions: list[SessionSummary]
    # Prefer list_sessions() (real SessionManager); fall back to list_all()
    # (stub / legacy) or an empty list if neither exists.
    list_fn = getattr(session_manager, "list_sessions", None) or getattr(
        session_manager, "list_all", None
    )
    if list_fn is not None:
        raw = list_fn()
        all_sessions = []
        for s in raw:
            # SessionRecord.created_at is a Unix timestamp float.
            created_raw = getattr(s, "created_at", None)
            if isinstance(created_raw, float):
                created_iso = datetime.fromtimestamp(
                    created_raw, tz=timezone.utc
                ).isoformat().replace("+00:00", "Z")
            elif hasattr(created_raw, "isoformat"):
                created_iso = created_raw.isoformat()
            else:
                created_iso = str(created_raw or "")

            # image_filename: prefer dedicated attribute, then derive from image_path
            image_filename = getattr(s, "image_filename", None)
            if image_filename is None:
                image_path = getattr(s, "image_path", None)
                if image_path:
                    image_filename = os.path.basename(image_path)

            # protections_applied: count from execution_report if available
            protections_applied = getattr(s, "protections_applied", 0)
            exec_report = getattr(s, "execution_report", None)
            if exec_report is not None and hasattr(exec_report, "transformations_applied"):
                protections_applied = len(exec_report.transformations_applied)

            all_sessions.append(
                SessionSummary(
                    session_id=s.session_id,
                    created_at=created_iso,
                    status=s.status,
                    image_filename=image_filename,
                    protections_applied=protections_applied,
                )
            )
    else:
        all_sessions = []

    total = len(all_sessions)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = all_sessions[start:end]

    return HistoryResponse(
        items=page_items,
        total=total,
        page=page,
        page_size=page_size,
        has_next=end < total,
    )


# ---------------------------------------------------------------------------
# GET /history/{session_id}/download
# ---------------------------------------------------------------------------

@router.get(
    "/{session_id}/download",
    response_class=FileResponse,
    summary="Download a pipeline artifact for a historical session.",
)
async def download_artifact(
    session_id: str,
    artifact: ArtifactType = Query(
        default="protected",
        description="Artifact type: protected | risk_json | provenance",
    ),
    request: Request = ...,
    session: Annotated[object, Depends(require_auth)] = ...,
) -> FileResponse:
    settings = request.app.state.settings
    results_dir: str = settings.results_dir

    artifact_map: dict[ArtifactType, tuple[str, str]] = {
        "protected": (
            os.path.join(results_dir, f"{session_id}_protected.png"),
            "image/png",
        ),
        "risk_json": (
            os.path.join(results_dir, f"{session_id}_risk_results.json"),
            "application/json",
        ),
        "provenance": (
            os.path.join(results_dir, f"{session_id}_provenance.json"),
            "application/json",
        ),
    }

    path, mime_type = artifact_map[artifact]

    if not os.path.isfile(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "PIPELINE_NOT_FOUND",
                    "message": (
                        f"Artifact '{artifact}' not found for session "
                        f"'{session_id}'."
                    ),
                    "details": {
                        "session_id": session_id,
                        "artifact": artifact,
                    },
                }
            },
        )

    return FileResponse(
        path=path,
        media_type=mime_type,
        filename=os.path.basename(path),
    )


# ---------------------------------------------------------------------------
# GET /history/{session_id}/audit
# ---------------------------------------------------------------------------

@router.get(
    "/{session_id}/audit",
    summary="Export the full audit trail for a session as JSON.",
)
async def export_audit(
    session_id: str,
    request: Request,
    session: Annotated[object, Depends(require_auth)],
) -> FileResponse:
    """Serve the audit trail JSON file produced during pipeline execution."""
    settings = request.app.state.settings
    audit_path = os.path.join(
        settings.results_dir, f"{session_id}_audit_trail.json"
    )

    if not os.path.isfile(audit_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "PIPELINE_NOT_FOUND",
                    "message": (
                        f"Audit trail not found for session '{session_id}'."
                    ),
                    "details": {"session_id": session_id},
                }
            },
        )

    return FileResponse(
        path=audit_path,
        media_type="application/json",
        filename=f"{session_id}_audit_trail.json",
    )
