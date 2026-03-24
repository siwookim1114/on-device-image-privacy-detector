from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from backend.schemas.responses import SessionResponse
from backend.services.session_manager import SESSION_TTL_SECONDS

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/session",
    response_model=SessionResponse,
    status_code=201,
    summary="Create a new session and return a bearer token.",
)
async def create_session(request: Request) -> SessionResponse:
    """Create a new anonymous session.

    The SessionManager (injected via app.state) mints a UUID token with an
    8-hour TTL.  No credentials are required — the token itself is the session
    identity.
    """
    session_manager = request.app.state.session_manager
    # SessionManager exposes create_session(); _StubSessionManager raises 503.
    session = session_manager.create_session()

    expires_ts = session.created_at + SESSION_TTL_SECONDS
    expires_at = datetime.fromtimestamp(expires_ts, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )

    return SessionResponse(
        session_id=session.session_id,
        token=session.token,
        expires_at=expires_at,
    )
