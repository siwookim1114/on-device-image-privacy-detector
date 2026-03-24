from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)


async def require_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> object:
    """Validate Bearer token against the SessionManager stored in app.state.

    Returns the Session object on success, raises HTTP 401 on failure.
    The session_manager is injected via app.state so that this dependency
    works across all routers without circular imports.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": "Missing or malformed Authorization header.",
                    "details": {},
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # Retrieve session manager from app.state (set during lifespan startup).
    session_manager = getattr(request.app.state, "session_manager", None)
    if session_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "PIPELINE_ERROR",
                    "message": "Session manager not initialised.",
                    "details": {},
                }
            },
        )

    session = session_manager.get_by_token(token)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": "Token is missing, expired, or invalid.",
                    "details": {},
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    return session
