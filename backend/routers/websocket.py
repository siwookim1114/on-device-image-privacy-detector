"""
routers/websocket.py — WebSocket endpoint for pipeline progress streaming.

Endpoint: WS /api/v1/pipeline/{session_id}/stream?token=<token>

Protocol (Server → Client only):
  connected        → sent immediately on connection; current session state
  stage_start      → pipeline entering a new stage
  stage_progress   → periodic progress within a stage (elements processed)
  stage_complete   → stage finished with timing + summary
  hitl_checkpoint  → pipeline paused; awaiting user approval
  pipeline_resumed → pipeline unpaused after /approve
  pipeline_complete→ all stages done
  pipeline_error   → fatal or recoverable error

Client → Server:
  Only standard WebSocket ping/pong keepalive is expected.  Any text messages
  received are silently discarded (all writes go through REST endpoints).

Authentication:
  Token is passed as ?token= query parameter (WebSocket handshake limitation).
  If the token is missing or invalid the connection is rejected with code 4001.
"""

import asyncio
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/pipeline/{session_id}/stream")
async def pipeline_stream(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(..., description="Bearer token issued by POST /auth/session"),
) -> None:
    """
    WebSocket endpoint for real-time pipeline progress streaming.

    Connection flow:
    1. Validate token against session_id before accepting the handshake.
    2. Accept and register the connection with WebSocketManager.
    3. Send an immediate "connected" event with current session state.
    4. Loop: receive and discard client messages; handle ping/pong keepalive.
    5. On disconnect: unregister and exit cleanly.

    The endpoint reads session_manager and ws_manager from app.state, which
    are populated during the FastAPI lifespan startup handler.
    """
    # Retrieve singletons from app state.
    # main.py must set app.state.ws_manager (WebSocketManager) and
    # app.state.session_manager (SessionManager) in its lifespan startup.
    app = websocket.app
    session_manager = getattr(app.state, "session_manager", None)
    ws_manager = getattr(app.state, "ws_manager", None)

    if session_manager is None or ws_manager is None:
        await websocket.close(
            code=1011,
            reason="Service unavailable: session or WebSocket manager not initialised",
        )
        logger.error(
            "WS rejected (service not ready): session_manager=%s ws_manager=%s",
            session_manager,
            ws_manager,
        )
        return

    # -----------------------------------------------------------------------
    # Token + session validation (pre-accept)
    # -----------------------------------------------------------------------
    session = session_manager.get_by_token(token)

    if session is None:
        # Reject without accepting — close code 4001 (custom: unauthorized)
        await websocket.close(code=4001, reason="Unauthorized: missing or expired token")
        logger.warning(
            "WS rejected (invalid token): session_id=%s", session_id
        )
        return

    if session.session_id != session_id:
        # Token is valid but belongs to a different session
        await websocket.close(code=4001, reason="Unauthorized: token does not match session_id")
        logger.warning(
            "WS rejected (session_id mismatch): token_session=%s  path_session=%s",
            session.session_id,
            session_id,
        )
        return

    # -----------------------------------------------------------------------
    # Accept the handshake and register with the connection manager
    # -----------------------------------------------------------------------
    await ws_manager.connect(session_id, websocket)

    # -----------------------------------------------------------------------
    # Send initial "connected" event with current state
    # -----------------------------------------------------------------------
    stages_completed = [
        stage
        for stage, timing_key in {
            "detection":  "detection_ms",
            "risk":       "risk_assessment_ms",
            "consent":    "consent_identity_ms",
            "strategy":   "strategy_ms",
            "sam":        "sam_segmentation_ms",
            "execution":  "execution_ms",
            "export":     "export_ms",
        }.items()
        if timing_key in session.stage_timings
    ]

    await ws_manager.send_personal(
        websocket,
        event_type="connected",
        session_id=session_id,
        payload={
            "current_status": session.status,
            "current_stage": session.current_stage,
            "stages_completed": stages_completed,
        },
    )

    # -----------------------------------------------------------------------
    # Main receive loop — keepalive + graceful disconnect
    # -----------------------------------------------------------------------
    try:
        while True:
            # Wait for incoming messages (client sends only ping/pong; we
            # receive the raw frame and discard any text/bytes payload).
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=30.0,  # 30 s receive timeout; guards against zombie connections
                )
            except asyncio.TimeoutError:
                # No message received in 30 s — check if socket is still open
                if websocket.client_state != WebSocketState.CONNECTED:
                    break
                # Send a ping to probe liveness
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
                continue

            # Client initiated close
            if message.get("type") == "websocket.disconnect":
                break

            # Text/bytes messages from client are intentionally ignored.
            # All writes go through REST endpoints per the API contract.

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug(
            "WS receive loop error (session=%s): %s", session_id, exc
        )
    finally:
        ws_manager.disconnect(session_id, websocket)
        logger.info("WS stream closed: session_id=%s", session_id)
