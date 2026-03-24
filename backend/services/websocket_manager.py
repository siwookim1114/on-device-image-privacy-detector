"""
websocket_manager.py — Per-session WebSocket connection registry and event broadcast.

Design principles:
- Server → Client only: pipeline progress events flow outward; writes go via REST.
- Multiple tabs: one session_id can have multiple simultaneous WS connections.
  All connections for a session receive every broadcast.
- Silent disconnect handling: if a client drops mid-stream the send error is
  swallowed and the connection is evicted from the registry.
- Envelope format matches API_CONTRACT.md exactly.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso_now() -> str:
    """Current UTC timestamp in ISO-8601 format (Z suffix)."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_envelope(
    session_id: str,
    event_type: str,
    payload: dict,
) -> dict:
    """
    Construct the standard WS event envelope.

    Shape::

        {
            "type": "<event_type>",
            "session_id": "<session_id>",
            "timestamp": "2026-03-24T12:00:00.000Z",
            "payload": { ... }
        }
    """
    return {
        "type": event_type,
        "session_id": session_id,
        "timestamp": _iso_now(),
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# WebSocketManager
# ---------------------------------------------------------------------------


class WebSocketManager:
    """
    Registry of active WebSocket connections, keyed by session_id.

    Thread-safety note: all methods are async and run on the event loop.
    The synchronous pipeline worker uses broadcast_from_thread() via
    asyncio.run_coroutine_threadsafe() — never call the async methods
    directly from a worker thread.
    """

    def __init__(self) -> None:
        # session_id → set of active WebSocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}

    # -----------------------------------------------------------------------
    # Connection lifecycle
    # -----------------------------------------------------------------------

    async def connect(self, session_id: str, ws: WebSocket) -> None:
        """
        Accept a WebSocket handshake and register the connection.

        Called from the WS endpoint after token validation.
        """
        await ws.accept()
        if session_id not in self._connections:
            self._connections[session_id] = set()
        self._connections[session_id].add(ws)
        logger.info(
            "WS connected: session_id=%s  active=%d",
            session_id,
            len(self._connections[session_id]),
        )

    def disconnect(self, session_id: str, ws: WebSocket) -> None:
        """
        Remove a WebSocket from the registry.

        Safe to call even if the connection was already removed.
        """
        conns = self._connections.get(session_id)
        if conns is not None:
            conns.discard(ws)
            if not conns:
                del self._connections[session_id]
        logger.info("WS disconnected: session_id=%s", session_id)

    # -----------------------------------------------------------------------
    # Broadcasting
    # -----------------------------------------------------------------------

    async def broadcast(
        self,
        session_id: str,
        event_type: str,
        payload: dict,
    ) -> None:
        """
        Send an event to every WebSocket registered for *session_id*.

        Connections that raise during send are silently evicted so a dropped
        tab never blocks pipeline progress events to healthy connections.

        Args:
            session_id: Target session.
            event_type: One of the event names defined in API_CONTRACT.md
                        (e.g. "stage_start", "stage_complete", "hitl_checkpoint").
            payload:    Event-specific data dict.
        """
        envelope = _build_envelope(session_id, event_type, payload)
        conns = self._connections.get(session_id)
        if not conns:
            # No clients connected — events are intentionally dropped.
            # They can reconstruct state via GET /pipeline/{id}/status.
            return

        dead: list = []
        for ws in list(conns):
            try:
                await ws.send_json(envelope)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "WS send failed for session_id=%s (%s) — evicting",
                    session_id,
                    exc,
                )
                dead.append(ws)

        for ws in dead:
            self.disconnect(session_id, ws)

    async def send_personal(
        self,
        ws: WebSocket,
        event_type: str,
        payload: dict,
        session_id: str = "",
    ) -> None:
        """
        Send an event to a single WebSocket connection.

        Used for the initial "connected" event sent only to the newly
        joining client (not broadcast to all tabs).

        Args:
            ws:         Target WebSocket.
            event_type: Event name.
            payload:    Event-specific data.
            session_id: Optional; included in the envelope if provided.
        """
        envelope = _build_envelope(session_id, event_type, payload)
        try:
            await ws.send_json(envelope)
        except Exception as exc:  # noqa: BLE001
            logger.debug("WS personal send failed (%s)", exc)

    # -----------------------------------------------------------------------
    # Thread-safe broadcast (called from synchronous worker threads)
    # -----------------------------------------------------------------------

    def broadcast_from_thread(
        self,
        session_id: str,
        event_type: str,
        payload: dict,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Schedule a broadcast from a synchronous worker thread.

        Uses asyncio.run_coroutine_threadsafe() to safely cross the
        thread/event-loop boundary.  Blocks until the coroutine is submitted
        (but NOT until the broadcast completes).

        Args:
            session_id: Target session.
            event_type: Event name.
            payload:    Event-specific data.
            loop:       The running event loop (passed in from the async
                        context that spawned the worker thread).
        """
        future = asyncio.run_coroutine_threadsafe(
            self.broadcast(session_id, event_type, payload),
            loop,
        )
        # We do not await the future — fire-and-forget from the worker side.
        # Log any exception that surfaces to prevent silent failures.
        def _log_exc(f: asyncio.Future) -> None:
            exc = f.exception()
            if exc is not None:
                logger.error(
                    "WS broadcast_from_thread error (session=%s, event=%s): %s",
                    session_id,
                    event_type,
                    exc,
                )

        future.add_done_callback(_log_exc)

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def get_connection_count(self, session_id: str) -> int:
        """Return the number of active connections for a session."""
        return len(self._connections.get(session_id, set()))

    async def broadcast_all_sessions(
        self, event_type: str, payload: dict
    ) -> None:
        """
        Broadcast an event to every connected session.

        Reserved for admin-level events; not currently used by the pipeline.
        """
        for session_id in list(self._connections.keys()):
            await self.broadcast(session_id, event_type, payload)
