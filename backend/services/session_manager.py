"""
session_manager.py — In-memory session store with TTL and thread-safe access.

One SessionRecord per active pipeline run. TTL is 8 hours; a background
asyncio task sweeps expired entries every 15 minutes (started by main.py).

Thread safety: SessionManager uses a threading.Lock so worker threads running
the synchronous PipelineOrchestrator can safely mutate session state while
the async event loop reads it for REST responses.
"""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_TTL_SECONDS: int = 8 * 3600  # 8 hours
SESSION_SWEEP_INTERVAL_SECONDS: int = 15 * 60  # 15 minutes

# Valid pipeline stage names (ordered)
VALID_STAGES: List[str] = [
    "detection",
    "risk",
    "consent",
    "strategy",
    "sam",
    "execution",
    "export",
]

# STAGE_DEPENDENCY_MAP[stage] = stages that must be re-run when stage is re-run
STAGE_DEPENDENCY_MAP: Dict[str, List[str]] = {
    "detection": ["risk", "consent", "strategy", "sam", "execution", "export"],
    "risk":      ["strategy", "sam", "execution", "export"],
    "consent":   ["strategy", "sam", "execution", "export"],
    "strategy":  ["sam", "execution", "export"],
    "sam":       ["execution", "export"],
    "execution": ["export"],
    "export":    [],
}

# ---------------------------------------------------------------------------
# Session record
# ---------------------------------------------------------------------------


@dataclass
class SessionRecord:
    """
    All mutable state for a single pipeline session.

    Fields are accessed from both the async event loop (REST handlers) and
    worker threads (PipelineOrchestrator). Use session_manager._lock when
    performing multi-field atomic updates from the worker thread.
    """

    session_id: str
    token: str
    created_at: float
    last_accessed: float

    # Pipeline state machine
    # Valid values: "idle" | "queued" | "running" | "hitl_paused" | "completed" | "failed"
    status: str = "idle"
    current_stage: str = "detection"

    # Stage-level progress (for stage_progress WS events)
    stage_step: int = 0
    stage_elements_processed: int = 0
    stage_elements_total: int = 0

    # Image paths
    image_path: Optional[str] = None
    protected_image_path: Optional[str] = None
    risk_map_path: Optional[str] = None
    protection_preview_path: Optional[str] = None

    # Run configuration submitted with POST /pipeline/run
    config: Optional[Dict[str, Any]] = None

    # Cached pipeline stage outputs (populated as pipeline advances)
    detections: Any = None            # DetectionResults
    risk_result: Any = None           # RiskAnalysisResult
    strategy_result: Any = None       # StrategyRecommendations
    pipeline_output: Any = None       # PipelineOutput (set on completion)
    execution_report: Any = None      # ExecutionReport

    # Per-stage timing: stage_name → elapsed_ms
    stage_timings: Dict[str, float] = field(default_factory=dict)

    # HITL state
    # checkpoint values: "risk_review" | "strategy_review" | "execution_verify"
    hitl_checkpoint: Optional[str] = None
    hitl_pending_approval: bool = False
    hitl_elements_requiring_review: List[str] = field(default_factory=list)
    hitl_actions_available: List[str] = field(default_factory=list)
    # Worker thread blocks on this Event; /approve endpoint sets it
    hitl_event: threading.Event = field(default_factory=threading.Event)

    # Override accumulator — frontend batches these before calling /rerun
    pending_overrides: List[Dict[str, Any]] = field(default_factory=list)

    # Immutable append-only audit trail
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

    # Error detail (set on failure)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_stage: Optional[str] = None

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def is_expired(self) -> bool:
        """True when the session has exceeded its TTL."""
        return (time.time() - self.last_accessed) > SESSION_TTL_SECONDS

    @property
    def is_running(self) -> bool:
        return self.status in ("queued", "running")


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


class SessionManager:
    """
    Thread-safe in-memory session store.

    Key design decisions:
    - Uses threading.Lock (not asyncio.Lock) so the synchronous pipeline
      worker thread can safely acquire it.
    - Two indexes: _sessions (session_id → record) and _token_index
      (token → session_id) for O(1) lookup by either key.
    - touch() is called by the auth dependency on every authenticated request
      to reset the TTL clock.
    """

    def __init__(self, ttl_hours: Optional[int] = None) -> None:
        """
        Args:
            ttl_hours: Override the default 8-hour session TTL.  When None the
                       module-level SESSION_TTL_SECONDS constant is used.
        """
        global SESSION_TTL_SECONDS  # noqa: PLW0603
        if ttl_hours is not None:
            SESSION_TTL_SECONDS = ttl_hours * 3600

        self._sessions: Dict[str, SessionRecord] = {}   # session_id → record
        self._token_index: Dict[str, str] = {}          # token → session_id
        self._lock = threading.Lock()
        self._sweep_task: Optional[asyncio.Task] = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def create_session(self) -> SessionRecord:
        """
        Generate a new session with unique session_id and Bearer token.

        Both identifiers are UUID4 hex strings.  The token is the credential
        presented in the Authorization header; the session_id is embedded in
        URL paths.

        Returns:
            The newly created SessionRecord.
        """
        session_id = uuid.uuid4().hex
        token = uuid.uuid4().hex
        now = time.time()

        record = SessionRecord(
            session_id=session_id,
            token=token,
            created_at=now,
            last_accessed=now,
            status="idle",
        )

        with self._lock:
            self._sessions[session_id] = record
            self._token_index[token] = session_id

        logger.info("Session created: session_id=%s", session_id)
        return record

    def validate_token(self, token: str) -> Optional[SessionRecord]:
        """
        Look up a session by Bearer token.

        Touches last_accessed on success to reset TTL.

        Returns:
            SessionRecord if token is valid and session is not expired.
            None otherwise.
        """
        with self._lock:
            session_id = self._token_index.get(token)
            if session_id is None:
                return None
            record = self._sessions.get(session_id)
            if record is None or record.is_expired:
                # Clean up stale entry
                self._token_index.pop(token, None)
                if session_id:
                    self._sessions.pop(session_id, None)
                return None
            record.last_accessed = time.time()
            return record

    def get_by_id(self, session_id: str) -> Optional[SessionRecord]:
        """
        Look up a session by session_id.

        Touches last_accessed on success to reset TTL.

        Returns:
            SessionRecord if found and not expired, None otherwise.
        """
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None or record.is_expired:
                return None
            record.last_accessed = time.time()
            return record

    def get_by_token(self, token: str) -> Optional[SessionRecord]:
        """Alias for validate_token — preferred name in auth dependency."""
        return self.validate_token(token)

    def touch(self, session: SessionRecord) -> None:
        """Reset the TTL clock for a session."""
        with self._lock:
            session.last_accessed = time.time()

    def append_audit_record(
        self,
        session_id: str,
        record: Dict[str, Any],
    ) -> None:
        """
        Append an OverrideRecord to the session's immutable audit trail.

        Thread-safe; can be called from any thread.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.audit_trail.append(record)

    def list_sessions(self) -> List[SessionRecord]:
        """
        Return all non-expired sessions (for history endpoint).

        Does NOT touch last_accessed.
        """
        with self._lock:
            return [s for s in self._sessions.values() if not s.is_expired]

    def cleanup_expired(self) -> int:
        """
        Remove all expired sessions from both indexes.

        Returns:
            Number of sessions removed.
        """
        with self._lock:
            expired_ids = [
                sid
                for sid, record in self._sessions.items()
                if record.is_expired
            ]
            for sid in expired_ids:
                record = self._sessions.pop(sid, None)
                if record is not None:
                    self._token_index.pop(record.token, None)

        if expired_ids:
            logger.info(
                "Session sweep: removed %d expired session(s)", len(expired_ids)
            )
        return len(expired_ids)

    # Alias used in architecture doc
    sweep_expired = cleanup_expired

    # -----------------------------------------------------------------------
    # Background sweep coroutine
    # -----------------------------------------------------------------------

    async def sweep_loop(self) -> None:
        """
        Long-running coroutine that periodically removes expired sessions.

        Start with asyncio.create_task(session_manager.sweep_loop()) in
        the FastAPI lifespan startup handler.
        """
        logger.info(
            "Session sweep loop started (interval=%ds)",
            SESSION_SWEEP_INTERVAL_SECONDS,
        )
        while True:
            await asyncio.sleep(SESSION_SWEEP_INTERVAL_SECONDS)
            count = self.cleanup_expired()
            if count:
                logger.info("Sweep removed %d session(s)", count)

    def start_sweep_task(self) -> asyncio.Task:
        """
        Convenience method called from lifespan startup.

        Returns the asyncio.Task so the caller can cancel it on shutdown.
        """
        self._sweep_task = asyncio.create_task(self.sweep_loop())
        return self._sweep_task

    def stop_sweep_task(self) -> None:
        """Cancel the background sweep task on shutdown."""
        if self._sweep_task and not self._sweep_task.done():
            self._sweep_task.cancel()
