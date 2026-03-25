"""
CoordinatorSession -- public entry point for the Coordinator Agent.

Usage (from backend/routers/hitl.py):
    coordinator = CoordinatorSession(
        session_id=session_id,
        ctx=node_context,
    )
    result = coordinator.handle_message("blur all faces")

    # result is a dict:
    # {
    #   "intent": {"action": ..., "confidence": ..., "natural_language": ...},
    #   "response_text": str,
    #   "pipeline_action_taken": Optional[str],
    #   "suggestions": List[str],
    # }

Each CoordinatorSession manages:
  - One CoordinatorState (conversation + pipeline state)
  - The compiled LangGraph app (or fallback sequential coordinator)
  - History truncation to keep context within VLM limits
  - Thread-safety via threading.Lock
  - Snapshot/undo support via PipelineSnapshot
  - Adaptive learning via PreferenceManager
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from agents.coordinator.coordinator_graph import build_coordinator_graph
from agents.coordinator.nodes import NodeContext
from agents.coordinator.state import (
    CoordinatorState,
    InnerPipelineState,
    DisagreementEvent,
    PipelineSnapshot,
)

logger = logging.getLogger(__name__)

# Maximum conversation turns to keep in history (to prevent context overflow)
_MAX_HISTORY_TURNS = 20


class CoordinatorSession:
    """
    Manages one user session with the Coordinator Agent.

    Thread safety: handle_message acquires self._lock so concurrent calls
    from WebSocket handlers or thread-pool workers are serialised.  The
    underlying LangGraph execution may call synchronous pipeline agents.

    Args:
        session_id: The pipeline session UUID.  Must match the session_id
                    in SessionManager to allow HITL event signalling.
        ctx:        NodeContext carrying all shared services and agents.
                    Pass None to create a context-free session (read-only
                    queries and no pipeline execution).
        image_path: Path to the image being processed.  May be updated
                    later by calling set_image_path().
        fallback_only: If True, no VLM calls are made in pipeline nodes.
        preference_manager: Optional PreferenceManager for adaptive learning.
                            If provided, it is attached to ctx for use by
                            the coordinator graph.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        ctx: Optional[NodeContext] = None,
        image_path: Optional[str] = None,
        fallback_only: bool = False,
        preference_manager: Any = None,
    ) -> None:
        self.session_id: str = session_id or uuid.uuid4().hex
        self._ctx: NodeContext = ctx or NodeContext(fallback_only=fallback_only)
        self._fallback_only = fallback_only
        self._lock = threading.Lock()

        # Attach preference_manager to ctx so coordinator_graph can access it
        if preference_manager is not None:
            self._ctx.preference_manager = preference_manager  # type: ignore[attr-defined]
        self._preference_manager = preference_manager

        # Build (or rebuild) the compiled graph
        self._graph = build_coordinator_graph(self._ctx)

        # Initialise empty CoordinatorState
        initial_pipeline_state: InnerPipelineState = {  # type: ignore[assignment]
            "session_id": self.session_id,
            "image_path": image_path or "",
            "detections": None,
            "risk_result": None,
            "identity_assessments": None,
            "strategy_result": None,
            "seg_results": None,
            "execution_report": None,
            "protected_image_path": None,
            "risk_map_path": None,
            "strategy_json_path": None,
            "entry_stage": "detection",
            "fallback_only": fallback_only,
            "pending_modifications": [],
            "stage_timings": {},
            "errors": {},
            "phase_disagreements": [],
            "_cached_image": None,
        }

        self._state: CoordinatorState = {  # type: ignore[assignment]
            "session_id": self.session_id,
            "conversation_history": [],
            "current_intent": None,
            "pipeline_state": initial_pipeline_state,
            "hitl_checkpoint": None,
            "pending_user_decision": False,
            "hitl_confidence": None,
            "response_text": "",
            "suggestions": [],
            "pipeline_action_taken": None,
            "audit_trail": [],
            "snapshots": [],
            "hitl_presentation": None,
        }
    # Public API

    def handle_message(self, message: str) -> Dict[str, Any]:
        """
        Process one user message and return the coordinator's response.

        This is a SYNCHRONOUS method.  The underlying graph may execute
        synchronous pipeline code.  For CPU-bound pipeline runs, wrap
        the call in a thread pool executor.

        Thread-safe: acquires self._lock for the duration of processing.

        Args:
            message: Raw user natural language message.

        Returns:
            Dict with keys:
              intent  -- {action, target_stage, target_elements, confidence,
                         natural_language}
              response_text -- str: coordinator's reply to the user
              pipeline_action_taken -- Optional[str]
              suggestions -- List[str]
              disagreements -- List[DisagreementEvent] (if any)
              hitl_presentation -- Optional[dict] (grouped HITL data)
        """
        if not message or not message.strip():
            return {
                "intent": {
                    "action": "query",
                    "target_stage": None,
                    "target_elements": [],
                    "confidence": 0.0,
                    "natural_language": "",
                },
                "response_text": "Please enter a message.",
                "pipeline_action_taken": None,
                "suggestions": [],
                "disagreements": [],
                "hitl_presentation": None,
            }

        with self._lock:
            # Append user message to history
            history = list(self._state.get("conversation_history") or [])
            history.append({"role": "user", "content": message.strip()})

            # Truncate history to avoid context overflow
            if len(history) > _MAX_HISTORY_TURNS * 2:
                # Keep the first 2 turns (context) + last N turns
                history = history[:2] + history[-((_MAX_HISTORY_TURNS - 1) * 2):]

            self._state["conversation_history"] = history

            # Reset turn-specific fields
            self._state["response_text"] = ""
            self._state["pipeline_action_taken"] = None

            try:
                # Invoke the LangGraph (or fallback) coordinator synchronously
                new_state = self._graph.invoke(self._state)
                if new_state is not None:
                    self._state = new_state
            except Exception as exc:
                logger.error(
                    "CoordinatorSession.handle_message failed (session=%s): %s",
                    self.session_id, exc,
                )
                self._state["response_text"] = (
                    f"An error occurred while processing your request: {exc}"
                )
                self._state["pipeline_action_taken"] = "error"

            # Build return value
            intent_dict = self._state.get("current_intent") or {}
            pipeline_state = self._state.get("pipeline_state") or {}

            return {
                "intent": {
                    "action": intent_dict.get("action", "query"),
                    "target_stage": intent_dict.get("target_stage"),
                    "target_elements": intent_dict.get("target_elements") or [],
                    "confidence": float(intent_dict.get("confidence", 0.0)),
                    "natural_language": intent_dict.get("natural_language", message),
                },
                "response_text": self._state.get("response_text", ""),
                "pipeline_action_taken": self._state.get("pipeline_action_taken"),
                "suggestions": list(self._state.get("suggestions") or []),
                "disagreements": list(pipeline_state.get("phase_disagreements") or []),
                "hitl_presentation": self._state.get("hitl_presentation"),
            }

    def set_image_path(self, image_path: str) -> None:
        """Update the image path in the current pipeline state."""
        with self._lock:
            pipeline_state = dict(self._state.get("pipeline_state") or {})
            pipeline_state["image_path"] = image_path
            # Clear cached image when path changes
            pipeline_state["_cached_image"] = None
            self._state["pipeline_state"] = pipeline_state  # type: ignore[assignment]

    def get_pipeline_state(self) -> Optional[InnerPipelineState]:
        """Return the current InnerPipelineState (read-only snapshot)."""
        return self._state.get("pipeline_state")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return the conversation history as a list of {role, content} dicts."""
        return list(self._state.get("conversation_history") or [])

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the coordinator-level audit trail."""
        return list(self._state.get("audit_trail") or [])

    def is_pending_user_decision(self) -> bool:
        """True if the coordinator is waiting for user approval at a HITL gate."""
        return bool(self._state.get("pending_user_decision", False))

    def get_hitl_info(self) -> Dict[str, Any]:
        """Return the current HITL checkpoint and confidence report."""
        return {
            "checkpoint": self._state.get("hitl_checkpoint"),
            "pending": self._state.get("pending_user_decision", False),
            "confidence": self._state.get("hitl_confidence"),
            "presentation": self._state.get("hitl_presentation"),
        }

    def get_disagreements(self) -> List[Dict[str, Any]]:
        """Return all Phase 1 vs Phase 2 disagreements for the current session."""
        pipeline_state = self._state.get("pipeline_state") or {}
        return list(pipeline_state.get("phase_disagreements") or [])
    # Snapshot management

    def _save_snapshot(self) -> Optional[str]:
        """
        Manually save a snapshot of the current pipeline state.
        Returns the snapshot_id or None if nothing to snapshot.
        """
        pipeline_state = self._state.get("pipeline_state") or {}
        if pipeline_state.get("detections") is None:
            return None

        from agents.coordinator.coordinator_graph import _take_snapshot, _MAX_SNAPSHOTS

        snapshot = _take_snapshot(self._state)
        snapshots = list(self._state.get("snapshots") or [])
        snapshots.append(snapshot)
        if len(snapshots) > _MAX_SNAPSHOTS:
            snapshots = snapshots[-_MAX_SNAPSHOTS:]
        self._state["snapshots"] = snapshots
        return snapshot.get("snapshot_id")  # type: ignore[union-attr]

    def _restore_snapshot(self, snapshot_id: Optional[str] = None) -> bool:
        """
        Restore a snapshot by ID, or the most recent one if ID is None.
        Returns True if restoration succeeded, False otherwise.
        """
        from agents.coordinator.coordinator_graph import _restore_snapshot

        snapshots = list(self._state.get("snapshots") or [])
        if not snapshots:
            return False

        if snapshot_id is None:
            target = snapshots.pop()
        else:
            target = None
            for i, s in enumerate(snapshots):
                if s.get("snapshot_id") == snapshot_id:
                    target = snapshots.pop(i)
                    break
            if target is None:
                return False

        self._state = _restore_snapshot(self._state, target)
        self._state["snapshots"] = snapshots
        return True

    def get_snapshots(self) -> List[Dict[str, Any]]:
        """Return summary list of available snapshots."""
        snapshots = self._state.get("snapshots") or []
        return [
            {
                "snapshot_id": s.get("snapshot_id", ""),
                "timestamp": s.get("timestamp", 0),
                "entry_stage": s.get("entry_stage", ""),
            }
            for s in snapshots
        ]
    # Serialisation helpers (for storing session state in SessionManager)

    def get_last_intent(self) -> Optional[Dict[str, Any]]:
        """Return the last classified intent dict (serialised ParsedIntent)."""
        return self._state.get("current_intent")

    def export_state_summary(self) -> Dict[str, Any]:
        """
        Export a lightweight summary of the coordinator state for persistence.

        Suitable for storing in SessionRecord.coordinator_history.
        """
        pipeline_state = self._state.get("pipeline_state") or {}
        timings = pipeline_state.get("stage_timings") or {}
        errors = pipeline_state.get("errors") or {}
        disagreements = pipeline_state.get("phase_disagreements") or []

        return {
            "session_id": self.session_id,
            "conversation_turns": len(self._state.get("conversation_history") or []) // 2,
            "last_intent_action": (
                (self._state.get("current_intent") or {}).get("action")
            ),
            "pipeline_entry_stage": pipeline_state.get("entry_stage"),
            "stage_timings": timings,
            "stage_errors": list(errors.keys()),
            "hitl_checkpoint": self._state.get("hitl_checkpoint"),
            "pending_user_decision": self._state.get("pending_user_decision", False),
            "pipeline_action_taken": self._state.get("pipeline_action_taken"),
            "disagreement_count": len(disagreements),
            "snapshot_count": len(self._state.get("snapshots") or []),
        }
