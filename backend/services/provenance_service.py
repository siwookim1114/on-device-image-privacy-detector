"""
provenance_service.py — Per-session event log for the privacy pipeline.

Responsibilities
----------------
- open_session(): initialise an in-memory event buffer and write a skeleton
  MongoDB document so queries can find the session immediately.
- record(): append a structured ProvenanceEvent.  Fire-and-forget — never raises.
- record_override(): bridge from SafetyKernel.OverrideRecord to a provenance event.
- finalize_session(): write PIPELINE_COMPLETE, compute a summary, flush to
  MongoDB, and persist a JSON file in both the provenance_logs dir and
  (optionally) the per-session results dir for download.
- get_provenance() / get_hitl_records(): query helpers for the history router.
"""

import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.models import ProvenanceEventType

logger = logging.getLogger(__name__)


class ProvenanceService:
    """
    Tracks every significant event that occurs during a pipeline session.

    Args:
        mongo_collection: Optional pymongo Collection for persistent storage.
        provenance_logs_dir: Directory where per-session JSON files are written.
    """

    def __init__(
        self,
        mongo_collection=None,
        provenance_logs_dir: str = "data/provenance_logs/",
    ) -> None:
        self._collection = mongo_collection
        self._logs_dir = Path(provenance_logs_dir)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        # In-memory event buffers keyed by session_id.  Cleared on finalize.
        self._buffers: Dict[str, List[dict]] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def open_session(
        self,
        session_id: str,
        image_path: str,
        run_mode: str = "hybrid",
        fallback_only: bool = False,
        user_id: Optional[str] = None,
    ) -> None:
        """Create the in-memory buffer and a skeleton MongoDB document."""
        image_hash = self._hash_file(image_path)

        self._buffers[session_id] = []

        doc = {
            "session_id": session_id,
            "image_path": image_path,
            "image_hash_sha256": image_hash,
            "user_id": user_id,
            "created_at": time.time(),
            "completed_at": None,
            "run_mode": run_mode,
            "fallback_only": fallback_only,
            "events": [],
            "summary": {},
        }

        if self._collection is not None:
            try:
                self._collection.update_one(
                    {"session_id": session_id},
                    {"$setOnInsert": doc},
                    upsert=True,
                )
            except Exception as exc:
                logger.warning("Provenance open_session MongoDB write failed: %s", exc)

        # Emit the initial pipeline_start event
        self.record(
            session_id,
            ProvenanceEventType.PIPELINE_START,
            "pipeline",
            {
                "image_path": image_path,
                "image_hash_sha256": image_hash,
                "run_mode": run_mode,
                "fallback_only": fallback_only,
            },
        )

    def finalize_session(
        self,
        session_id: str,
        phases_completed: List[str],
        total_time_ms: float,
        protections_applied: int,
        protected_image_path: Optional[str] = None,
        results_dir: Optional[str] = None,
    ) -> None:
        """
        Emit PIPELINE_COMPLETE, compute a summary, and flush all events to disk
        and MongoDB.

        Args:
            session_id:           Active session UUID.
            phases_completed:     List of stage names that ran successfully.
            total_time_ms:        Wall-clock time for the full pipeline.
            protections_applied:  Number of obfuscation transformations applied.
            protected_image_path: Path to the protected output image (may be None).
            results_dir:          If provided, also write a copy here for download.
        """
        protected_hash = self._hash_file(protected_image_path) if protected_image_path else None

        self.record(
            session_id,
            ProvenanceEventType.PIPELINE_COMPLETE,
            "pipeline",
            {
                "total_time_ms": total_time_ms,
                "phases_completed": phases_completed,
                "protections_applied": protections_applied,
                "protected_image_path": protected_image_path,
                "protected_hash_sha256": protected_hash,
            },
        )

        events = self._buffers.get(session_id, [])
        summary = self._compute_summary(
            events, phases_completed, total_time_ms, protections_applied
        )

        # Persist to MongoDB
        if self._collection is not None:
            try:
                self._collection.update_one(
                    {"session_id": session_id},
                    {"$set": {"summary": summary, "completed_at": time.time()}},
                )
            except Exception as exc:
                logger.warning("Provenance finalize_session MongoDB update failed: %s", exc)

        # Write canonical JSON log
        self._write_json(session_id, events, summary)

        # Optionally write a copy into the results dir for the download endpoint
        if results_dir:
            results_path = Path(results_dir) / f"{session_id}_provenance.json"
            self._write_json_to_path(results_path, events, summary, session_id)

        # Release the in-memory buffer
        self._buffers.pop(session_id, None)

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record(
        self,
        session_id: str,
        event_type: Any,
        phase: str,
        data: Dict[str, Any],
        detection_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Append a provenance event.  Fire-and-forget — never raises.

        Args:
            session_id:   Active session UUID.
            event_type:   A ProvenanceEventType member (or its string value).
            phase:        Pipeline phase name (e.g. "detection", "risk").
            data:         Arbitrary structured payload for this event.
            detection_id: Optional element ID to link the event to a detection.

        Returns:
            The event_id string, or None if recording failed.
        """
        try:
            event = {
                "event_id": str(uuid.uuid4()),
                "session_id": session_id,
                "timestamp": time.time(),
                "event_type": event_type.value if hasattr(event_type, "value") else str(event_type),
                "phase": phase,
                "detection_id": detection_id,
                "data": data,
            }

            # In-memory buffer (may not exist if open_session was not called)
            if session_id in self._buffers:
                self._buffers[session_id].append(event)

            # MongoDB $push — keep the events array in sync
            if self._collection is not None:
                try:
                    self._collection.update_one(
                        {"session_id": session_id},
                        {"$push": {"events": event}},
                    )
                except Exception as exc:
                    logger.warning("Provenance record MongoDB $push failed: %s", exc)

            return event["event_id"]

        except Exception as exc:
            logger.warning("Provenance record failed silently: %s", exc)
            return None

    def record_override(self, session_id: str, override_record: Any) -> None:
        """
        Convert a SafetyKernel OverrideRecord into a provenance event.

        The *override_record* is the object returned by
        ``SafetyKernel.validate_override()``.

        Args:
            session_id:      Active session UUID.
            override_record: An OverrideRecord (or duck-typed equivalent).
        """
        _action_map = {
            "allow": ProvenanceEventType.SAFETY_ALLOW,
            "block": ProvenanceEventType.SAFETY_BLOCK,
            "challenge": ProvenanceEventType.SAFETY_CHALLENGE,
        }

        action = override_record.safety_action
        action_str = action.value if hasattr(action, "value") else str(action)
        event_type = _action_map.get(action_str, ProvenanceEventType.SAFETY_ALLOW)

        beneficiary = override_record.beneficiary
        beneficiary_str = beneficiary.value if hasattr(beneficiary, "value") else str(beneficiary)

        self.record(
            session_id,
            event_type,
            "safety",
            {
                "override_record_id": override_record.record_id,
                "override_type": override_record.override_type,
                "element_type": override_record.element_type,
                "original_value": override_record.original_value,
                "requested_value": override_record.requested_value,
                "approved_value": override_record.approved_value,
                "rule_id": override_record.rule_id,
                "reason": override_record.reason,
                "beneficiary": beneficiary_str,
            },
            detection_id=override_record.detection_id,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_provenance(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Return the full provenance document for *session_id*.

        Tries MongoDB first; falls back to the on-disk JSON file.
        """
        if self._collection is not None:
            try:
                doc = self._collection.find_one(
                    {"session_id": session_id}, {"_id": 0}
                )
                if doc:
                    return doc
            except Exception as exc:
                logger.warning("Provenance get_provenance MongoDB query failed: %s", exc)

        json_path = self._logs_dir / f"{session_id}.json"
        if json_path.exists():
            try:
                with open(json_path) as fh:
                    return json.load(fh)
            except Exception as exc:
                logger.warning("Provenance get_provenance JSON read failed: %s", exc)

        return None

    def get_events_by_type(
        self, session_id: str, event_type: Any
    ) -> List[Dict[str, Any]]:
        """Return all events of *event_type* for *session_id*."""
        events = self._get_events(session_id)
        type_str = event_type.value if hasattr(event_type, "value") else str(event_type)
        return [e for e in events if e.get("event_type") == type_str]

    def get_events_by_detection(
        self, session_id: str, detection_id: str
    ) -> List[Dict[str, Any]]:
        """Return all events linked to *detection_id* for *session_id*."""
        events = self._get_events(session_id)
        return [e for e in events if e.get("detection_id") == detection_id]

    def get_hitl_records(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Return HITL-relevant events formatted for the frontend ProvenanceViewer.

        Includes override applications, approvals, rejections, and safety
        blocks/challenges.

        Returns a list of dicts with keys:
            timestamp, action, detection_id, old_value, new_value, reason.
        """
        events = self._get_events(session_id)
        _hitl_action_map = {
            "hitl_override_applied": "override",
            "hitl_approved": "approve",
            "hitl_rejected": "reject",
            "safety_block": "reject",
            "safety_challenge": "challenge",
        }
        records = []
        for event in events:
            action = _hitl_action_map.get(event.get("event_type", ""))
            if action is None:
                continue
            data = event.get("data", {})
            records.append({
                "timestamp": event.get("timestamp"),
                "action": action,
                "detection_id": event.get("detection_id") or data.get("detection_id", ""),
                "old_value": data.get("original_value", data.get("old_value", "")),
                "new_value": data.get("approved_value", data.get("new_value", "")),
                "reason": data.get("reason", ""),
            })
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_events(self, session_id: str) -> List[Dict[str, Any]]:
        """Return the event list for *session_id* from buffer or storage."""
        if session_id in self._buffers:
            return self._buffers[session_id]
        doc = self.get_provenance(session_id)
        return doc.get("events", []) if doc else []

    def _compute_summary(
        self,
        events: List[dict],
        phases_completed: List[str],
        total_time_ms: float,
        protections_applied: int,
    ) -> Dict[str, Any]:
        """Aggregate event type counts into a lightweight summary dict."""
        type_counts: Dict[str, int] = {}
        for event in events:
            t = event.get("event_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "total_events": len(events),
            "phases_completed": phases_completed,
            "total_time_ms": total_time_ms,
            "protections_applied": protections_applied,
            "overrides_count": type_counts.get("hitl_override_applied", 0),
            "safety_blocks": type_counts.get("safety_block", 0),
            "safety_challenges": type_counts.get("safety_challenge", 0),
            "hitl_pauses": type_counts.get("hitl_paused", 0),
        }

    def _write_json(
        self, session_id: str, events: List[dict], summary: Dict[str, Any]
    ) -> None:
        """Write the canonical provenance JSON file."""
        path = self._logs_dir / f"{session_id}.json"
        self._write_json_to_path(path, events, summary, session_id)

    def _write_json_to_path(
        self,
        path: Path,
        events: List[dict],
        summary: Dict[str, Any],
        session_id: str,
    ) -> None:
        """Serialise events + summary to *path* as pretty-printed JSON."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            doc = {
                "session_id": session_id,
                "events": events,
                "summary": summary,
            }
            with open(path, "w") as fh:
                json.dump(doc, fh, indent=2, default=str)
        except Exception as exc:
            logger.warning("Provenance JSON write failed (%s): %s", path, exc)

    @staticmethod
    def _hash_file(path: Optional[str]) -> Optional[str]:
        """Return the SHA-256 hex digest of *path*, or None on any failure."""
        if not path:
            return None
        try:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None
