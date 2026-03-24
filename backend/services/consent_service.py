"""
ConsentService — REST-API wrapper around the existing FaceDatabase.

Responsibilities:
  - Expose person/consent CRUD to the FastAPI routers.
  - NEVER return raw FaceNet embedding vectors (512-float arrays) to callers.
  - Accept image file paths or raw bytes, run MTCNN + FaceNet internally,
    and store via FaceDatabase.add_person() / add_embedding_to_person().
  - Handle MongoDB connection failures gracefully (returns error dict rather
    than raising unhandled exceptions into the router layer).

Import path note:
  This file lives at backend/services/consent_service.py.
  The project root (one level above backend/) is injected into sys.path so
  that `from utils.storage import FaceDatabase` resolves correctly.
"""

from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------------
# Project-root path injection so utils.* imports resolve regardless
# of where the FastAPI process is launched from.
# ------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.models import ConsentHistory, FaceEmbedding, PersonEntry

logger = logging.getLogger(__name__)

# Safe fields that may be returned to API clients.
_SAFE_PERSON_FIELDS = [
    "person_id",
    "label",
    "relationship",
    "consent_status",
    "embedding_count",
    "consent_history",
    "first_seen",
    "last_seen",
    "notes",
]


class ConsentService:
    """
    Service layer for consent / face-identity management.

    ``initialize()`` must be called once before any other method — typically
    inside the FastAPI lifespan startup handler.  All public methods are
    ``async`` to match FastAPI's async routing, but the underlying MongoDB
    driver calls are synchronous (pymongo); this is acceptable for the
    current deployment profile (single-machine, low concurrency).
    """

    def __init__(self) -> None:
        self._db: Optional[Any] = None   # FaceDatabase (imported lazily)
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, mongo_url: str = "mongodb://localhost:27017/",
                   database_name: str = "privacy_guard",
                   encryption_key_path: str = "data/face_db/.encryption_key",
                   encryption_enabled: bool = True) -> None:
        """
        Connect to MongoDB and initialise the FaceDatabase.

        Parameters are forwarded directly to FaceDatabase.__init__.
        Raises ``RuntimeError`` if the connection cannot be established so
        that the FastAPI lifespan startup fails loudly rather than silently.
        """
        try:
            from utils.storage import FaceDatabase  # lazy import; heavy deps
            self._db = FaceDatabase(
                mongo_uri=mongo_url,
                database_name=database_name,
                encryption_key_path=encryption_key_path,
                encryption_enabled=encryption_enabled,
            )
            self._connected = True
            logger.info("ConsentService: FaceDatabase connected (%s / %s)", mongo_url, database_name)
        except Exception as exc:
            self._connected = False
            logger.error("ConsentService: failed to connect to MongoDB: %s", exc)
            raise RuntimeError(f"ConsentService initialisation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------

    async def get_persons(self, limit: int = 50, offset: int = 0) -> dict:
        """
        List persons with basic consent metadata.

        Returns a dict with keys ``total`` and ``persons``.
        Embeddings are stripped; only safe fields are included.
        """
        if not self._connected or self._db is None:
            return {"total": 0, "persons": [], "error": "Database not connected"}

        try:
            all_persons: List[PersonEntry] = self._db.get_all_persons() or []
            total = len(all_persons)

            # Manual pagination (FaceDatabase.get_all_persons returns full list)
            page = all_persons[offset: offset + limit]
            redacted = [self._redact_person_entry(p) for p in page]
            return {"total": total, "persons": redacted}

        except Exception as exc:
            logger.error("get_persons failed: %s", exc)
            return {"total": 0, "persons": [], "error": str(exc)}

    async def register_person(
        self,
        label: str,
        relationship: str,
        consent_status: str,
        image_paths: List[str],
        notes: Optional[str] = None,
    ) -> dict:
        """
        Register a new person in the consent database.

        For each path in *image_paths* the method runs MTCNN face detection
        followed by FaceNet 512-D embedding extraction and stores the result
        via ``FaceDatabase.add_person()``.

        Returns the redacted person document (no raw embeddings) or an error
        dict if registration fails.
        """
        if not self._connected or self._db is None:
            return {"error": "Database not connected"}

        try:
            embeddings = self._extract_embeddings_from_paths(image_paths)

            if not embeddings:
                return {
                    "error": "No face detected in the provided image(s). "
                             "Ensure each image contains exactly one clearly visible face."
                }

            person = PersonEntry(
                person_id=str(uuid.uuid4()),
                label=label,
                relationship=relationship,
                embeddings=embeddings,
                consent_history=ConsentHistory(),
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                notes=notes,
            )

            success = self._db.add_person(person)
            if not success:
                return {"error": f"Person with label '{label}' could not be stored (possible duplicate ID)."}

            return self._redact_person_entry(person)

        except Exception as exc:
            logger.error("register_person failed for label=%s: %s", label, exc)
            return {"error": str(exc)}

    async def update_person(self, person_id: str, updates: Dict[str, Any]) -> dict:
        """
        Update consent level or metadata for an existing person.

        Supported *updates* keys: ``label``, ``relationship``, ``consent_status``,
        ``notes``.  Embedding changes require re-registration.

        Returns the updated redacted document, or an error dict.
        """
        if not self._connected or self._db is None:
            return {"error": "Database not connected"}

        try:
            person: Optional[PersonEntry] = self._db.get_person(person_id)
            if person is None:
                return {"error": f"Person '{person_id}' not found."}

            # Apply supported field updates
            if "label" in updates:
                person.label = updates["label"]
            if "relationship" in updates:
                person.relationship = updates["relationship"]
            if "notes" in updates:
                person.notes = updates["notes"]
            if "consent_status" in updates:
                # Persist a human-readable status in notes (ConsentHistory tracks
                # approval rates; consent_status is derived from relationship + history)
                existing_notes = person.notes or ""
                tag = f"[consent_status={updates['consent_status']}]"
                if tag not in existing_notes:
                    person.notes = f"{existing_notes} {tag}".strip()

            person.last_seen = datetime.now()
            success = self._db.update_person(person)
            if not success:
                return {"error": "Update failed; see server logs for details."}

            return self._redact_person_entry(person)

        except Exception as exc:
            logger.error("update_person failed for person_id=%s: %s", person_id, exc)
            return {"error": str(exc)}

    async def enroll_face(
        self,
        user_id: str,
        image_bytes: bytes,
        label: str,
        relationship: str = "self",
    ) -> dict:
        """
        Enroll a single face from raw image bytes.

        Uses MTCNN for face detection and FaceNet for 512-D embedding
        extraction, then stores the result via FaceDatabase.

        If a person entry already exists for ``user_id`` (matched via notes
        tag ``[user_id=...]``), an additional embedding is appended to that
        person.  Otherwise a new PersonEntry is created.

        Parameters
        ----------
        user_id:      Caller's session/user identifier (used as lookup key).
        image_bytes:  Raw bytes of the uploaded image.
        label:        Human-readable name for the face entry.
        relationship: Relationship type (e.g. "self", "family", "friend").

        Returns
        -------
        Redacted person document dict on success, or error dict on failure.
        """
        if not self._connected or self._db is None:
            return {"error": "Database not connected"}

        try:
            import io as _io

            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1
            from PIL import Image

            device = torch.device("cpu")
            mtcnn = MTCNN(keep_all=False, device=device, post_process=True)
            facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

            img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
            face_tensor = mtcnn(img)

            if face_tensor is None:
                return {
                    "error": (
                        "No face detected in the provided image. "
                        "Ensure the image contains exactly one clearly visible face."
                    )
                }

            with torch.no_grad():
                embedding_tensor = facenet(face_tensor.unsqueeze(0).to(device))

            embedding_vec: List[float] = embedding_tensor.squeeze(0).cpu().numpy().tolist()

            new_embedding = FaceEmbedding(
                embedding=embedding_vec,
                source_image=f"upload:{user_id}",
                timestamp=datetime.now(),
            )

            # Try to find an existing person entry for this user
            all_persons = self._db.get_all_persons() or []
            existing_person: Optional[PersonEntry] = None
            for p in all_persons:
                if p.notes and f"[user_id={user_id}]" in (p.notes or ""):
                    existing_person = p
                    break

            if existing_person is not None:
                existing_person.embeddings.append(new_embedding)
                existing_person.last_seen = datetime.now()
                self._db.update_person(existing_person)
                return self._redact_person_entry(existing_person)

            # Create a new person entry
            person = PersonEntry(
                person_id=str(uuid.uuid4()),
                label=label,
                relationship=relationship,
                embeddings=[new_embedding],
                consent_history=ConsentHistory(),
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                notes=f"[user_id={user_id}]",
            )
            success = self._db.add_person(person)
            if not success:
                return {"error": "Could not store face embedding — possible duplicate."}

            return self._redact_person_entry(person)

        except Exception as exc:
            logger.error("enroll_face failed for user_id=%s: %s", user_id, exc)
            return {"error": str(exc)}

    async def delete_person(self, person_id: str) -> bool:
        """
        Delete a person and all associated face embeddings.

        Returns ``True`` on success, ``False`` if the person was not found or
        the delete operation failed.
        """
        if not self._connected or self._db is None:
            logger.warning("delete_person called but database is not connected.")
            return False

        try:
            existing = self._db.get_person(person_id)
            if existing is None:
                logger.info("delete_person: person_id=%s not found.", person_id)
                return False

            return self._db.delete_person(person_id)

        except Exception as exc:
            logger.error("delete_person failed for person_id=%s: %s", person_id, exc)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _redact_person(self, person_doc: dict) -> dict:
        """
        Strip embeddings and internal MongoDB fields from a raw person
        document (dict representation, as returned from MongoDB cursor).

        Only fields listed in ``_SAFE_PERSON_FIELDS`` are forwarded.
        """
        return {k: person_doc.get(k) for k in _SAFE_PERSON_FIELDS if k in person_doc}

    def _redact_person_entry(self, person: PersonEntry) -> dict:
        """
        Build a safe dict from a ``PersonEntry`` Pydantic model.

        Importantly, ``person.embeddings`` (the 512-float vectors) are never
        included; instead only the *count* is exposed so the client can
        display how many face samples are registered.
        """
        consent_hist = person.consent_history
        return {
            "person_id": person.person_id,
            "label": person.label,
            "relationship": person.relationship,
            "embedding_count": len(person.embeddings),
            "consent_history": {
                "times_appeared": consent_hist.times_appeared,
                "times_approved": consent_hist.times_approved,
                "times_protected": consent_hist.times_protected,
                "approval_rate": consent_hist.approval_rate,
                "protection_rate": consent_hist.protection_rate,
                "last_consent_decision": consent_hist.last_consent_decision,
                "consent_confidence": consent_hist.consent_confidence,
            },
            "first_seen": person.first_seen.isoformat() if person.first_seen else None,
            "last_seen": person.last_seen.isoformat() if person.last_seen else None,
            "notes": person.notes,
        }

    def _extract_embeddings_from_paths(self, image_paths: List[str]) -> List[FaceEmbedding]:
        """
        Run MTCNN face detection and FaceNet embedding extraction on each
        image path.

        Skips images where no face is detected (logs a warning).  Returns the
        list of successfully extracted ``FaceEmbedding`` objects.

        The MTCNN and InceptionResnetV1 (FaceNet) models are imported lazily
        here so that importing ``consent_service`` does not force heavy ML
        dependency loading at module import time.
        """
        try:
            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1
            from PIL import Image
        except ImportError as exc:
            logger.error("Required ML dependencies not available: %s", exc)
            raise RuntimeError(
                "facenet_pytorch and/or torch are not installed. "
                "Cannot extract face embeddings."
            ) from exc

        device = torch.device("cpu")
        mtcnn = MTCNN(keep_all=False, device=device, post_process=True)
        facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

        embeddings: List[FaceEmbedding] = []

        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                face_tensor = mtcnn(img)  # Returns aligned face tensor or None

                if face_tensor is None:
                    logger.warning("No face detected in image: %s", path)
                    continue

                # face_tensor shape: [3, 160, 160]
                with torch.no_grad():
                    embedding_tensor = facenet(face_tensor.unsqueeze(0).to(device))

                embedding_vec: List[float] = embedding_tensor.squeeze(0).cpu().numpy().tolist()

                embeddings.append(
                    FaceEmbedding(
                        embedding=embedding_vec,
                        source_image=path,
                        timestamp=datetime.now(),
                    )
                )
                logger.debug("Extracted embedding from %s (dim=%d)", path, len(embedding_vec))

            except Exception as exc:
                logger.warning("Failed to process image %s: %s", path, exc)
                continue

        return embeddings
