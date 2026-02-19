"""
Agent 2.5: Adaptive Consent & Identity Learning Agent

Deterministic agent that:
1. Matches detected face embeddings against MongoDB database
2. Classifies faces: PRIMARY_SUBJECT (user), KNOWN_CONTACT, BYSTANDER
3. Determines consent status and adjusts risk severity
4. Tracks appearances and learns from user decisions

Pipeline: Detection -> Risk Assessment -> Consent Identity 
"""
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "utils"))

from utils.models import (
    DetectionResults,
    RiskAnalysisResult,
    RiskAssessment,
    RiskLevel,
    PrivacyProfile,
    PersonEntry,
    FaceEmbedding,
    ConsentHistory,
    IdentityMatch,
    PersonClassification,
    ConsentStatus
)
from utils.storage import FaceDatabase
from agents.tools import FaceDetectionTool

class ConsentIdentityAgent:
    """
    Agent 2.5: Adaptive Consent & Identity Learning Agent

    Deterministic agent that identifies faces via embedding matching
    against a MongoDB database and adjusts risk levels based on identity and consent status.
    """
    def __init__(self, config, privacy_profile: Optional[PrivacyProfile] = None, face_db: Optional[FaceDatabase] = None):
        self.config = config
        self.privacy_profile = privacy_profile or PrivacyProfile()

        # Thresholds from config
        self.similarity_threshold = config.get(
            "models.face_recognition.similarity_threshold", 0.85
        )
        self.confident_threshold = config.get(
            "models.face_recognition.confident_threshold", 0.90
        )
        self.max_embeddings = config.get(
            "storage.max_embeddings_per_person", 5
        )

        # Learning settings
        self.learning_enabled = config.get("learning.enabled", True)
        self.min_appearances_for_trust = config.get(
            "learning.min_appearances_for_trust", 3
        )
        self.trust_approval_threshold = config.get(
            "learning.trust_approval_threshold", 0.8
        )
        self.risk_decay_per_approval = config.get(
            "learning.risk_decay_per_approval", 0.1
        )
        self.risk_increase_per_protection = config.get(
            "learning.risk_increase_per_protection", 0.05
        )

        # Initialize face database (MongoDB)
        if face_db is not None:
            self.face_db = face_db
        else:
            self.face_db = FaceDatabase(
                mongo_uri = config.get(
                    "storage.mongo_uri", "mongodb://localhost:27017/"
                ),
                database_name = config.get(
                    "storage.database_name", "privacy_guard"
                ),
                encryption_key_path = config.get(
                    "storage.encryption_key_path", "data/face_db/.encryption_key"
                ),
                encryption_enabled = config.get(
                    "storage.encryption_enabled", True
                )
            )
        
        # In-memory embedding cache for fast matching
        self._embedding_cache: List[Tuple[str, np.ndarray]] = []
        self._cache_dirty = True
        self._refresh_cache()

        # Face detection tool (only for registration)
        self._face_tool = None

        print(f"\n[ConsentIdentityAgent] Initialized")
        print(f"  Similarity threshold: {self.similarity_threshold}")
        print(f"  Confident threshold: {self.confident_threshold}")
        print(f"  Learning enabled: {self.learning_enabled}")
        print(f"  Cached embeddings: {len(self._embedding_cache)}")

    def _get_face_tool(self) -> FaceDetectionTool:
        """Lazy-init"""
        if self._face_tool is None:
            self._face_tool = FaceDetectionTool(self.config)
        return self._face_tool
    
    def _refresh_cache(self):
        """Reload all embeddings from MongoDB into memory cache"""
        raw = self.face_db.get_all_embeddings()
        self._embedding_cache = [
            (pid, np.array(emb, dtype = np.float32))
            for pid, emb in raw
        ]
        self._cache_dirty = False
    
    # Core Pipeline
    def run(self, detections: DetectionResults, risk_result: RiskAnalysisResult) -> RiskAnalysisResult:
        """
        Main consent identity pipeline.

        For each face in detections:
        1. Extract embedding from face.attributes["embedding"]
        2. Match against MongoDB embeddings (consine similarity)
        3. Classify: PRIMARY_SUBJECT / KNOWN_CONTACT / BYSTANDER
        4. Determine consent status
        5. Adjust risk severity on the matching RiskAssessment

        Args:
            detections: Detection results (contains face embeddings)
            risk_result: Risk analysis result from Agent 2
        
        Returns:
            Updated RiskAnalysisResult with identity-adjusted risks
        """
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Consent Identity Agent - Starting")
        print(f"{'='*60}")

        if self._cache_dirty:
            self._refresh_cache()

        face_count = len(detections.faces)
        if face_count == 0:
            print("No faces to process")
            self._print_summary(risk_result, start_time)
            return risk_result
        
        print(f"  Faces to identify: {face_count}")
        print(f"  Database embeddings: {len(self._embedding_cache)}")

        for face in detections.faces:
            face_id = face.id
            embedding = face.attributes.get('embedding')

            if embedding is None:
                print(f"  [{face_id[:8]}] No embedding, skipping")
                continue

            query = np.array(embedding, dtype = np.float32)

            # Match against database
            match = self._match_face(query, face_id)

            # Find and update the corresonding risk assessment
            for assessment in risk_result.risk_assessments:
                if assessment.detection_id == face_id and assessment.element_type == "face":
                    self._apply_identity(assessment, match)
                    break
            
            # Track appearance and optionally add new embedding
            if match.person_id:
                self._track_appearance(match, query)
            
            print(
                f"  [{face_id[:8]}] → {match.classification.value} "
                f"({match.match_type}, sim={match.recognition_confidence:.3f})"
                f" → consent={match.consent_status.value}"
            )
        
        # Update aggregate counters
        risk_result.faces_pending_identity = sum(
            1 for a in risk_result.risk_assessments 
            if a.element_type == "face" and (
                a.consent_status is None 
                or a.consent_status in (ConsentStatus.NONE, ConsentStatus.UNCLEAR)
            )
        )
        risk_result.confirmed_risks = sum(1 for a in risk_result.risk_assessments if a.requires_protection)
        
        # Recalculating overall risk
        risk_result.overall_risk_level = self._calculate_overall_risk(
            risk_result.risk_assessments
        )

        self._print_summary(risk_result, start_time)
        return risk_result
    
    def _match_face(self, query: np.ndarray, detection_id: str) -> IdentityMatch:
        """
        Match a face embedding against all stored embeddings.

        1. Compute cosine similarity against all cached embeddings
        2. Group by person_id, take max similarity per person
        3. Classify based on thresholds:
            >= confident_threshold (0.90) -> confident
            >= similarity_threshold (0.85) -> probable
            < similarity_threshold -> no_match
        """
        if len(self._embedding_cache) == 0:
            return self._no_match(detection_id)
    
        # Normalize_query
        query_norm = query / (np.linalg.norm(query) + 1e-8)

        # Compute similarities, group by person
        person_scores: Dict[str, float] = {}
        for person_id, stored_emb in self._embedding_cache:
            stored_norm = stored_emb / (np.linalg.norm(stored_emb) + 1e-8)
            sim = float(np.dot(query_norm, stored_norm))
            if person_id not in person_scores or sim > person_scores[person_id]:
                person_scores[person_id] = sim
        
        if not person_scores:
            return self._no_match(detection_id)

        # Best Match
        best_pid = max(person_scores, key = person_scores.get)
        best_sim = person_scores[best_pid]

        if best_sim >= self.confident_threshold:
            match_type = "confident"
        elif best_sim >= self.similarity_threshold:
            match_type = "probable"
        else:
            return self._no_match(detection_id)

        # Get person info
        person = self.face_db.get_person(best_pid)
        if person is None:
            return self._no_match(detection_id)
        
        classification = self._classify_person(person)
        consent_status = self._determine_consent(person, classification)
        risk_adjustment = self._get_risk_adjustment(
            classification, consent_status, person
        )

        return IdentityMatch(
            detection_id = detection_id,
            person_id = person.person_id,
            person_label = person.label,
            classification = classification,
            consent_status = consent_status,
            consent_confidence = best_sim,
            recognition_confidence = best_sim,
            match_type = match_type,
            history = person.consent_history,
            ethical_risk = (
                "none" if classification == PersonClassification.PRIMARY_SUBJECT
                else "low"
            ),
            risk_adjustment = risk_adjustment
        )

    def _no_match(self, detection_id: str) -> IdentityMatch:
        """IdentityMatch for unrecognized faces (bystanders)"""
        return IdentityMatch(
            detection_id=detection_id,
            person_id=None,
            person_label=None,
            classification=PersonClassification.BYSTANDER,
            consent_status=ConsentStatus.NONE,
            consent_confidence=0.0,
            recognition_confidence=0.0,
            match_type="no_match",
            history=None,
            ethical_risk="high",
            risk_adjustment="severity_unchanged",
          )


    

