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
from utils.config import get_risk_color
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
    
    # Face Matching
    
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
    
    def _classify_person(self, person: PersonEntry) -> PersonClassification:
        """Classify based on stored relationship."""
        if person.relationship == "self":
            return PersonClassification.PRIMARY_SUBJECT
        elif person.relationship in ("family", "friend", "colleague"):
            return PersonClassification.KNOWN_CONTACT
        else:
            return PersonClassification.BYSTANDER
    
    def _determine_consent(self, person: PersonEntry, classification: PersonClassification) -> ConsentStatus:
        """Determine consent from identity and approval history"""
        if classification == PersonClassification.PRIMARY_SUBJECT:
            return ConsentStatus.EXPLICIT
        
        if classification == PersonClassification.KNOWN_CONTACT:  
            h = person.consent_history
            trusted = (
                h.times_appeared >= self.min_appearances_for_trust
                and h.approval_rate >= self.trust_approval_threshold
            )
            return ConsentStatus.ASSUMED if trusted else ConsentStatus.UNCLEAR
        
        return ConsentStatus.NONE
    
    def _get_risk_adjustment(self, classification: PersonClassification, consent_status: ConsentStatus, person: PersonEntry) -> str:
        """Compute risk adjustment string."""
        if classification == PersonClassification.PRIMARY_SUBJECT:
            return "severity_reduced_to_low"
        
        if classification == PersonClassification.KNOWN_CONTACT:
            if consent_status == ConsentStatus.ASSUMED:
                key = {
                    "family": "family_faces",
                    "friend": "friend_faces",
                    "colleague": "friend_faces"
                }.get(person.relationship, "bystander_faces")
                sensitivity = self.privacy_profile.identity_sensitivity.get(
                    key, "medium"
                )
                return f"severity_reduced_to_{sensitivity}"
            return "severity_reduced_to_high"
        return "severity_unchanged"
    
    # Risk Application
    
    def _apply_identity(self, assessment: RiskAssessment, match: IdentityMatch):
        """Apply identity match to a risk assessment in-place."""
        assessment.person_id = match.person_id
        assessment.person_label = match.person_label
        assessment.classification = match.classification
        assessment.consent_status = match.consent_status
        assessment.consent_confidence = match.consent_confidence

        if match.classification == PersonClassification.PRIMARY_SUBJECT:
            assessment.severity = RiskLevel.LOW
            assessment.color_code = get_risk_color(self.config, "low")
            assessment.requires_protection = False
            assessment.reasoning = (
                f"Identified as user ({match.person_label})"
            )

        elif match.classification == PersonClassification.KNOWN_CONTACT:
            if match.consent_status == ConsentStatus.ASSUMED:
                # Parse target severity from risk_adjustment
                target = match.risk_adjustment.replace(
                    "severity_reduced_to_", ""
                )
                sev_map = {
                    "critical": RiskLevel.CRITICAL,
                    "high": RiskLevel.HIGH,
                    "medium": RiskLevel.MEDIUM,
                    "low": RiskLevel.LOW
                }
                assessment.severity = sev_map.get(target, RiskLevel.MEDIUM)
                assessment.color_code = get_risk_color(
                    self.config, assessment.severity.value
                )
                assessment.requires_protection = assessment.severity in (RiskLevel.CRITICAL, RiskLevel.HIGH)
                rate = match.history.approval_rate if match.history else 0
                assessment.reasoning = (
                    f"Known contact ({match.person_label}), "
                    f"trusted (approval rate: {rate:.0%})"
                )
            else:
                assessment.severity = RiskLevel.HIGH
                assessment.color_code = get_risk_color(self.config, "high")
                assessment.requires_protection = True
                assessment.reasoning = (
                    f"Known contact ({match.person_label}), "
                    f"not yet trusted"
                )

        # BYSTANDER: no changes - severity stays CRITICAL from Phase1

    # Appearance Tracking

    def _track_appearance(self, match: IdentityMatch, query_embedding: np.ndarray):
        """Update appearance count and optionally add diverse embedding."""
        person = self.face_db.get_person(match.person_id)
        if person is None:
            return
        
        person.consent_history.times_appeared += 1
        person.last_seen = datetime.now()
        self.face_db.update_person(person)
        self._cache_dirty = True
        self._refresh_cache()

        # Add embedding if diverse enough and under max limit
        if len(person.embeddings) < self.max_embeddings:
            q_norm = query_embedding / (
                np.linalg.norm(query_embedding) + 1e-8
            )
            is_diverse = True
            for existing in person.embeddings:
                e_arr = np.array(existing.embedding, dtype = np.float32)
                e_norm = e_arr / (np.linalg.norm(e_arr) + 1e-8)
                if float(np.dot(q_norm, e_norm)) >= 0.95:
                    is_diverse = False
                    break
            
            if is_diverse:
                self.face_db.add_embedding_to_person(
                    person.person_id,
                    query_embedding.tolist(),
                    source_image="pipeline_detection"
                )
                self._cache_dirty = True
                self._refresh_cache()

    # Registration

    def register_user_face(self, image_path: str, label: str = "Me") -> Optional[str]:
        """
        Register the user's own face.

        Detects face in image, extracts embeddings, stores in MongoDB with relationship = "self".

        Returns:
            person_id if successful, None otherwise
        """
        return self._register_face(image_path, label, relationship="self")
    
    def register_contact(self, image_path: str, label: str, relationship: str = "friend") -> Optional[str]:
        """
        Register a known contact's face.

        Args:
            image_path: Path to image containing contact's face
            label: Display label (e.g.,"Mom", "John", etc)
            relationship: "family", "friend", or "colleague"
        
        Returns:
            person_id if successful, None otherwise
        """
        return self._register_face(image_path, label, relationship)
    
    def _register_face(self, image_path: str, label: str, relationship: str) -> Optional[str]:
        """Core registration: detect face -> extract embedding -> store."""
        face_tool = self._get_face_tool()
        result = face_tool._run(image_path)
        data = json.loads(result)

        if "error" in data or not data.get("faces"):
            print(f"Registration failed: no face in {image_path}")
            return None
        
        face_data = data["faces"][0]
        embedding = face_data.get("embedding")
        if embedding is None:
            print(f"Registration failed: no embedding extracted")
            return None
        
        person = PersonEntry(
            label = label,
            relationship = relationship,
            embeddings = [FaceEmbedding(embedding = embedding, source_image = image_path)],
            consent_history = ConsentHistory(
                times_appeared = 1 if relationship == "self" else 0,
                times_approved = 1 if relationship == "self" else 0
            ),
            risk_decay_factor = 0.0 if relationship == "self" else 1.0
        )
        success = self.face_db.add_person(person)
        if success:
            self._cache_dirty = True
            self._refresh_cache()
            print(
                f"Registered: {label} ({relationship})"
                f" → {person.person_id}"
            )
            return person.person_id
        return None
    
    # Adaptive Learning

    def record_user_decision(self, person_id: str, decision: str, context: str = "user_review"): 
        """
        Record user's decision for adaptive learning

        Args:
            person_id: Person's unique decision for adaptive learning.
            decision: "approved" or "protected"
            context: e.g., "user_review", "social_media"
        """
        if not self.learning_enabled:
            return
        
        person = self.face_db.get_person(person_id)
        if person is None:
            print(f"Person {person_id} not found")
            return
        
        h = person.consent_history
        if decision == "approved":
            h.times_approved += 1
            h.last_consent_decision = "approved" 
            person.risk_decay_factor = max(
                0.0, person.risk_decay_factor - self.risk_decay_per_approval
            )
        
        elif decision == "protected":
            h.times_protected += 1
            h.last_consent_decision = "protected"
            person.risk_decay_factor = min(
                2.0,
                person.risk_decay_factor + self.risk_increase_per_protection,
            )
        
        if h.times_appeared > 0:
            h.consent_confidence = h.approval_rate
        if context not in h.contexts:
            h.contexts.append(context)
        
        self.face_db.update_person(person)
        print(
            f"Decision recorded for {person.label}: {decision} "
            f"(approval rate: {h.approval_rate:.0%})"
        )
    
    def _calculate_overall_risk(self, assessments: List[RiskAssessment]) -> RiskLevel:
        """Recalculate overall risk after identity adjustments."""
        if not assessments:
            return RiskLevel.LOW
        
        critical = sum(1 for a in assessments if a.severity == RiskLevel.CRITICAL)
        high = sum(1 for a in assessments if a.severity == RiskLevel.HIGH)
        medium = sum(1 for a in assessments if a.severity == RiskLevel.MEDIUM)
        if critical > 0:
            return RiskLevel.CRITICAL
        elif high >= 2:
            return RiskLevel.CRITICAL
        elif high >= 1:
            return RiskLevel.HIGH
        elif medium >= 3:
            return RiskLevel.HIGH
        elif medium >= 1:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
    
    def _print_summary(self, result: RiskAnalysisResult, start_time: float):
        """Print identity resolution summary."""
        elapsed = (time.time() - start_time) * 1000
        print(f"\n{'='*60}")
        print(f"Consent Identity Complete")
        print(f"{'='*60}")
        print(f"  Processing time: {elapsed:.2f}ms")
        print(f"  Overall risk: {result.overall_risk_level.value.upper()}")
        print(f"  Faces pending identity: {result.faces_pending_identity}")
        print(f"  Confirmed risks: {result.confirmed_risks}")
        for a in result.risk_assessments:
            if a.element_type == "face":
                label = a.person_label or "Unknown"
                cls = (
                    a.classification.value
                    if a.classification
                    else "bystander"
                )
                consent = (
                    a.consent_status.value
                    if hasattr(a.consent_status, "value")
                    else a.consent_status
                )
                print(
                    f"    Face [{a.detection_id[:8]}]: {label} → "
                    f"{cls}, consent={consent}, "
                    f"severity={a.severity.value}"
                )
        print(f"{'='*60}\n")

    def close(self):
        """Close database connection."""
        if self.face_db:
            self.face_db.close()


        
        
    

            
                









        


