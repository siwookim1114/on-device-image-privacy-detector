"""
Test script for Consent Identity Agent (Agent 2.5)

Tests face matching, identity classification, consent determination,
risk adjustment, and adaptive learning against MongoDB.

Requires: MongoDB running locally (mongod)
"""

import sys
import json
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "utils"))

from utils.config import load_config
from utils.models import (
    DetectionResults,
    FaceDetection,
    BoundingBox,
    RiskAnalysisResult,
    RiskAssessment,
    RiskLevel,
    RiskType,
    PrivacyProfile,
    PersonEntry,
    FaceEmbedding,
    ConsentHistory,
    PersonClassification,
    ConsentStatus,
)
from utils.storage import FaceDatabase
from agents.consent_identity_agent import ConsentIdentityAgent

# Test database name (separate from production)
TEST_DB_NAME = "privacy_guard_test_consent"


def make_embedding(seed: int = 0) -> list:
    """Create a deterministic 512-dim L2-normalized embedding."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb.tolist()


def make_similar_embedding(base: list, noise_scale: float = 0.05, seed: int = 99) -> list:
    """Create an embedding similar to base (high cosine similarity)."""
    rng = np.random.RandomState(seed)
    base_arr = np.array(base, dtype=np.float32)
    noise = rng.randn(512).astype(np.float32) * noise_scale
    similar = base_arr + noise
    similar /= np.linalg.norm(similar)
    return similar.tolist()


def make_face_detection(face_id: str, embedding: list) -> FaceDetection:
    """Create a FaceDetection with an embedding in attributes."""
    return FaceDetection(
        id=face_id,
        bbox=BoundingBox(x=100, y=100, width=200, height=200),
        confidence=0.98,
        size="medium",
        clarity="high",
        attributes={
            "has_landmarks": True,
            "has_embedding": True,
            "embedding": embedding,
        },
    )


def make_risk_assessment(face_id: str) -> RiskAssessment:
    """Create a default CRITICAL face risk assessment (Phase 1 output)."""
    return RiskAssessment(
        detection_id=face_id,
        element_type="face",
        element_description="Face (medium, high clarity)",
        risk_type=RiskType.IDENTITY_EXPOSURE,
        severity=RiskLevel.CRITICAL,
        color_code="#FF0000",
        reasoning="Tool-based assessment",
        user_sensitivity_applied="bystander_faces",
        bbox=BoundingBox(x=100, y=100, width=200, height=200),
        requires_protection=True,
        consent_status=ConsentStatus.NONE,
        consent_confidence=0.0,
    )


def cleanup_db(face_db: FaceDatabase):
    """Drop the test database."""
    try:
        face_db.client.drop_database(TEST_DB_NAME)
    except Exception:
        pass


def run_tests():
    """Run all consent identity agent tests."""
    print("=" * 60)
    print("Testing Consent Identity Agent (Agent 2.5)")
    print("=" * 60 + "\n")

    # Load config
    config = load_config()
    passed = 0
    failed = 0

    # Create test database
    face_db = FaceDatabase(
        mongo_uri="mongodb://localhost:27017/",
        database_name=TEST_DB_NAME,
        encryption_key_path=config.get(
            "storage.encryption_key_path", "data/face_db/.encryption_key"
        ),
        encryption_enabled=True,
    )

    try:
        # ---- Test 1: Empty database — all faces stay BYSTANDER/CRITICAL ----
        print("\nTest 1: Empty database → BYSTANDER/CRITICAL")
        print("-" * 40)
        try:
            cleanup_db(face_db)
            face_db = FaceDatabase(
                mongo_uri="mongodb://localhost:27017/",
                database_name=TEST_DB_NAME,
                encryption_key_path=config.get(
                    "storage.encryption_key_path",
                    "data/face_db/.encryption_key",
                ),
                encryption_enabled=True,
            )

            agent = ConsentIdentityAgent(
                config, privacy_profile=PrivacyProfile(), face_db=face_db
            )

            emb = make_embedding(seed=42)
            face = make_face_detection("face_test_1", emb)
            detections = DetectionResults(
                image_path="test.jpg", faces=[face]
            )
            risk_result = RiskAnalysisResult(
                image_path="test.jpg",
                risk_assessments=[make_risk_assessment("face_test_1")],
                overall_risk_level=RiskLevel.CRITICAL,
            )

            result = agent.run(detections, risk_result)

            a = result.risk_assessments[0]
            assert a.severity == RiskLevel.CRITICAL, (
                f"Expected CRITICAL, got {a.severity}"
            )
            assert a.consent_status in (
                ConsentStatus.NONE, "none"
            ), f"Expected NONE consent, got {a.consent_status}"
            assert a.classification in (
                PersonClassification.BYSTANDER, "bystander"
            ), f"Expected BYSTANDER, got {a.classification}"
            assert result.faces_pending_identity == 1

            print("  PASSED: Unknown face stays BYSTANDER/CRITICAL\n")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

        # ---- Test 2: Register user face → PRIMARY_SUBJECT/EXPLICIT/LOW ----
        print("\nTest 2: User face → PRIMARY_SUBJECT/EXPLICIT/LOW")
        print("-" * 40)
        try:
            cleanup_db(face_db)
            face_db = FaceDatabase(
                mongo_uri="mongodb://localhost:27017/",
                database_name=TEST_DB_NAME,
                encryption_key_path=config.get(
                    "storage.encryption_key_path",
                    "data/face_db/.encryption_key",
                ),
                encryption_enabled=True,
            )

            agent = ConsentIdentityAgent(
                config, privacy_profile=PrivacyProfile(), face_db=face_db
            )

            # Register user's face embedding directly
            user_emb = make_embedding(seed=10)
            person = PersonEntry(
                label="Me",
                relationship="self",
                embeddings=[FaceEmbedding(embedding=user_emb)],
                consent_history=ConsentHistory(
                    times_appeared=1, times_approved=1
                ),
                risk_decay_factor=0.0,
            )
            face_db.add_person(person)
            agent._cache_dirty = True

            # Create detection with similar embedding (simulates same face)
            query_emb = make_similar_embedding(
                user_emb, noise_scale=0.02, seed=11
            )
            face = make_face_detection("face_user", query_emb)
            detections = DetectionResults(
                image_path="test.jpg", faces=[face]
            )
            risk_result = RiskAnalysisResult(
                image_path="test.jpg",
                risk_assessments=[make_risk_assessment("face_user")],
                overall_risk_level=RiskLevel.CRITICAL,
            )

            result = agent.run(detections, risk_result)

            a = result.risk_assessments[0]
            assert a.severity == RiskLevel.LOW, (
                f"Expected LOW, got {a.severity}"
            )
            assert a.consent_status in (
                ConsentStatus.EXPLICIT, "explicit"
            ), f"Expected EXPLICIT, got {a.consent_status}"
            assert a.classification in (
                PersonClassification.PRIMARY_SUBJECT, "primary_subject"
            ), f"Expected PRIMARY_SUBJECT, got {a.classification}"
            assert a.person_label == "Me"
            assert a.requires_protection is False
            assert result.faces_pending_identity == 0
            assert result.overall_risk_level == RiskLevel.LOW

            print("  PASSED: User face → LOW/EXPLICIT/PRIMARY_SUBJECT\n")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

        # ---- Test 3: Contact (untrusted → trusted after approvals) ----
        print("\nTest 3: Contact trust progression")
        print("-" * 40)
        try:
            cleanup_db(face_db)
            face_db = FaceDatabase(
                mongo_uri="mongodb://localhost:27017/",
                database_name=TEST_DB_NAME,
                encryption_key_path=config.get(
                    "storage.encryption_key_path",
                    "data/face_db/.encryption_key",
                ),
                encryption_enabled=True,
            )

            agent = ConsentIdentityAgent(
                config, privacy_profile=PrivacyProfile(), face_db=face_db
            )

            # Register contact with 0 appearances (not yet trusted)
            contact_emb = make_embedding(seed=20)
            person = PersonEntry(
                label="John",
                relationship="friend",
                embeddings=[FaceEmbedding(embedding=contact_emb)],
                consent_history=ConsentHistory(
                    times_appeared=0, times_approved=0
                ),
            )
            face_db.add_person(person)
            person_id = person.person_id
            agent._cache_dirty = True

            # First run: untrusted → UNCLEAR/HIGH
            query_emb = make_similar_embedding(
                contact_emb, noise_scale=0.02, seed=21
            )
            face = make_face_detection("face_contact1", query_emb)
            detections = DetectionResults(
                image_path="test.jpg", faces=[face]
            )
            risk_result = RiskAnalysisResult(
                image_path="test.jpg",
                risk_assessments=[make_risk_assessment("face_contact1")],
                overall_risk_level=RiskLevel.CRITICAL,
            )
            result = agent.run(detections, risk_result)

            a = result.risk_assessments[0]
            assert a.severity == RiskLevel.HIGH, (
                f"Expected HIGH (untrusted), got {a.severity}"
            )
            assert a.consent_status in (
                ConsentStatus.UNCLEAR, "unclear"
            ), f"Expected UNCLEAR, got {a.consent_status}"
            print("  Step 1: Untrusted contact → HIGH/UNCLEAR ✓")

            # Simulate 3 approvals (min_appearances_for_trust=3)
            for _ in range(3):
                agent.record_user_decision(
                    person_id, "approved", context="test"
                )

            # Manually set times_appeared to match
            p = face_db.get_person(person_id)
            p.consent_history.times_appeared = max(
                p.consent_history.times_appeared, 3
            )
            face_db.update_person(p)
            agent._cache_dirty = True

            # Second run: now trusted → ASSUMED
            face2 = make_face_detection("face_contact2", query_emb)
            detections2 = DetectionResults(
                image_path="test.jpg", faces=[face2]
            )
            risk_result2 = RiskAnalysisResult(
                image_path="test.jpg",
                risk_assessments=[make_risk_assessment("face_contact2")],
                overall_risk_level=RiskLevel.CRITICAL,
            )
            result2 = agent.run(detections2, risk_result2)

            a2 = result2.risk_assessments[0]
            assert a2.consent_status in (
                ConsentStatus.ASSUMED, "assumed"
            ), f"Expected ASSUMED after 3 approvals, got {a2.consent_status}"
            # Default friend_faces sensitivity = "medium" → MEDIUM severity
            assert a2.severity == RiskLevel.MEDIUM, (
                f"Expected MEDIUM for trusted friend, got {a2.severity}"
            )
            print("  Step 2: After 3 approvals → ASSUMED/MEDIUM ✓")

            print("  PASSED: Trust progression works correctly\n")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

        # ---- Test 4: Threshold verification ----
        print("\nTest 4: Similarity thresholds")
        print("-" * 40)
        try:
            cleanup_db(face_db)
            face_db = FaceDatabase(
                mongo_uri="mongodb://localhost:27017/",
                database_name=TEST_DB_NAME,
                encryption_key_path=config.get(
                    "storage.encryption_key_path",
                    "data/face_db/.encryption_key",
                ),
                encryption_enabled=True,
            )

            agent = ConsentIdentityAgent(
                config, privacy_profile=PrivacyProfile(), face_db=face_db
            )

            base_emb = make_embedding(seed=30)
            person = PersonEntry(
                label="TestPerson",
                relationship="self",
                embeddings=[FaceEmbedding(embedding=base_emb)],
                consent_history=ConsentHistory(
                    times_appeared=1, times_approved=1
                ),
            )
            face_db.add_person(person)
            agent._cache_dirty = True

            # a) Confident match (very similar, noise_scale=0.01)
            q_confident = make_similar_embedding(
                base_emb, noise_scale=0.01, seed=31
            )
            match_c = agent._match_face(
                np.array(q_confident, dtype=np.float32), "det_c"
            )
            assert match_c.match_type == "confident", (
                f"Expected confident, got {match_c.match_type} "
                f"(sim={match_c.recognition_confidence:.4f})"
            )
            print(
                f"  Confident: sim={match_c.recognition_confidence:.4f} ✓"
            )

            # b) No match (completely different embedding)
            q_different = make_embedding(seed=999)
            match_n = agent._match_face(
                np.array(q_different, dtype=np.float32), "det_n"
            )
            assert match_n.match_type == "no_match", (
                f"Expected no_match, got {match_n.match_type} "
                f"(sim={match_n.recognition_confidence:.4f})"
            )
            print(
                f"  No match: sim={match_n.recognition_confidence:.4f} ✓"
            )

            print("  PASSED: Threshold classification correct\n")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

        # ---- Test 5: Adaptive learning (record_user_decision) ----
        print("\nTest 5: Adaptive learning")
        print("-" * 40)
        try:
            cleanup_db(face_db)
            face_db = FaceDatabase(
                mongo_uri="mongodb://localhost:27017/",
                database_name=TEST_DB_NAME,
                encryption_key_path=config.get(
                    "storage.encryption_key_path",
                    "data/face_db/.encryption_key",
                ),
                encryption_enabled=True,
            )

            agent = ConsentIdentityAgent(
                config, privacy_profile=PrivacyProfile(), face_db=face_db
            )

            emb = make_embedding(seed=50)
            person = PersonEntry(
                label="LearnTest",
                relationship="friend",
                embeddings=[FaceEmbedding(embedding=emb)],
                consent_history=ConsentHistory(
                    times_appeared=5,
                    times_approved=0,
                    times_protected=0,
                ),
                risk_decay_factor=1.0,
            )
            face_db.add_person(person)
            pid = person.person_id

            # Record 3 approvals
            for _ in range(3):
                agent.record_user_decision(pid, "approved")

            p = face_db.get_person(pid)
            assert p.consent_history.times_approved == 3
            assert p.consent_history.last_consent_decision == "approved"
            expected_decay = 1.0 - 3 * 0.1  # 0.7
            assert abs(p.risk_decay_factor - expected_decay) < 0.01, (
                f"Expected decay ~{expected_decay}, got {p.risk_decay_factor}"
            )
            print(
                f"  After 3 approvals: decay={p.risk_decay_factor:.2f}, "
                f"rate={p.consent_history.approval_rate:.0%} ✓"
            )

            # Record 1 protection
            agent.record_user_decision(pid, "protected")
            p = face_db.get_person(pid)
            assert p.consent_history.times_protected == 1
            assert p.consent_history.last_consent_decision == "protected"
            expected_decay2 = 0.7 + 0.05  # 0.75
            assert abs(p.risk_decay_factor - expected_decay2) < 0.01
            print(
                f"  After 1 protection: decay={p.risk_decay_factor:.2f} ✓"
            )

            print("  PASSED: Adaptive learning updates correctly\n")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

        # ---- Test 6: Embedding diversity (max 5, duplicates rejected) ----
        print("\nTest 6: Embedding diversity")
        print("-" * 40)
        try:
            cleanup_db(face_db)
            face_db = FaceDatabase(
                mongo_uri="mongodb://localhost:27017/",
                database_name=TEST_DB_NAME,
                encryption_key_path=config.get(
                    "storage.encryption_key_path",
                    "data/face_db/.encryption_key",
                ),
                encryption_enabled=True,
            )

            agent = ConsentIdentityAgent(
                config, privacy_profile=PrivacyProfile(), face_db=face_db
            )

            base_emb = make_embedding(seed=60)
            person = PersonEntry(
                label="DiversityTest",
                relationship="self",
                embeddings=[FaceEmbedding(embedding=base_emb)],
                consent_history=ConsentHistory(
                    times_appeared=1, times_approved=1
                ),
            )
            face_db.add_person(person)
            pid = person.person_id
            agent._cache_dirty = True

            # Run with nearly identical embedding (sim > 0.95) → should NOT add
            identical_emb = make_similar_embedding(
                base_emb, noise_scale=0.001, seed=61
            )
            face = make_face_detection("face_dup", identical_emb)
            detections = DetectionResults(
                image_path="test.jpg", faces=[face]
            )
            risk_result = RiskAnalysisResult(
                image_path="test.jpg",
                risk_assessments=[make_risk_assessment("face_dup")],
                overall_risk_level=RiskLevel.CRITICAL,
            )
            agent.run(detections, risk_result)

            p = face_db.get_person(pid)
            assert len(p.embeddings) == 1, (
                f"Expected 1 embedding (duplicate rejected), got {len(p.embeddings)}"
            )
            print("  Duplicate rejected (sim > 0.95) ✓")

            # Run with diverse embedding (different angle) → should add
            diverse_emb = make_similar_embedding(
                base_emb, noise_scale=0.15, seed=62
            )
            face2 = make_face_detection("face_diverse", diverse_emb)
            detections2 = DetectionResults(
                image_path="test.jpg", faces=[face2]
            )
            risk_result2 = RiskAnalysisResult(
                image_path="test.jpg",
                risk_assessments=[make_risk_assessment("face_diverse")],
                overall_risk_level=RiskLevel.CRITICAL,
            )
            agent.run(detections2, risk_result2)

            p = face_db.get_person(pid)
            assert len(p.embeddings) == 2, (
                f"Expected 2 embeddings (diverse added), got {len(p.embeddings)}"
            )
            print("  Diverse embedding added ✓")

            print("  PASSED: Embedding diversity management correct\n")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

        # ---- Test 7: Multiple faces (user + bystander in same image) ----
        print("\nTest 7: Multiple faces — user + bystander")
        print("-" * 40)
        try:
            cleanup_db(face_db)
            face_db = FaceDatabase(
                mongo_uri="mongodb://localhost:27017/",
                database_name=TEST_DB_NAME,
                encryption_key_path=config.get(
                    "storage.encryption_key_path",
                    "data/face_db/.encryption_key",
                ),
                encryption_enabled=True,
            )

            agent = ConsentIdentityAgent(
                config, privacy_profile=PrivacyProfile(), face_db=face_db
            )

            # Register user
            user_emb = make_embedding(seed=70)
            person = PersonEntry(
                label="Me",
                relationship="self",
                embeddings=[FaceEmbedding(embedding=user_emb)],
                consent_history=ConsentHistory(
                    times_appeared=1, times_approved=1
                ),
            )
            face_db.add_person(person)
            agent._cache_dirty = True

            # Two faces: user + stranger
            user_query = make_similar_embedding(
                user_emb, noise_scale=0.02, seed=71
            )
            stranger_emb = make_embedding(seed=999)

            f1 = make_face_detection("face_me", user_query)
            f2 = make_face_detection("face_stranger", stranger_emb)

            detections = DetectionResults(
                image_path="test.jpg", faces=[f1, f2]
            )
            risk_result = RiskAnalysisResult(
                image_path="test.jpg",
                risk_assessments=[
                    make_risk_assessment("face_me"),
                    make_risk_assessment("face_stranger"),
                ],
                overall_risk_level=RiskLevel.CRITICAL,
            )

            result = agent.run(detections, risk_result)

            # User face → LOW
            a_me = next(
                a
                for a in result.risk_assessments
                if a.detection_id == "face_me"
            )
            assert a_me.severity == RiskLevel.LOW
            assert a_me.person_label == "Me"

            # Stranger → stays CRITICAL
            a_stranger = next(
                a
                for a in result.risk_assessments
                if a.detection_id == "face_stranger"
            )
            assert a_stranger.severity == RiskLevel.CRITICAL

            # Overall stays CRITICAL (bystander present)
            assert result.overall_risk_level == RiskLevel.CRITICAL
            assert result.faces_pending_identity == 1

            print("  User → LOW, Stranger → CRITICAL ✓")
            print("  Overall → CRITICAL (bystander present) ✓")
            print("  PASSED: Multi-face identification correct\n")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

    finally:
        # Cleanup
        cleanup_db(face_db)
        face_db.close()
        print(f"\nCleaned up test database: {TEST_DB_NAME}")

    # Summary
    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    print("\nPrivacy Guard - Consent Identity Agent Test\n")
    success = run_tests()
    sys.exit(0 if success else 1)