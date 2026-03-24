"""
Full Pipeline Integration Test
Detection → Risk Assessment → Consent Identity → Strategy → JSON + Risk Map

Runs the complete privacy protection pipeline including:
- Agent 1: Detection (faces, text, objects)
- Agent 2: Risk Assessment (Phase 1 deterministic + optional Phase 2 VLM)
- Agent 2.5: Consent Identity (face matching against MongoDB)
- Agent 3: Strategy (Phase 1 rule defaults + optional Phase 2 VLM review)
- JSON export with all identity fields + strategy recommendations
- Visual risk map with severity-colored bounding boxes

Requires: MongoDB running locally (mongod)
"""

import sys
import time 
import traceback
import argparse
from pathlib import Path
from PIL import Image



# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "utils"))

from utils.config import load_config
from utils.models import RiskLevel, PrivacyProfile
from utils.segmentation import PrecisionSegmenter
from utils.visualization import export_risk_results_json, generate_risk_map, export_strategy_results_json, generate_protection_preview
from utils.storage import FaceDatabase
from agents.detection_agent import DetectionAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.consent_identity_agent import ConsentIdentityAgent
from agents.strategy_agent import StrategyAgent
from agents.execution_agent import ExecutionAgent


# Test database (separate from production)
TEST_DB_NAME = "privacy_guard_test_pipeline"

# Image containing user's face for registration
USER_FACE_IMAGE = "sample1.png"


def print_risk_summary(result):
    """Print a detailed summary of risk assessment results."""
    print(f"\n  Risk Assessment Summary:")
    print(f"  ├─ Overall risk: {result.overall_risk_level.value.upper()}")
    print(f"  ├─ Total assessments: {len(result.risk_assessments)}")
    print(f"  ├─ Requires protection: {result.confirmed_risks}")
    print(f"  ├─ Faces pending identity: {result.faces_pending_identity}")
    print(f"  └─ Processing time: {result.processing_time_ms:.2f}ms")

    critical = result.get_critical_risks()
    high = result.get_high_risks()
    medium = result.get_by_severity(RiskLevel.MEDIUM)
    low = result.get_by_severity(RiskLevel.LOW)

    print(f"\n  Severity Breakdown:")
    print(f"  ├─ Critical: {len(critical)}")
    print(f"  ├─ High: {len(high)}")
    print(f"  ├─ Medium: {len(medium)}")
    print(f"  └─ Low: {len(low)}")

    if result.risk_assessments:
        print(f"\n  Assessment Details:")
        for i, a in enumerate(result.risk_assessments, 1):
            severity_tag = a.severity.value.upper()
            protection = "PROTECT" if a.requires_protection else "safe"

            print(f"    {i}. [{severity_tag:>8}] {a.element_description}")
            print(f"       Type: {a.element_type} | Risk: {a.risk_type.value}")
            print(f"       Protection: {protection} | Consent: {a.consent_status}")

            # Show identity info for faces
            if a.element_type == "face":
                label = a.person_label or "Unknown"
                cls = a.classification.value if hasattr(a.classification, "value") else (a.classification or "—")
                print(f"       Identity: {label} ({cls}) | Confidence: {a.consent_confidence:.3f}")

            print(f"       Reasoning: {a.reasoning[:80]}...")
            print()


def test_full_pipeline(
    dataset_path: str,
    reasoning_mode: str = "balanced",
    fallback_only: bool = False,
    keep_db: bool = False,
):
    """
    Test the full Detection → Risk Assessment → Consent Identity pipeline.

    Args:
        dataset_path: Path to directory containing test images
        reasoning_mode: "fast" | "balanced" | "thorough"
        fallback_only: If True, skip VLM and test Phase 1 only
        keep_db: If True, don't drop test database after run
    """
    print("=" * 80)
    print("Full Pipeline Integration Test")
    print("Detection → Risk Assessment → Consent Identity → Strategy → JSON + Risk Map")
    print("=" * 80 + "\n")

    pipeline_start = time.time()

    # ---- Step 1: Load configuration ----
    print("Step 1: Loading configuration...")
    try:
        config = load_config()
        print(f"  Config loaded (device: {config.system.device})\n")
    except Exception as e:
        print(f"  Failed to load configuration: {e}")
        return False

    # ---- Step 2: Initialize agents ----
    print("Step 2: Initializing agents...")

    # 2a: Detection Agent
    try:
        detection_agent = DetectionAgent(config)
        print(f"  Detection Agent ready")
    except Exception as e:
        print(f"  Failed to initialize Detection Agent: {e}")
        traceback.print_exc()
        return False

    # 2b: Risk Assessment Agent
    try:
        risk_agent = RiskAssessmentAgent(
            config=config,
            privacy_profile=PrivacyProfile(),
            reasoning_mode=reasoning_mode,
            ocr_reader=detection_agent.text_tool.detector,  # Share EasyOCR reader (~500MB savings)
        )
        print(f"  Risk Assessment Agent ready")
    except Exception as e:
        print(f"  Failed to initialize Risk Assessment Agent: {e}")
        traceback.print_exc()
        return False

    # 2c: Consent Identity Agent (with test MongoDB)
    face_db = None
    try:
        face_db = FaceDatabase(
            mongo_uri="mongodb://localhost:27017/",
            database_name=TEST_DB_NAME,
            encryption_key_path=config.get(
                "storage.encryption_key_path", "data/face_db/.encryption_key"
            ),
            encryption_enabled=True,
        )
        # Clean slate for reproducible results
        face_db.client.drop_database(TEST_DB_NAME)
        face_db = FaceDatabase(
            mongo_uri="mongodb://localhost:27017/",
            database_name=TEST_DB_NAME,
            encryption_key_path=config.get(
                "storage.encryption_key_path", "data/face_db/.encryption_key"
            ),
            encryption_enabled=True,
        )

        consent_agent = ConsentIdentityAgent(
            config=config,
            privacy_profile=PrivacyProfile(),
            face_db=face_db,
        )
        print(f"  Consent Identity Agent ready (DB: {TEST_DB_NAME})")
    except Exception as e:
        print(f"  Failed to initialize Consent Identity Agent: {e}")
        print(f"  Make sure MongoDB is running: mongod")
        traceback.print_exc()
        return False

    # 2d: Strategy Agent
    try:
        strategy_agent = StrategyAgent(
            config=config,
            privacy_profile=PrivacyProfile(),
            vlm_backend="llama-cpp",
        )
        print(f"  Strategy Agent ready")
    except Exception as e:
        print(f"  Failed to initialize Strategy Agent: {e}")
        traceback.print_exc()
        return False

    execution_agent = ExecutionAgent(
        vlm_backend=None if fallback_only else "llama-cpp"
    )
    print(f"  Execution Agent ready")

    print()

    # ---- Step 3: Find test images ----
    print("Step 3: Finding test images...")
    dataset_path = Path(dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path

    if not dataset_path.exists():
        print(f"  Dataset path does not exist: {dataset_path}")
        return False

    supported_formats = [".jpg", ".jpeg", ".png", ".webp"]
    image_files = sorted([
        f for f in dataset_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported_formats
    ])

    if not image_files:
        print(f"  No images found in {dataset_path}")
        return False
    print(f"  Found {len(image_files)} test images\n")

    # ---- Step 4: Register user's face ----
    print("Step 4: Registering user's face...")
    user_image = dataset_path / USER_FACE_IMAGE
    if user_image.exists():
        person_id = consent_agent.register_user_face(str(user_image), label="Me")
        if person_id:
            print(f"  User face registered: {person_id}")
            stats = face_db.get_statistics()
            print(f"  Database: {stats['total_persons']} persons, {stats['total_embeddings']} embeddings\n")
        else:
            print(f"  Warning: Failed to register user face from {USER_FACE_IMAGE}")
            print(f"  Continuing without user registration...\n")
    else:
        print(f"  Warning: {USER_FACE_IMAGE} not found in {dataset_path}")
        print(f"  Continuing without user registration...\n")

    # Create output directory
    output_dir = project_root / "data" / "full_pipeline_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 5: Run full pipeline on each image ----
    print("Step 5: Running full pipeline...")
    print("-" * 80)

    results_summary = []

    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(image_files)}] Processing: {image_path.name}")
        print(f"{'='*80}")

        try:
            # Phase A: Detection
            print(f"\n  Phase A: Detection Agent...")
            detections = detection_agent.run(str(image_path))

            print(f"  Detection Results:")
            print(f"    Faces: {len(detections.faces)}")
            print(f"    Text regions: {len(detections.text_regions)}")
            print(f"    Objects: {len(detections.objects)}")

            annotated_image = detection_agent.get_annotated_image()

            # Phase B: Risk Assessment
            print(f"\n  Phase B: Risk Assessment Agent...")
            if fallback_only:
                print(f"  (Phase 1 only — no VLM)")
                start = time.time()
                img = Image.open(str(image_path))
                w, h = img.size
                image_context = {
                    "width": w, "height": h,
                    "total_faces": len(detections.faces),
                    "total_texts": len(detections.text_regions),
                    "total_objects": len(detections.objects),
                }
                assessments = risk_agent._tool_based_assessment(detections, image_context)
                risk_result = risk_agent._build_result(assessments, str(image_path), start)
            else:
                risk_result = risk_agent.run(detections, annotated_image)

            print(f"  Risk Assessment: {len(risk_result.risk_assessments)} assessments, "
                  f"overall={risk_result.overall_risk_level.value.upper()}")

            # Phase C: Consent Identity
            print(f"\n  Phase C: Consent Identity Agent...")
            risk_result = consent_agent.run(detections, risk_result)

            # Print full results
            print_risk_summary(risk_result)

            # Phase D: Strategy Recommendations
            print(f"\n  Phase D: Strategy Agent...")
            if fallback_only:
                strategy_result = strategy_agent.run(risk_result, str(image_path))
            else:
                strategy_result = strategy_agent.run(risk_result, str(image_path), annotated_image)

            # Phase D.5: Precise Segmentation (SAM)
            if not fallback_only:
                print(f"\n  Phase D.5: SAM Segmentation...")
                try:
                    if not hasattr(test_full_pipeline, '_segmenter'):
                        test_full_pipeline._segmenter = PrecisionSegmenter(device="cpu")
                    segmenter = test_full_pipeline._segmenter

                    seg_results = segmenter.process_strategies(
                        str(image_path),
                        strategy_result.strategies,
                        risk_result.risk_assessments,
                        output_dir=str(output_dir),
                    )
                    # Store mask paths on strategies for Agent 4
                    for strategy in strategy_result.strategies:
                        if strategy.detection_id in seg_results:
                            mask_data = seg_results[strategy.detection_id]
                            strategy.segmentation_mask_path = mask_data.get("mask_path")
                    print(f"  SAM: {len(seg_results)} masks generated")

                    # Generate side-by-side protection preview (bbox vs SAM)
                    if seg_results:
                        preview_path = output_dir / f"{image_path.stem}_protection_preview.png"
                        generate_protection_preview(
                            str(image_path),
                            strategy_result,
                            risk_result,
                            seg_results,
                            output_path=str(preview_path),
                        )
                except ImportError:
                    print(f"  SAM skipped (mobile-sam not installed)")
                except FileNotFoundError as e:
                    print(f"  SAM skipped ({e})")
                except Exception as e:
                    print(f"  SAM error (non-fatal): {e}")
                    import traceback; traceback.print_exc()

            # Phase E: Agent 4 — Execution (apply protections)
            print(f"\n  Phase E: Execution Agent...")
            protected_path = output_dir / f"{image_path.stem}_protected.png"
            execution_report = execution_agent.run(
                strategy_result=strategy_result,
                risk_result=risk_result,
                image_path=str(image_path),
                output_path=str(protected_path),
            )
            print(f"  Protected image saved: {protected_path}")

            # Phase F: Export JSON + Risk Map + Strategy JSON
            json_path = output_dir / f"{image_path.stem}_risk_results.json"
            export_risk_results_json(
                risk_result, detections=detections, output_path=str(json_path)
            )

            risk_map_path = output_dir / f"{image_path.stem}_risk_map.png"
            generate_risk_map(
                risk_result, str(image_path), output_path=str(risk_map_path)
            )

            strategy_json_path = output_dir / f"{image_path.stem}_strategies.json"
            export_strategy_results_json(
                strategy_result, output_path=str(strategy_json_path)
            )

            results_summary.append({
                "image": image_path.name,
                "success": True,
                "overall_risk": risk_result.overall_risk_level.value,
                "total_assessments": len(risk_result.risk_assessments),
                "critical": len(risk_result.get_critical_risks()),
                "high": len(risk_result.get_high_risks()),
                "low": len(risk_result.get_by_severity(RiskLevel.LOW)),
                "requires_protection": risk_result.confirmed_risks,
                "faces_identified": sum(
                    1 for a in risk_result.risk_assessments
                    if a.element_type == "face" and a.person_id is not None
                ),
                "faces_total": sum(
                    1 for a in risk_result.risk_assessments
                    if a.element_type == "face"
                ),
                "strategies_total": len(strategy_result.strategies),
                "protections_recommended": strategy_result.total_protections_recommended,
                "user_confirmations": strategy_result.requires_user_confirmation,
            })

        except Exception as e:
            print(f"\n  Error processing {image_path.name}: {e}")
            traceback.print_exc()
            results_summary.append({
                "image": image_path.name,
                "success": False,
                "error": str(e),
            })

    # ---- Final Summary ----
    pipeline_time = time.time() - pipeline_start
    print(f"\n{'='*80}")
    print("Full Pipeline Complete")
    print(f"{'='*80}")

    successful = sum(1 for r in results_summary if r["success"])
    print(f"\nImages processed: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(image_files) - successful}")
    print(f"Total pipeline time: {pipeline_time:.1f}s")

    if successful > 0:
        print(f"\nPer-image results:")
        for r in results_summary:
            if r["success"]:
                print(
                    f"  {r['image']:20s} | Risk: {r['overall_risk']:>8s} | "
                    f"Assessments: {r['total_assessments']:>2d} | "
                    f"Critical: {r['critical']} | Low: {r['low']} | "
                    f"Faces: {r['faces_identified']}/{r['faces_total']} identified | "
                    f"Protections: {r['protections_recommended']}"
                )
            else:
                print(f"  {r['image']:20s} | FAILED: {r['error'][:50]}")

        total_faces = sum(r.get("faces_total", 0) for r in results_summary if r["success"])
        identified_faces = sum(r.get("faces_identified", 0) for r in results_summary if r["success"])
        print(f"\nFace identification: {identified_faces}/{total_faces} faces matched")

    # Database stats
    stats = face_db.get_statistics()
    print(f"\nMongoDB ({TEST_DB_NAME}):")
    print(f"  Persons: {stats.get('total_persons', 0)}")
    print(f"  Embeddings: {stats.get('total_embeddings', 0)}")

    print(f"\nResults saved to: {output_dir}")

    # Cleanup
    if keep_db:
        print(f"\nTest database preserved: {TEST_DB_NAME}")
    else:
        face_db.client.drop_database(TEST_DB_NAME)
        print(f"\nTest database dropped: {TEST_DB_NAME}")

    consent_agent.close()
    print("=" * 80)

    return successful == len(image_files)


def main():
    parser = argparse.ArgumentParser(
        description="Full Pipeline Test: Detection → Risk Assessment → Consent Identity → Strategy"
    )
    parser.add_argument(
        "--dataset", type=str, default="data/test_images",
        help="Path to test images directory (default: data/test_images)",
    )
    parser.add_argument(
        "--mode", type=str, default="balanced",
        choices=["fast", "balanced", "thorough"],
        help="Reasoning mode (default: balanced)",
    )
    parser.add_argument(
        "--fallback-only", action="store_true",
        help="Phase 1 only — no VLM review (no llama-server needed)",
    )
    parser.add_argument(
        "--keep-db", action="store_true",
        help="Don't drop test MongoDB database after run",
    )

    args = parser.parse_args()

    success = test_full_pipeline(
        dataset_path=args.dataset,
        reasoning_mode=args.mode,
        fallback_only=args.fallback_only,
        keep_db=args.keep_db,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
