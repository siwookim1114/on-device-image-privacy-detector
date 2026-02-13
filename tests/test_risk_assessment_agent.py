"""
Test script for Risk Assessment Agent
Runs the full pipeline: Detection Agent → Risk Assessment Agent
Tests that all detection data flows correctly into risk assessment.
"""
import sys
import json
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from utils.models import RiskLevel, PrivacyProfile
from agents.detection_agent import DetectionAgent
from agents.risk_assessment_agent import RiskAssessmentAgent


def print_risk_summary(result):
    """Print a detailed summary of risk assessment results."""
    print(f"\n  Risk Assessment Summary:")
    print(f"  ├─ Overall risk: {result.overall_risk_level.value.upper()}")
    print(f"  ├─ Total assessments: {len(result.risk_assessments)}")
    print(f"  ├─ Requires protection: {result.confirmed_risks}")
    print(f"  ├─ Faces pending identity: {result.faces_pending_identity}")
    print(f"  └─ Processing time: {result.processimg_time_ms:.2f}ms")

    # Breakdown by severity
    critical = result.get_critical_risks()
    high = result.get_high_risks()
    medium = result.get_by_serverity(RiskLevel.MEDIUM)
    low = result.get_by_serverity(RiskLevel.LOW)

    print(f"\n  Severity Breakdown:")
    print(f"  ├─ Critical: {len(critical)}")
    print(f"  ├─ High: {len(high)}")
    print(f"  ├─ Medium: {len(medium)}")
    print(f"  └─ Low: {len(low)}")

    # Individual assessment details
    if result.risk_assessments:
        print(f"\n  Assessment Details:")
        for i, assessment in enumerate(result.risk_assessments, 1):
            severity_color = {
                RiskLevel.CRITICAL: "CRITICAL",
                RiskLevel.HIGH: "HIGH",
                RiskLevel.MEDIUM: "MEDIUM",
                RiskLevel.LOW: "LOW"
            }.get(assessment.severity, "UNKNOWN")

            protection = "PROTECT" if assessment.requires_protection else "safe"

            print(f"    {i}. [{severity_color:>8}] {assessment.element_description}")
            print(f"       Type: {assessment.element_type} | Risk: {assessment.risk_type.value}")
            print(f"       Protection: {protection} | Consent: {assessment.consent_status}")
            print(f"       Reasoning: {assessment.reasoning[:80]}...")
            print()


def save_risk_results_json(result, detections, output_path: str):
    """Save risk assessment results as JSON."""
    results = {
        "image_path": result.image_path,
        "overall_risk_level": result.overall_risk_level.value,
        "processing_time_ms": result.processimg_time_ms,
        "total_assessments": len(result.risk_assessments),
        "confirmed_risks": result.confirmed_risks,
        "faces_pending_identity": result.faces_pending_identity,
        "detection_summary": {
            "faces": len(detections.faces),
            "text_regions": len(detections.text_regions),
            "objects": len(detections.objects),
            "total": detections.total_detections
        },
        "assessments": [
            {
                "detection_id": a.detection_id,
                "element_type": a.element_type,
                "element_description": a.element_description,
                "risk_type": a.risk_type.value,
                "severity": a.severity.value,
                "color_code": a.color_code,
                "reasoning": a.reasoning,
                "bbox": a.bbox.to_list(),
                "requires_protection": a.requires_protection,
                "consent_status": a.consent_status.value if a.consent_status else None,
                "consent_confidence": a.consent_confidence,
                "user_sensitivity_applied": a.user_sensitivity_applied
            }
            for a in result.risk_assessments
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved: {output_path}")


def test_risk_assessment_pipeline(
    dataset_path: str,
    reasoning_mode: str = "balanced",
    fallback_only: bool = False,
    skip_vlm: bool = False,
    save_json: bool = True
):
    """
    Test the full Detection → Risk Assessment pipeline.

    Args:
        dataset_path: Path to directory containing test images
        reasoning_mode: "fast" | "balanced" | "thorough"
        fallback_only: If True, skip ReAct agent and test fallback path only
        save_json: Whether to save results as JSON
    """
    print("=" * 80)
    print("Testing Risk Assessment Agent (Full Pipeline)")
    print("=" * 80 + "\n")

    # Step 1: Load configuration
    print("Step 1: Loading configuration...")
    try:
        config = load_config()
        print(f"  Config loaded (device: {config.system.device})\n")
    except Exception as e:
        print(f"  Failed to load configuration: {e}")
        return False

    # Step 2: Initialize Detection Agent
    print("Step 2: Initializing Detection Agent...")
    try:
        detection_agent = DetectionAgent(config)
        print(f"  Detection Agent ready\n")
    except Exception as e:
        print(f"  Failed to initialize Detection Agent: {e}")
        traceback.print_exc()
        return False

    # Step 3: Initialize Risk Assessment Agent
    print("Step 3: Initializing Risk Assessment Agent...")
    try:
        risk_agent = RiskAssessmentAgent(
            config=config,
            privacy_profile=PrivacyProfile(),
            reasoning_mode=reasoning_mode
        )
        print(f"  Risk Assessment Agent ready\n")
    except Exception as e:
        print(f"  Failed to initialize Risk Assessment Agent: {e}")
        traceback.print_exc()
        return False

    # Step 4: Find test images
    print("Step 4: Finding test images...")
    dataset_path = Path(dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path

    if not dataset_path.exists():
        print(f"  Dataset path does not exist: {dataset_path}")
        return False

    supported_formats = ['.jpg', '.jpeg', '.png', '.webp']
    image_files = sorted([
        f for f in dataset_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported_formats
    ])

    if not image_files:
        print(f"  No images found in {dataset_path}")
        return False

    print(f"  Found {len(image_files)} test images\n")

    # Create output directory
    output_dir = project_root / "data" / "risk_assessment_results"
    output_dir.mkdir(exist_ok=True)

    # Step 5: Run full pipeline on each image
    print("Step 5: Running Detection → Risk Assessment pipeline...")
    print("-" * 80)

    results_summary = []

    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(image_files)}] Processing: {image_path.name}")
        print(f"{'='*80}")

        try:
            # Phase A: Detection
            print(f"\n  Phase A: Running Detection Agent...")
            detections = detection_agent.run(str(image_path))

            print(f"  Detection Results:")
            print(f"    - Faces: {len(detections.faces)}")
            print(f"    - Text regions: {len(detections.text_regions)}")
            print(f"    - Objects: {len(detections.objects)}")
            print(f"    - Total: {detections.total_detections}")

            # Get annotated image
            annotated_image = detection_agent.get_annotated_image()

            # Phase B: Risk Assessment
            print(f"\n  Phase B: Running Risk Assessment Agent...")

            if fallback_only:
                # Test fallback path directly (no LLM needed)
                print(f"  (Using fallback mode - no LLM required)")
                import time
                start_time = time.time()
                image_obj = __import__('PIL.Image', fromlist=['Image']).open(str(image_path))
                w, h = image_obj.size
                image_context = {
                    "width": w, "height": h,
                    "total_faces": len(detections.faces),
                    "total_texts": len(detections.text_regions),
                    "total_objects": len(detections.objects)
                }
                assessments = risk_agent._tool_based_assessment(detections, image_context)
                risk_result = risk_agent._build_result(assessments, str(image_path), start_time)
            elif skip_vlm:
                # Run full ReAct Phase 1 but skip Phase 2 VLM review
                print(f"  (Skipping VLM Phase 2 review)")
                risk_result = risk_agent.run(detections, annotated_image=None)
            else:
                risk_result = risk_agent.run(detections, annotated_image)

            # Print results
            print_risk_summary(risk_result)

            # Save results
            if save_json:
                json_path = output_dir / f"{image_path.stem}_risk_results.json"
                save_risk_results_json(risk_result, detections, str(json_path))

            results_summary.append({
                "image": image_path.name,
                "success": True,
                "overall_risk": risk_result.overall_risk_level.value,
                "total_assessments": len(risk_result.risk_assessments),
                "critical": len(risk_result.get_critical_risks()),
                "high": len(risk_result.get_high_risks()),
                "requires_protection": risk_result.confirmed_risks,
                "detection_time_ms": detections.processing_time_ms,
                "risk_time_ms": risk_result.processimg_time_ms
            })

        except Exception as e:
            print(f"\n  Error processing {image_path.name}: {e}")
            traceback.print_exc()
            results_summary.append({
                "image": image_path.name,
                "success": False,
                "error": str(e)
            })

    # Final summary
    print(f"\n{'='*80}")
    print("Test Complete - Final Summary")
    print(f"{'='*80}")

    successful = sum(1 for r in results_summary if r["success"])
    print(f"\nImages processed: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(image_files) - successful}")

    if successful > 0:
        print(f"\nPer-image results:")
        for r in results_summary:
            if r["success"]:
                print(f"  {r['image']:20s} | Risk: {r['overall_risk']:>8s} | "
                      f"Assessments: {r['total_assessments']:>2d} | "
                      f"Critical: {r['critical']} | High: {r.get('high', 0)} | "
                      f"Protect: {r['requires_protection']}")
            else:
                print(f"  {r['image']:20s} | FAILED: {r['error'][:50]}")

        total_assessments = sum(r.get("total_assessments", 0) for r in results_summary if r["success"])
        total_critical = sum(r.get("critical", 0) for r in results_summary if r["success"])
        total_protection = sum(r.get("requires_protection", 0) for r in results_summary if r["success"])

        print(f"\nAggregate stats:")
        print(f"  Total assessments: {total_assessments}")
        print(f"  Total critical risks: {total_critical}")
        print(f"  Total requiring protection: {total_protection}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)

    return successful == len(image_files)


def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Risk Assessment Agent (Full Pipeline)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/test_images",
        help="Path to test images directory (default: data/test_images)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "thorough"],
        help="Reasoning mode (default: balanced)"
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Skip ReAct agent and test fallback tool-based path only (no LLM needed)"
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Run Phase 1 (ReAct or fallback) but skip Phase 2 VLM visual review"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving JSON results"
    )

    args = parser.parse_args()

    success = test_risk_assessment_pipeline(
        dataset_path=args.dataset,
        reasoning_mode=args.mode,
        fallback_only=args.fallback_only,
        skip_vlm=args.skip_vlm,
        save_json=not args.no_json
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
