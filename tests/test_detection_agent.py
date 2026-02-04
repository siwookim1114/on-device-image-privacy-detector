"""
Test script for Detection Agent
Run the detection agent on a simple dataset and visualize results
"""
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from agents.detection_agent import DetectionAgent
from PIL import Image, ImageDraw, ImageFont


def visualize_detections(image_path: str, detections, output_path: str = None):
    """
    Visualize detection results on the image

    Args:
        image_path: Path to original image
        detections: DetectionResults object
        output_path: Path to save visualization (optional)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Draw faces (red boxes)
    for face in detections.faces:
        bbox = face.bbox
        x1, y1 = bbox.x, bbox.y
        x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label = f"Face {face.confidence:.2f}"
        draw.text((x1, y1 - 20), label, fill="red", font=font)

    # Draw text regions (green boxes)
    for text_region in detections.text_regions:
        bbox = text_region.bbox
        x1, y1 = bbox.x, bbox.y
        x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height

        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        label = f"Text: {text_region.text_content[:20]}"
        draw.text((x1, y1 - 20), label, fill="green", font=small_font)

    # Draw objects (blue boxes)
    for obj in detections.objects:
        bbox = obj.bbox
        x1, y1 = bbox.x, bbox.y
        x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height

        draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
        label = f"{obj.object_class} {obj.confidence:.2f}"
        draw.text((x1, y1 - 20), label, fill="blue", font=font)

    # Save or display
    if output_path:
        image.save(output_path)
        print(f"  Visualization saved: {output_path}")
    else:
        image.show()

    return image


def print_detection_summary(detections):
    """Print a summary of detections"""
    print(f"\n  Summary:")
    print(f"  ├─ Total detections: {detections.total_detections}")
    print(f"  ├─ Faces: {len(detections.faces)}")
    print(f"  ├─ Text regions: {len(detections.text_regions)}")
    print(f"  ├─ Objects: {len(detections.objects)}")
    print(f"  └─ Processing time: {detections.processing_time_ms:.2f}ms")

    if detections.faces:
        print(f"\n  Face details:")
        for i, face in enumerate(detections.faces, 1):
            print(f"    {i}. Confidence: {face.confidence:.3f}, Size: {face.size}, Clarity: {face.clarity}")

    if detections.text_regions:
        print(f"\n  Text details:")
        for i, text in enumerate(detections.text_regions, 1):
            content_preview = text.text_content[:50] + "..." if len(text.text_content) > 50 else text.text_content
            print(f"    {i}. Type: {text.text_type}, Confidence: {text.confidence:.3f}")
            print(f"       Content: '{content_preview}'")

    if detections.objects:
        print(f"\n  Object details:")
        for i, obj in enumerate(detections.objects, 1):
            print(f"    {i}. Class: {obj.object_class}, Confidence: {obj.confidence:.3f}")


def save_results_json(detections, output_path: str):
    """Save detection results as JSON"""
    results = {
        "image_path": detections.image_path,
        "processing_time_ms": detections.processing_time_ms,
        "total_detections": detections.total_detections,
        "faces": [
            {
                "bbox": face.bbox.to_list(),
                "confidence": face.confidence,
                "size": face.size,
                "clarity": face.clarity
            }
            for face in detections.faces
        ],
        "text_regions": [
            {
                "bbox": text.bbox.to_list(),
                "confidence": text.confidence,
                "text_content": text.text_content,
                "text_type": text.text_type
            }
            for text in detections.text_regions
        ],
        "objects": [
            {
                "bbox": obj.bbox.to_list(),
                "confidence": obj.confidence,
                "object_class": obj.object_class,
                "contains_screen": obj.contains_screen
            }
            for obj in detections.objects
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved: {output_path}")


def test_detection_agent_on_dataset(dataset_path: str, visualize: bool = True, save_json: bool = True):
    """
    Test detection agent on all images in a dataset directory

    Args:
        dataset_path: Path to directory containing test images
        visualize: Whether to create visualization images
        save_json: Whether to save results as JSON
    """
    print("=" * 80)
    print("Testing Detection Agent on Dataset")
    print("=" * 80 + "\n")

    # Load configuration
    print("Step 1: Loading configuration...")
    try:
        config = load_config()
        print(f"✓ Configuration loaded")
        print(f"  Device: {config.system.device}")
        print(f"  Face detector: {config.models.detection.face_detector}")
        print(f"  Text detector: {config.models.detection.text_detector}")
        print(f"  Object detector: {config.models.detection.object_detector}\n")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False

    # Initialize agent
    print("Step 2: Initializing Detection Agent...")
    try:
        agent = DetectionAgent(config)
        print(f"✓ Agent initialized\n")
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Find test images
    print("Step 3: Finding test images...")
    dataset_path = Path(dataset_path)
    # If path is relative, resolve it relative to project root
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    if not dataset_path.exists():
        print(f"✗ Dataset path does not exist: {dataset_path}")
        print(f"\nPlease create the directory and add test images:")
        print(f"  mkdir -p {dataset_path}")
        print(f"  # Then copy some test images to {dataset_path}/")
        return False

    supported_formats = ['.jpg', '.jpeg', '.png', '.webp']
    image_files = [
        f for f in dataset_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported_formats
    ]

    if not image_files:
        print(f"✗ No images found in {dataset_path}")
        print(f"  Supported formats: {', '.join(supported_formats)}")
        return False

    print(f"✓ Found {len(image_files)} test images\n")

    # Create output directory
    output_dir = project_root / "data" / "test_results_after_update_2"
    output_dir.mkdir(exist_ok=True)

    # Process each image
    print("Step 4: Running detection on test images...")
    print("-" * 80)

    results_summary = []

    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        print("-" * 80)

        try:
            # Run detection
            detections = agent.run(str(image_path))

            # Print summary
            print_detection_summary(detections)

            # Save results
            if save_json:
                json_path = output_dir / f"{image_path.stem}_results.json"
                save_results_json(detections, str(json_path))

            # Visualize
            if visualize:
                vis_path = output_dir / f"{image_path.stem}_visualized.png"
                visualize_detections(str(image_path), detections, str(vis_path))

            results_summary.append({
                "image": image_path.name,
                "success": True,
                "faces": len(detections.faces),
                "text_regions": len(detections.text_regions),
                "objects": len(detections.objects),
                "processing_time_ms": detections.processing_time_ms
            })

        except Exception as e:
            print(f"  ✗ Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                "image": image_path.name,
                "success": False,
                "error": str(e)
            })

    # Final summary
    print("\n" + "=" * 80)
    print("Test Complete - Summary")
    print("=" * 80)

    successful = sum(1 for r in results_summary if r["success"])
    print(f"\nImages processed: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(image_files) - successful}")

    if successful > 0:
        total_faces = sum(r.get("faces", 0) for r in results_summary if r["success"])
        total_text = sum(r.get("text_regions", 0) for r in results_summary if r["success"])
        total_objects = sum(r.get("objects", 0) for r in results_summary if r["success"])
        avg_time = sum(r.get("processing_time_ms", 0) for r in results_summary if r["success"]) / successful

        print(f"\nTotal detections:")
        print(f"  Faces: {total_faces}")
        print(f"  Text regions: {total_text}")
        print(f"  Objects: {total_objects}")
        print(f"\nAverage processing time: {avg_time:.2f}ms")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)

    return successful == len(image_files)


def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Detection Agent on a dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/test_images",
        help="Path to test images directory (default: data/test_images)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip creating visualization images"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving JSON results"
    )

    args = parser.parse_args()

    success = test_detection_agent_on_dataset(
        dataset_path=args.dataset,
        visualize=not args.no_visualize,
        save_json=not args.no_json
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
