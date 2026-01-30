"""
Download sample test images for testing the Detection Agent
These are public domain or Creative Commons licensed images
"""
import urllib.request
from pathlib import Path
import sys

def download_image(url: str, filename: str, dest_dir: Path) -> bool:
    """Download an image from URL"""
    try:
        dest_path = dest_dir / filename
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"  ✓ Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    # Create test images directory
    test_dir = Path(__file__).parent.parent / "data" / "test_images"
    test_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Downloading Sample Test Images")
    print("=" * 70 + "\n")

    # Sample images from various sources
    # Note: These are placeholder URLs - you can replace with actual test images
    samples = [
        {
            "name": "sample_faces.jpg",
            "url": "https://picsum.photos/800/600?random=1",
            "description": "Random photo (may contain faces)"
        },
        {
            "name": "sample_text.jpg",
            "url": "https://picsum.photos/800/600?random=2",
            "description": "Random photo (may contain text)"
        },
        {
            "name": "sample_objects.jpg",
            "url": "https://picsum.photos/800/600?random=3",
            "description": "Random photo (may contain objects)"
        }
    ]

    print("Downloading sample images from Lorem Picsum (random photos)...")
    print("Note: These are random photos and may not contain specific elements.\n")

    success_count = 0
    for sample in samples:
        print(f"{sample['description']}")
        if download_image(sample["url"], sample["name"], test_dir):
            success_count += 1
        print()

    print("=" * 70)
    print(f"Downloaded {success_count}/{len(samples)} images")
    print(f"Saved to: {test_dir}")
    print("=" * 70)

    print("\nRecommendation:")
    print("For better testing, add your own images with:")
    print("  - Photos with people (for face detection)")
    print("  - Documents or screenshots (for text detection)")
    print("  - Photos with objects like laptops, phones, cars (for object detection)")
    print(f"\nCopy your images to: {test_dir}")

    return success_count > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
