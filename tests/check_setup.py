"""
Quick setup checker for testing the Detection Agent
Run this before running the actual tests to verify everything is configured correctly
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_setup():
    """Check if everything is set up correctly for testing"""
    print("=" * 70)
    print("Detection Agent Setup Checker")
    print("=" * 70 + "\n")

    all_checks_passed = True

    # Check 1: Configuration file
    print("1. Checking configuration file...")
    config_path = project_root / "configs" / "config.yaml"
    if config_path.exists():
        print("   ✓ configs/config.yaml found")
    else:
        print("   ✗ configs/config.yaml NOT found")
        all_checks_passed = False

    # Check 2: Required Python modules
    print("\n2. Checking Python dependencies...")
    required_modules = [
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("yaml", "PyYAML"),
        ("langchain", "LangChain"),
    ]

    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"   ✓ {display_name} installed")
        except ImportError:
            print(f"   ✗ {display_name} NOT installed")
            all_checks_passed = False

    # Check 3: Project structure
    print("\n3. Checking project structure...")
    required_dirs = [
        "agents",
        "utils",
        "configs",
        "data/test_images",
        "data/test_results",
    ]

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"   ✓ {dir_path}/ exists")
        else:
            print(f"   ✗ {dir_path}/ NOT found")
            if dir_path.startswith("data/"):
                print(f"      Run: mkdir -p {full_path}")
            all_checks_passed = False

    # Check 4: Required files
    print("\n4. Checking required files...")
    required_files = [
        "utils/config.py",
        "utils/models.py",
        "agents/detection_agent.py",
        "agents/tools.py",
        "agents/local_wrapper.py",
    ]

    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✓ {file_path} exists")
        else:
            print(f"   ✗ {file_path} NOT found")
            all_checks_passed = False

    # Check 5: Test images
    print("\n5. Checking test images...")
    test_images_dir = project_root / "data" / "test_images"
    if test_images_dir.exists():
        image_files = list(test_images_dir.glob("*.jpg")) + \
                     list(test_images_dir.glob("*.jpeg")) + \
                     list(test_images_dir.glob("*.png")) + \
                     list(test_images_dir.glob("*.webp"))

        if image_files:
            print(f"   ✓ Found {len(image_files)} test image(s):")
            for img in image_files[:5]:  # Show first 5
                print(f"      - {img.name}")
            if len(image_files) > 5:
                print(f"      ... and {len(image_files) - 5} more")
        else:
            print(f"   ⚠ No test images found in data/test_images/")
            print(f"      Add some test images to get started:")
            print(f"      cp /path/to/your/images/*.jpg data/test_images/")
    else:
        print(f"   ✗ data/test_images/ directory NOT found")
        all_checks_passed = False

    # Check 6: PyTorch device availability
    print("\n6. Checking compute devices...")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"   ✓ CUDA available")
            print(f"      GPU: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"      Memory: {memory_gb:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"   ✓ MPS (Apple Silicon) available")
        else:
            print(f"   ⚠ Using CPU (GPU not available)")

    except Exception as e:
        print(f"   ✗ Error checking PyTorch: {e}")
        all_checks_passed = False

    # Check 7: Try loading configuration
    print("\n7. Testing configuration loading...")
    try:
        from utils.config import load_config
        config = load_config()
        print(f"   ✓ Configuration loaded successfully")
        print(f"      Device: {config.system.device}")
        print(f"      Face detector: {config.models.detection.face_detector}")
        print(f"      Text detector: {config.models.detection.text_detector}")
        print(f"      Object detector: {config.models.detection.object_detector}")
    except Exception as e:
        print(f"   ✗ Failed to load configuration: {e}")
        all_checks_passed = False

    # Final summary
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("✓ All checks passed! You're ready to run tests.")
        print("\nRun the test with:")
        print("  python tests/test_detection_agent.py")
    else:
        print("✗ Some checks failed. Please fix the issues above before testing.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Create test images directory: mkdir -p data/test_images")
        print("  3. Add test images: cp /path/to/images/*.jpg data/test_images/")
    print("=" * 70)

    return all_checks_passed


if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)
