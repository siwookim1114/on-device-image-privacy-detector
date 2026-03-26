"""Baseline implementations for comparison."""
import time
from pathlib import Path
from typing import Optional
from .models import BenchmarkResults, ImageResult

def run_blur_all_baseline(dataset, output_dir) -> BenchmarkResults:
    """Detect all elements, blur everything. Maximum protection, high false positives."""
    from PIL import Image, ImageFilter

    results = BenchmarkResults(config_name="blur_all")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ann in dataset:
        t0 = time.perf_counter()
        img_path = dataset.resolve_image_path(ann)
        try:
            img = Image.open(img_path).convert("RGB")
            blurred = img.filter(ImageFilter.GaussianBlur(radius=15))
            save_path = out / f"{ann.image_id}_blurred.jpg"
            blurred.save(str(save_path))
            elapsed = (time.perf_counter() - t0) * 1000

            result = ImageResult(
                image_id=ann.image_id,
                config_name="blur_all",
                latency={"total_ms": elapsed},
            )
            results.image_results.append(result)
            results.successful_images += 1
        except Exception:
            results.image_results.append(ImageResult(image_id=ann.image_id, config_name="blur_all"))
        results.total_images += 1

    return results


def run_face_only_baseline(dataset, output_dir) -> BenchmarkResults:
    """MTCNN face detection + blur. Represents DeepPrivacy2/CIAGAN class."""
    results = BenchmarkResults(config_name="face_only")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from facenet_pytorch import MTCNN
        from PIL import Image, ImageFilter
        import numpy as np
    except ImportError:
        return results

    mtcnn = MTCNN(keep_all=True, device="cpu")

    for ann in dataset:
        t0 = time.perf_counter()
        img_path = dataset.resolve_image_path(ann)
        try:
            img = Image.open(img_path).convert("RGB")
            boxes, _ = mtcnn.detect(img)

            if boxes is not None:
                img_arr = np.array(img)
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face_crop = img.crop((x1, y1, x2, y2))
                    blurred_face = face_crop.filter(ImageFilter.GaussianBlur(radius=15))
                    img.paste(blurred_face, (x1, y1))

            save_path = out / f"{ann.image_id}_face_only.jpg"
            img.save(str(save_path))
            elapsed = (time.perf_counter() - t0) * 1000

            result = ImageResult(
                image_id=ann.image_id,
                config_name="face_only",
                latency={"total_ms": elapsed},
            )
            results.image_results.append(result)
            results.successful_images += 1
        except Exception:
            results.image_results.append(ImageResult(image_id=ann.image_id, config_name="face_only"))
        results.total_images += 1

    return results


def run_phase1_only_baseline(dataset, output_dir) -> BenchmarkResults:
    """Our pipeline with fallback_only=True (no VLM)."""
    from agents.pipeline import PipelineOrchestrator, PipelineConfig

    results = BenchmarkResults(config_name="phase1_only")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(fallback_only=True)
    orc = PipelineOrchestrator(config=config)

    try:
        for ann in dataset:
            img_path = dataset.resolve_image_path(ann)
            try:
                output = orc.run(img_path)
                latency = getattr(output, "phase_timings", {})
                if not latency:
                    latency = {"total_ms": output.total_time_ms}

                result = ImageResult(
                    image_id=ann.image_id,
                    config_name="phase1_only",
                    latency=latency,
                    pipeline_output=output,
                )
                results.image_results.append(result)
                if output.success:
                    results.successful_images += 1
            except Exception:
                results.image_results.append(ImageResult(image_id=ann.image_id, config_name="phase1_only"))
            results.total_images += 1
    finally:
        orc.close()

    return results


def get_baseline_runner(name: str):
    BASELINES = {
        "blur_all": run_blur_all_baseline,
        "face_only": run_face_only_baseline,
        "phase1_only": run_phase1_only_baseline,
    }
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINES.keys())}")
    return BASELINES[name]
