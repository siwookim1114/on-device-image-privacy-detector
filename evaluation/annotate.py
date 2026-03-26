"""
Annotation helper: pre-fills ground truth from pipeline predictions.

Usage:
    conda run -n lab_env python -m evaluation.annotate --images DIR --output DIR [--preview]

Workflow:
    1. Run this tool to generate pre-filled annotations (Phase 1 only, no VLM).
    2. Manually review/correct the JSON files:
         - Verify or fix severity_gt (critical/high/medium/low)
         - Verify or flip should_protect (true/false)
         - Correct element_type if the detector mislabelled something
         - Add text_content for text elements if useful for audit
    3. Run benchmark:
         python -m evaluation.run_benchmark --dataset evaluation/data/benchmark_v1

Output layout:
    <output_dir>/
        annotations/<image_id>.json   -- one annotation file per image
        previews/<image_id>_preview.jpg  -- bbox preview (only with --preview)
        manifest.json                 -- dataset manifest (all images)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path when running as __main__ or as a module.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_images(
    image_dir: str,
    output_dir: str,
    preview: bool = False,
    dataset_name: str = "benchmark_v1",
) -> None:
    """
    Run Phase 1 detection + risk assessment on every image in *image_dir*,
    write one annotation JSON per image to *output_dir*/annotations/, and
    write a manifest.json aggregating all annotations.

    Args:
        image_dir:    Directory containing source images.
        output_dir:   Root directory for generated artefacts.
        preview:      When True, also draw coloured bbox previews.
        dataset_name: Dataset name embedded in manifest.json.
    """
    from agents.pipeline import PipelineConfig, PipelineOrchestrator

    image_dir_path = Path(image_dir).resolve()
    output_dir_path = Path(output_dir).resolve()
    ann_dir = output_dir_path / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Collect images (deterministic order across runs).
    _SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images: List[Path] = sorted(
        f for f in image_dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in _SUPPORTED_SUFFIXES
    )
    if not images:
        print(f"No images found in {image_dir_path}")
        return
    print(f"Found {len(images)} image(s) in {image_dir_path}")

    # Phase 1 only — no VLM calls, fast (~1-3 s per image on Apple Silicon).
    config = PipelineConfig(
        fallback_only=True,
        enable_consent=False,   # Skip MongoDB lookup; consent=unclear by default.
        enable_sam=False,       # SAM not needed for annotation.
    )
    orc = PipelineOrchestrator(config=config)

    manifest_images: List[Dict[str, Any]] = []

    try:
        for img_path in images:
            image_id = img_path.stem
            print(f"\nProcessing {image_id}...")

            t0 = time.perf_counter()
            try:
                output = orc.run(str(img_path))
            except Exception as exc:
                print(f"  ERROR: {exc}")
                continue
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

            elements = _extract_elements(output)

            annotation = _build_annotation(
                image_id=image_id,
                img_path=img_path,
                elements=elements,
                elapsed_ms=elapsed_ms,
            )

            ann_path = ann_dir / f"{image_id}.json"
            ann_path.write_text(json.dumps(annotation, indent=2, default=str))

            face_count   = sum(1 for e in elements if e["element_type"] == "face")
            text_count   = sum(1 for e in elements if e["element_type"] == "text")
            screen_count = sum(1 for e in elements if e["element_type"] == "screen")
            print(
                f"  Saved: {ann_path} "
                f"({len(elements)} elements — "
                f"{face_count} face, {text_count} text, {screen_count} screen, "
                f"category={annotation['metadata']['category']})"
            )

            manifest_images.append(annotation)

            if preview:
                preview_path = output_dir_path / "previews" / f"{image_id}_preview.jpg"
                _draw_preview(str(img_path), elements, str(preview_path))
    finally:
        orc.close()

    # Write dataset manifest.
    manifest: Dict[str, Any] = {
        "name": dataset_name,
        "version": "1.0",
        "description": (
            f"Auto-annotated from {image_dir_path}. "
            "NEEDS MANUAL REVIEW: verify severity_gt and should_protect for each element."
        ),
        "images": manifest_images,
    }
    manifest_path = output_dir_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print(f"Annotated {len(manifest_images)} / {len(images)} image(s).")
    print(f"Annotations : {ann_dir}")
    print(f"Manifest    : {manifest_path}")
    if preview:
        print(f"Previews    : {output_dir_path / 'previews'}")
    print(
        "IMPORTANT: Review and correct annotations before running the benchmark.\n"
        "  Fields to check: severity_gt, should_protect, element_type, text_content."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_elements(output) -> List[Dict[str, Any]]:
    """
    Convert PipelineOutput.risk_analysis into a flat list of element dicts
    aligned with the GroundTruthElement schema in evaluation/models.py.
    """
    elements: List[Dict[str, Any]] = []

    if not output.risk_analysis:
        return elements

    for i, assess in enumerate(output.risk_analysis.risk_assessments):
        # --- bbox -----------------------------------------------------------
        bbox = assess.bbox
        if hasattr(bbox, "x"):
            bbox_list: List[int] = [bbox.x, bbox.y, bbox.width, bbox.height]
        else:
            bbox_list = [int(v) for v in bbox]

        # --- element_type ---------------------------------------------------
        # screen_state is set only for screen devices; collapse object→screen.
        screen_state: Optional[str] = getattr(assess, "screen_state", None)
        etype: str = assess.element_type
        if screen_state is not None:
            etype = "screen"

        # --- severity -------------------------------------------------------
        severity: str = (
            assess.severity.value
            if hasattr(assess.severity, "value")
            else str(assess.severity)
        ).lower()

        # --- should_protect -------------------------------------------------
        should_protect: bool = bool(assess.requires_protection)

        # --- consent --------------------------------------------------------
        consent: Optional[str] = None
        if getattr(assess, "consent_status", None):
            consent = (
                assess.consent_status.value
                if hasattr(assess.consent_status, "value")
                else str(assess.consent_status)
            )

        # --- element id (stable, human-readable) ----------------------------
        element_id = f"{etype}_{i + 1:03d}"

        elem: Dict[str, Any] = {
            "id": element_id,
            "element_type": etype,
            "bbox": bbox_list,
            "severity_gt": severity,
            "should_protect": should_protect,
        }

        # Optional fields — only emit when meaningful to keep JSON tidy.
        if consent:
            elem["consent_status"] = consent
        person_label = getattr(assess, "person_label", None)
        if person_label:
            elem["person_label"] = person_label
        if screen_state:
            elem["screen_state"] = screen_state

        # For text elements, surface element_description as notes (contains
        # the OCR text).  The text_type lives in the description or factors
        # but is not a direct field on RiskAssessment; record it via notes.
        if etype == "text":
            description = getattr(assess, "element_description", None)
            if description:
                elem["notes"] = description

        elements.append(elem)

    return elements


def _build_annotation(
    image_id: str,
    img_path: Path,
    elements: List[Dict[str, Any]],
    elapsed_ms: float,
) -> Dict[str, Any]:
    """Assemble the full annotation dict for one image."""
    face_count   = sum(1 for e in elements if e["element_type"] == "face")
    text_count   = sum(1 for e in elements if e["element_type"] == "text")
    screen_count = sum(1 for e in elements if e["element_type"] == "screen")

    # Derive a coarse category from element counts.
    if face_count > 0 and text_count > 0:
        category = "mixed"
    elif face_count > 0 and face_count >= text_count:
        category = "face_dominant"
    elif text_count > 0:
        category = "text_dominant"
    elif screen_count > 0:
        category = "screen_dominant"
    else:
        category = "other"

    return {
        "image_id": image_id,
        "image_path": str(img_path),
        "scene_description": (
            f"Auto-annotated: {face_count} face(s), "
            f"{text_count} text region(s), {screen_count} screen(s)"
        ),
        "elements": elements,
        "metadata": {
            "category": category,
            "source": "auto_annotated",
            "pipeline_time_ms": elapsed_ms,
            "needs_review": True,
        },
    }


def _draw_preview(
    image_path: str,
    elements: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Draw coloured bounding boxes on a copy of the source image and save it.

    Severity colour mapping:
        critical → red
        high     → orange
        medium   → yellow
        low      → green
        (other)  → white
    """
    try:
        from PIL import Image, ImageDraw

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        _SEVERITY_COLORS: Dict[str, str] = {
            "critical": "red",
            "high":     "orange",
            "medium":   "yellow",
            "low":      "green",
        }

        for elem in elements:
            x, y, w, h = elem["bbox"]
            color = _SEVERITY_COLORS.get(elem["severity_gt"], "white")
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            label = f"{elem['element_type']} [{elem['severity_gt']}]"
            # Place label above the box; clamp to image top.
            label_y = max(0, y - 15)
            draw.text((x, label_y), label, fill=color)

        img.save(output_path, quality=90)
        print(f"  Preview: {output_path}")
    except Exception as exc:
        print(f"  Preview failed: {exc}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-fill benchmark ground truth annotations from pipeline predictions. "
            "Runs Phase 1 (detection + deterministic risk assessment) only — no VLM calls."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  conda run -n lab_env python -m evaluation.annotate \\\n"
            "      --images data/test_images/ \\\n"
            "      --output evaluation/data/benchmark_v1 \\\n"
            "      --preview\n"
        ),
    )
    parser.add_argument(
        "--images",
        required=True,
        metavar="DIR",
        help="Directory containing source images (.jpg/.jpeg/.png/.bmp/.webp).",
    )
    parser.add_argument(
        "--output",
        default="evaluation/data/benchmark_v1",
        metavar="DIR",
        help="Output root directory for annotations and manifest (default: evaluation/data/benchmark_v1).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate annotated bbox preview images under <output>/previews/.",
    )
    parser.add_argument(
        "--name",
        default="benchmark_v1",
        metavar="NAME",
        help="Dataset name written into manifest.json (default: benchmark_v1).",
    )

    args = parser.parse_args()
    annotate_images(
        image_dir=args.images,
        output_dir=args.output,
        preview=args.preview,
        dataset_name=args.name,
    )


if __name__ == "__main__":
    main()
