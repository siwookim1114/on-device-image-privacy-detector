"""
Visualization & Export Utilities

Functions for exporting risk analysis results to JSON and generating
visual risk maps with color-coded bounding boxes by severity level.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from utils.models import (
    ObfuscationMethod,
    RiskAnalysisResult,
    RiskLevel,
    DetectionResults,
    StrategyRecommendations,
)
from utils.avatar import apply_avatar_bbox, apply_avatar_mask, apply_silhouette_bbox, apply_silhouette_mask


def export_risk_results_json(
    result: RiskAnalysisResult,
    detections: Optional[DetectionResults] = None,
    output_path: Optional[str] = None,
) -> dict:
    """
    Export risk analysis results (Agent 2 + Agent 2.5) as JSON.

    Serializes all RiskAssessment fields including identity fields
    from the Consent Identity Agent (person_id, person_label,
    classification, consent_status).

    Args:
        result: RiskAnalysisResult from the pipeline
        detections: Optional DetectionResults for detection summary
        output_path: Optional file path to save JSON

    Returns:
        Serialized dict of all risk analysis data
    """
    results = {
        "image_path": result.image_path,
        "overall_risk_level": result.overall_risk_level.value,
        "processing_time_ms": result.processing_time_ms,
        "total_assessments": len(result.risk_assessments),
        "confirmed_risks": result.confirmed_risks,
        "faces_pending_identity": result.faces_pending_identity,
        "assessments": [
            _serialize_assessment(a) for a in result.risk_assessments
        ],
    }

    if detections is not None:
        results["detection_summary"] = {
            "faces": len(detections.faces),
            "text_regions": len(detections.text_regions),
            "objects": len(detections.objects),
            "total": detections.total_detections,
        }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Risk results saved: {output_path}")

    return results


def _serialize_assessment(a) -> dict:
    """Serialize a single RiskAssessment to a JSON-safe dict."""
    classification = a.classification
    if hasattr(classification, "value"):
        classification = classification.value

    consent_status = a.consent_status
    if hasattr(consent_status, "value"):
        consent_status = consent_status.value

    return {
        "detection_id": a.detection_id,
        "element_type": a.element_type,
        "element_description": a.element_description,
        "risk_type": a.risk_type.value,
        "severity": a.severity.value,
        "color_code": a.color_code,
        "reasoning": a.reasoning,
        "user_sensitivity_applied": a.user_sensitivity_applied,
        "bbox": a.bbox.to_list(),
        "requires_protection": a.requires_protection,
        "legal_requirement": a.legal_requirement,
        # Screen-only bbox for screen devices (Agent 4 uses this instead of full device bbox)
        "screen_bbox": a.screen_bbox.to_list() if a.screen_bbox else None,
        # Phase 1.5a screen state verification
        "screen_state": a.screen_state,
        # Identity fields (from Agent 2.5)
        "person_id": a.person_id,
        "person_label": a.person_label,
        "classification": classification,
        "consent_status": consent_status,
        "consent_confidence": a.consent_confidence,
        "text_polygon": a.text_polygon,
    }


def generate_risk_map(
    result: RiskAnalysisResult,
    image_path: str,
    output_path: Optional[str] = None,
) -> Image.Image:
    """
    Generate a visual risk map with color-coded bounding boxes by severity.

    Colors by risk severity level (from config):
        - Critical: #FF0000 (red), 3px border
        - High:     #FF6600 (orange), 3px border
        - Medium:   #FFD700 (gold), 2px border
        - Low:      #90EE90 (light green), 2px border

    Labels show element description, severity, and identity info for faces.
    Labels are placed with collision avoidance to prevent overlapping.

    Args:
        result: RiskAnalysisResult with assessments to visualize
        image_path: Path to original image
        output_path: Optional file path to save the risk map

    Returns:
        PIL Image with risk-level color-coded bounding boxes
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    # Scale font size relative to image height
    font_size = max(10, min(14, img_h // 100))

    # Font setup with platform-specific fallbacks
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", font_size
        )
    except Exception:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except Exception:
            font = ImageFont.load_default()

    # Sort assessments so critical renders on top (draw low first, critical last)
    severity_order = {
        RiskLevel.LOW: 0,
        RiskLevel.MEDIUM: 1,
        RiskLevel.HIGH: 2,
        RiskLevel.CRITICAL: 3,
    }
    sorted_assessments = sorted(
        result.risk_assessments,
        key=lambda a: severity_order.get(a.severity, 0),
    )

    # Track occupied label regions for collision avoidance
    occupied_regions = []

    def _regions_overlap(r1, r2):
        """Check if two (x1, y1, x2, y2) regions overlap."""
        return not (r1[2] <= r2[0] or r2[2] <= r1[0] or r1[3] <= r2[1] or r2[3] <= r1[1])

    def _find_label_position(bbox_x1, bbox_y1, bbox_x2, bbox_y2, lw, lh):
        """Find a non-overlapping label position: try above, below, then right."""
        candidates = [
            # Above the bounding box
            (max(0, bbox_x1), max(0, bbox_y1 - lh - 2)),
            # Below the bounding box
            (max(0, bbox_x1), min(img_h - lh, bbox_y2 + 2)),
            # Right side of the bounding box
            (min(img_w - lw, bbox_x2 + 4), max(0, bbox_y1)),
        ]
        for lx, ly in candidates:
            # Clamp to image bounds
            lx = max(0, min(lx, img_w - lw))
            ly = max(0, min(ly, img_h - lh))
            candidate_region = (lx, ly, lx + lw, ly + lh)
            if not any(_regions_overlap(candidate_region, occ) for occ in occupied_regions):
                return lx, ly
        # Fallback: use first candidate (above), clamped
        lx = max(0, min(bbox_x1, img_w - lw))
        ly = max(0, min(bbox_y1 - lh - 2, img_h - lh))
        return lx, ly

    for assessment in sorted_assessments:
        color = assessment.color_code
        line_width = (
            3
            if assessment.severity in (RiskLevel.CRITICAL, RiskLevel.HIGH)
            else 2
        )

        # Draw bounding box (always use full detection bbox for risk map)
        x1, y1, x2, y2 = assessment.bbox.to_xyxy()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Build label text
        severity_tag = assessment.severity.value.upper()

        if assessment.element_type == "face":
            # Face labels: prioritize identity info over generic description
            identity_parts = []
            if assessment.person_label:
                identity_parts.append(assessment.person_label)
            else:
                cls = assessment.classification
                if cls:
                    if hasattr(cls, "value"):
                        cls = cls.value
                    identity_parts.append(cls)
                else:
                    identity_parts.append("unknown")
            if assessment.consent_status:
                consent = assessment.consent_status
                if hasattr(consent, "value"):
                    consent = consent.value
                identity_parts.append(consent)
            label = f"Face [{severity_tag}] ({', '.join(identity_parts)})"
        else:
            label = f"{assessment.element_description} [{severity_tag}]"

        # Truncate long labels (faces already compact, mainly for text descriptions)
        if len(label) > 40:
            label = label[:37] + "..."

        # Measure label size
        text_bbox = draw.textbbox((0, 0), label, font=font)
        label_w = text_bbox[2] - text_bbox[0]
        label_h = text_bbox[3] - text_bbox[1]

        # Find collision-free label position
        label_x, label_y = _find_label_position(x1, y1, x2, y2, label_w, label_h)

        # Draw label
        draw.text((label_x, label_y), label, fill=color, font=font)

        # Register occupied region
        occupied_regions.append((label_x, label_y, label_x + label_w, label_y + label_h))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"  Risk map saved: {output_path}")

    return image


def export_strategy_results_json(
    result: StrategyRecommendations,
    output_path: Optional[str] = None,
) -> dict:
    """
    Export strategy recommendations as JSON.

    Args:
        result: StrategyRecommendations from the Strategy Agent
        output_path: Optional file path to save JSON

    Returns:
        Serialized dict of all strategy data
    """
    results = {
        "image_path": result.image_path,
        "total_strategies": len(result.strategies),
        "total_protections_recommended": result.total_protections_recommended,
        "requires_user_confirmation": result.requires_user_confirmation,
        "estimated_processing_time_ms": result.estimated_processing_time_ms,
        "strategies": [
            _serialize_strategy(s) for s in result.strategies
        ],
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Strategy results saved: {output_path}")

    return results


def _serialize_strategy(s) -> dict:
    """Serialize a single ProtectionStrategy to a JSON-safe dict."""
    return {
        "detection_id": s.detection_id,
        "element": s.element,
        "severity": s.severity.value if hasattr(s.severity, "value") else s.severity,
        "recommended_action": s.recommended_action,
        "recommended_method": s.recommended_method.value if s.recommended_method else None,
        "parameters": s.parameters,
        "reasoning": s.reasoning,
        "alternative_options": [
            {
                "method": a.method.value if hasattr(a.method, "value") else a.method,
                "parameters": a.parameters,
                "reasoning": a.reasoning,
                "score": a.score,
            }
            for a in s.alternative_options
        ],
        "ethical_compliance": s.ethical_compliance,
        "execution_priority": s.execution_priority,
        "optional": s.optional,
        "requires_user_decision": s.requires_user_decision,
        "user_can_override": s.user_can_override,
        "segmentation_mask_path": s.segmentation_mask_path,
    }


def generate_protection_preview(
    image_path: str,
    strategies: StrategyRecommendations,
    risk_result: RiskAnalysisResult,
    seg_results: Dict[str, Dict],
    output_path: Optional[str] = None,
) -> Image.Image:
    """
    Generate a side-by-side comparison: bbox obfuscation vs SAM mask obfuscation.

    Left half:  Rectangular bbox-based blur/pixelate/overlay (traditional approach)
    Right half: SAM mask-based obfuscation (precise, natural-looking)

    Only elements with method != none are shown. Text elements use bbox on both
    sides (SAM is not applied to text).

    Args:
        image_path: Path to the original image
        strategies: StrategyRecommendations from Agent 3
        risk_result: RiskAnalysisResult from Agent 2
        seg_results: SAM results dict from PrecisionSegmenter.process_strategies()
        output_path: Optional file path to save the preview

    Returns:
        PIL Image with side-by-side comparison
    """
    original = Image.open(image_path).convert("RGB")
    img_w, img_h = original.size

    # Create two copies
    bbox_img = original.copy()
    sam_img = original.copy()

    # Build assessment lookup
    assessment_map = {a.detection_id: a for a in risk_result.risk_assessments}

    for strategy in strategies.strategies:
        method = strategy.recommended_method
        if method is None or method == ObfuscationMethod.NONE:
            continue

        assessment = assessment_map.get(strategy.detection_id)
        if assessment is None:
            continue

        bbox = assessment.bbox.to_list()  # [x, y, w, h]
        x, y, w, h = bbox

        # --- Apply bbox-based obfuscation (left side) ---
        _apply_obfuscation_bbox(bbox_img, x, y, w, h, method, strategy.parameters)

        # --- Apply SAM mask-based obfuscation (right side) ---
        if strategy.detection_id in seg_results:
            mask = seg_results[strategy.detection_id]["mask"]
            _apply_obfuscation_mask(sam_img, mask, method, strategy.parameters)
        else:
            # No SAM mask (text elements) → use bbox on both sides
            _apply_obfuscation_bbox(sam_img, x, y, w, h, method, strategy.parameters)

    # Create side-by-side canvas
    gap = 4
    canvas_w = img_w * 2 + gap
    canvas = Image.new("RGB", (canvas_w, img_h + 30), (30, 30, 30))

    canvas.paste(bbox_img, (0, 30))
    canvas.paste(sam_img, (img_w + gap, 30))

    # Add labels
    draw = ImageDraw.Draw(canvas)
    font_size = max(12, min(18, img_h // 60))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    draw.text((img_w // 2 - 60, 6), "BBOX (traditional)", fill=(255, 100, 100), font=font)
    draw.text((img_w + gap + img_w // 2 - 60, 6), "SAM (precise)", fill=(100, 255, 100), font=font)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
        print(f"  Protection preview saved: {output_path}")

    return canvas


def _apply_obfuscation_bbox(
    image: Image.Image, x: int, y: int, w: int, h: int,
    method: ObfuscationMethod, params: dict, padding: int = 0,
):
    """Apply obfuscation to a rectangular bbox region.
    Optional padding expands the region (used for closing gaps between text fragments)."""
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(image.width, x + w + padding), min(image.height, y + h + padding)
    if x2 <= x1 or y2 <= y1:
        return

    region = image.crop((x1, y1, x2, y2))

    if method == ObfuscationMethod.BLUR:
        kernel = params.get("kernel_size", 25)
        region = region.filter(ImageFilter.GaussianBlur(radius=kernel // 2))
    elif method == ObfuscationMethod.PIXELATE:
        block = params.get("block_size", 16)
        small_w = max(1, (x2 - x1) // block)
        small_h = max(1, (y2 - y1) // block)
        region = region.resize((small_w, small_h), Image.NEAREST)
        region = region.resize((x2 - x1, y2 - y1), Image.NEAREST)
    elif method == ObfuscationMethod.SOLID_OVERLAY:
        color = params.get("color", "black")
        if color == "black":
            color = (0, 0, 0)
        elif isinstance(color, str):
            color = (0, 0, 0)
        region = Image.new("RGB", (x2 - x1, y2 - y1), color)
    elif method == ObfuscationMethod.AVATAR_REPLACE:
        apply_avatar_bbox(image, x1, y1, x2, y2, params)
        return
    elif method == ObfuscationMethod.SILHOUETTE:
        apply_silhouette_bbox(image, x1, y1, x2, y2, params)
        return

    image.paste(region, (x1, y1))


def _apply_obfuscation_mask(
    image: Image.Image, mask: np.ndarray,
    method: ObfuscationMethod, params: dict,
):
    """Apply obfuscation only to pixels where mask > 0."""
    img_np = np.array(image)

    if method == ObfuscationMethod.BLUR:
        kernel = params.get("kernel_size", 25)
        blurred = np.array(image.filter(ImageFilter.GaussianBlur(radius=kernel // 2)))
        img_np[mask > 0] = blurred[mask > 0]
    elif method == ObfuscationMethod.PIXELATE:
        block = params.get("block_size", 16)
        h, w = img_np.shape[:2]
        small = image.resize((max(1, w // block), max(1, h // block)), Image.NEAREST)
        pixelated = np.array(small.resize((w, h), Image.NEAREST))
        img_np[mask > 0] = pixelated[mask > 0]
    elif method == ObfuscationMethod.SOLID_OVERLAY:
        color = params.get("color", "black")
        if color == "black" or isinstance(color, str):
            color = [0, 0, 0]
        img_np[mask > 0] = color
    elif method == ObfuscationMethod.AVATAR_REPLACE:
        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            img_np = apply_avatar_mask(img_np, mask, bbox, params)
    elif method == ObfuscationMethod.SILHOUETTE:
        img_np = apply_silhouette_mask(img_np, mask, params)

    image.paste(Image.fromarray(img_np))
