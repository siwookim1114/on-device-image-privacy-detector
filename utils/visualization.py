"""
Visualization & Export Utilities

Functions for exporting risk analysis results to JSON and generating
visual risk maps with color-coded bounding boxes by severity level.
"""

import json
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

from utils.models import (
    RiskAnalysisResult,
    RiskLevel,
    DetectionResults,
)


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
        "processing_time_ms": result.processimg_time_ms,
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
        # Identity fields (from Agent 2.5)
        "person_id": a.person_id,
        "person_label": a.person_label,
        "classification": classification,
        "consent_status": consent_status,
        "consent_confidence": a.consent_confidence,
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

        # Draw bounding box
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
