"""
Precise Segmentation using MobileSAM for Targeted Obfuscation

Element-type-aware segmentation:
  - Faces: SAM point prompt at face center -> precise face contour mask
  - Screens: SAM point prompt at estimated screen center -> display panel mask
  - Text: Skipped (rectangular bbox redaction is cleaner for text)

Only runs on elements that need protection (method != none).
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

from mobile_sam import sam_model_registry, SamPredictor

from utils.models import (
    ObfuscationMethod,
    ProtectionStrategy,
    RiskAssessment,
)

# Default model path
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "backend-engines" / "models" / "mobile_sam.pt"

# Screen device keywords
SCREEN_DEVICES = {"tv", "laptop", "cell phone", "monitor"}


def _is_screen_device(assessment: RiskAssessment) -> bool:
    """Check if an assessment refers to a screen device."""
    desc = assessment.element_description.lower()
    return any(device in desc for device in SCREEN_DEVICES)


def _get_device_type(description: str) -> str:
    """Extract device type from element description."""
    desc = description.lower()
    if "laptop" in desc:
        return "laptop"
    if "cell phone" in desc or "phone" in desc:
        return "cell_phone"
    # tv, monitor → treat as full-screen devices
    return "monitor"


class PrecisionSegmenter:
    """
    Precise segmentation using MobileSAM for targeted obfuscation.

    Element-type-aware: uses different prompt strategies for faces vs screens.
    Selective: only segments elements that need protection.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Load MobileSAM model and initialize predictor.

        Args:
            model_path: Path to mobile_sam.pt checkpoint
            device: Device to run on ("cpu", "mps", "cuda")
        """
        model_path = model_path or str(DEFAULT_MODEL_PATH)

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MobileSAM checkpoint not found: {model_path}\n"
                f"Download from: https://github.com/ChaoningZhang/MobileSAM\n"
                f"Place at: {DEFAULT_MODEL_PATH}"
            )

        sam = sam_model_registry["vit_t"](checkpoint=model_path)
        sam.to(device)
        sam.eval()

        self.predictor = SamPredictor(sam)
        self.device = device
        print(f"  PrecisionSegmenter ready (MobileSAM on {device})")

    def segment_face(
        self, image_np: np.ndarray, face_bbox: List[int]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Segment a face region using box + point prompt.

        The MTCNN face bbox constrains SAM to the face region (prevents
        segmenting the entire person). The smallest good-scoring mask
        is selected since the face contour is smaller than full-person.

        Args:
            image_np: Full image as numpy array (H, W, 3)
            face_bbox: [x, y, w, h] face bounding box

        Returns:
            mask: Binary mask (H, W) of face pixels
            refined_bbox: [x, y, w, h] tight bbox around the mask
        """
        x, y, w, h = face_bbox
        cx = x + w // 2
        cy = y + h // 2

        box = np.array([x, y, x + w, y + h])
        point_coords = np.array([[cx, cy]])
        point_labels = np.array([1])  # 1 = foreground

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )

        # Pick smallest mask among good-scoring candidates (face < person)
        min_score = max(scores) * 0.8
        best_idx = -1
        best_area = float("inf")
        for i, (m, s) in enumerate(zip(masks, scores)):
            if s >= min_score and m.sum() < best_area:
                best_area = m.sum()
                best_idx = i
        if best_idx < 0:
            best_idx = np.argmax(scores)

        mask = masks[best_idx].astype(np.uint8)

        refined_bbox = self._mask_to_bbox(mask)
        return mask, refined_bbox

    def segment_screen(
        self, image_np: np.ndarray, device_bbox: List[int], device_type: str = "monitor"
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Segment screen display panel from device bounding box.

        Uses a combined box + point prompt for better screen isolation:
          - Box prompt: YOLO device bbox gives SAM the rough region
          - Point prompt: center point guides SAM toward the screen surface
          - For laptops: point placed in upper portion (screen area)
          - For TV/monitor/phone: point placed at center (screen IS the device)

        Args:
            image_np: Full image as numpy array (H, W, 3)
            device_bbox: [x, y, w, h] device bounding box
            device_type: "laptop", "monitor", "cell_phone"

        Returns:
            mask: Binary mask (H, W) of screen pixels
            refined_bbox: [x, y, w, h] tight bbox around screen
        """
        x, y, w, h = device_bbox
        cx = x + w // 2

        if device_type == "laptop":
            # Constrain box to upper 60% of YOLO bbox (screen region only)
            screen_h = int(h * 0.6)
            box = np.array([x, y, x + w, y + screen_h])
            cy = y + screen_h // 2
        else:
            # TV, monitor, phone — screen is the whole device
            box = np.array([x, y, x + w, y + h])
            cy = y + h // 2

        point_coords = np.array([[cx, cy]])
        point_labels = np.array([1])

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )

        # For screens, pick the largest mask among good-scoring candidates
        # (the screen panel is typically the largest flat surface in the region)
        min_score = max(scores) * 0.8  # Allow masks within 80% of best score
        best_idx = -1
        best_area = 0
        for i, (m, s) in enumerate(zip(masks, scores)):
            if s >= min_score and m.sum() > best_area:
                best_area = m.sum()
                best_idx = i
        if best_idx < 0:
            best_idx = np.argmax(scores)

        mask = masks[best_idx].astype(np.uint8)

        refined_bbox = self._mask_to_bbox(mask)
        return mask, refined_bbox

    def process_strategies(
        self,
        image_path: str,
        strategies: List[ProtectionStrategy],
        risk_assessments: List[RiskAssessment],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Run selective SAM segmentation on strategies that need protection.

        Only processes:
          - Faces with method != none
          - Screen devices with method != none
        Skips text (bbox is better for text redaction).

        Args:
            image_path: Path to the original image
            strategies: List of ProtectionStrategy from Agent 3
            risk_assessments: List of RiskAssessment from Agent 2
            output_dir: Optional directory to save mask images

        Returns:
            Dict mapping detection_id -> {"mask": np.ndarray, "refined_bbox": [x,y,w,h]}
        """
        image_np = np.array(Image.open(image_path).convert("RGB"))
        self.predictor.set_image(image_np)  # Set once, reuse for all prompts

        # Build lookup from detection_id to assessment
        assessment_map = {a.detection_id: a for a in risk_assessments}

        results = {}
        segmented_count = 0

        for strategy in strategies:
            # Skip items that don't need protection
            if strategy.recommended_method is None or strategy.recommended_method == ObfuscationMethod.NONE:
                continue

            assessment = assessment_map.get(strategy.detection_id)
            if assessment is None:
                continue

            bbox = assessment.bbox.to_list()

            if assessment.element_type == "face":
                mask, refined_bbox = self.segment_face(image_np, bbox)
                results[strategy.detection_id] = {
                    "mask": mask,
                    "refined_bbox": refined_bbox,
                    "element_type": "face",
                }
                segmented_count += 1
                print(f"    SAM face: {assessment.element_description} -> mask {mask.sum()} px")

            elif assessment.element_type == "object" and _is_screen_device(assessment):
                device_type = _get_device_type(assessment.element_description)
                mask, refined_bbox = self.segment_screen(image_np, bbox, device_type)
                results[strategy.detection_id] = {
                    "mask": mask,
                    "refined_bbox": refined_bbox,
                    "element_type": "screen",
                }
                segmented_count += 1
                print(f"    SAM screen ({device_type}): {assessment.element_description} -> mask {mask.sum()} px")

            # Text elements: skip SAM (bbox redaction is cleaner)

        # Save masks to disk if output_dir provided
        if output_dir and results:
            mask_dir = Path(output_dir) / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)

            for det_id, data in results.items():
                mask_path = mask_dir / f"{det_id}_mask.npy"
                np.save(str(mask_path), data["mask"])
                data["mask_path"] = str(mask_path)

        print(f"  SAM segmentation complete: {segmented_count} elements segmented")
        return results

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> List[int]:
        """Convert a binary mask to a tight [x, y, w, h] bounding box."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any():
            return [0, 0, 0, 0]

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
