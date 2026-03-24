"""
Text detection backends with a common interface.

Abstracts text detection behind a protocol so the pipeline can swap
between EasyOCR (CRAFT) and docTR (DBNet) via config. Recognition
always uses EasyOCR's CRNN.

Usage:
    backend = EasyOCRBackend()          # Current behavior
    backend = DBNetEasyOCRBackend()     # Tier 2: tight polygons
    regions = backend.detect(img_array)
    for r in regions:
        print(r.polygon, r.bbox, r.text, r.confidence)
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class TextRegion:
    """A detected text region with polygon, bbox, text content, and confidence."""
    polygon: List[List[int]]   # N-point polygon [[x,y], ...]
    bbox: List[int]            # [x, y, w, h] axis-aligned
    text: str                  # Recognized text content
    confidence: float          # Recognition confidence 0-1


class EasyOCRBackend:
    """Current EasyOCR (CRAFT detect + CRNN recognize) wrapped as a backend."""

    def __init__(self, gpu: bool = False):
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def detect(self, image: np.ndarray, **kwargs) -> List[TextRegion]:
        """Run EasyOCR readtext and return TextRegion list."""
        results = self.reader.readtext(
            image, width_ths=0.7, low_text=0.3, add_margin=0.05, paragraph=False
        )
        regions = []
        for bbox_points, text, confidence in results:
            if confidence < 0.3:
                continue
            x_coords = [p[0] for p in bbox_points]
            y_coords = [p[1] for p in bbox_points]
            bbox = [
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords) - min(x_coords)),
                int(max(y_coords) - min(y_coords)),
            ]
            polygon = [[int(p[0]), int(p[1])] for p in bbox_points]
            regions.append(TextRegion(
                polygon=polygon, bbox=bbox,
                text=text, confidence=float(confidence),
            ))
        return regions


class DBNetEasyOCRBackend:
    """DBNet detection (docTR) + EasyOCR recognition on crops.

    docTR's DBNet produces tight shrink-polygons with built-in instance
    separation. Each detected region is cropped and fed to EasyOCR's
    CRNN recognizer for text content extraction.

    Install: pip install python-doctr[torch]
    """

    def __init__(self, gpu: bool = False, dbnet_model: str = "db_resnet50"):
        import easyocr
        import torch
        from doctr.models import detection_predictor

        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

        # Load docTR detection model
        self.det_model = detection_predictor(dbnet_model, pretrained=True)

        # Move to MPS if available
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        self.device = torch.device("mps") if mps_available else torch.device("cpu")
        self.det_model.model = self.det_model.model.to(self.device)

        print(f"  DBNet backend: {dbnet_model} on {self.device}")

    def detect(self, image: np.ndarray, **kwargs) -> List[TextRegion]:
        """Run DBNet detection + EasyOCR recognition on crops.

        docTR detection_predictor returns list of dicts per image, each with
        'words' key: ndarray shape (N, 5) = [x_min, y_min, x_max, y_max, conf]
        in normalized [0,1] coordinates.
        """
        h, w = image.shape[:2]

        # DBNet detection: input is list of numpy arrays
        result = self.det_model([image])

        # result is a list (one dict per image)
        if not result:
            return []

        page = result[0]  # first image
        words = page.get("words", np.array([]))
        if len(words) == 0:
            return []

        regions = []
        for detection in words:
            # Each detection: [x_min, y_min, x_max, y_max, confidence] normalized
            rx1, ry1, rx2, ry2, det_conf = detection
            x1, y1 = int(rx1 * w), int(ry1 * h)
            x2, y2 = int(rx2 * w), int(ry2 * h)
            bw, bh = x2 - x1, y2 - y1
            if bw < 5 or bh < 5:
                continue

            # Crop with margin for EasyOCR recognition
            pad = 5
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)
            crop = image[cy1:cy2, cx1:cx2]

            # EasyOCR recognition on crop
            try:
                ocr_results = self.reader.readtext(
                    crop, width_ths=0.9, paragraph=False
                )
            except Exception:
                continue

            if not ocr_results:
                continue

            # Take best result from crop
            best = max(ocr_results, key=lambda r: r[2])
            text = best[1]
            conf = float(best[2])
            if conf < 0.25 or len(text.strip()) < 1:
                continue

            bbox = [x1, y1, bw, bh]
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            regions.append(TextRegion(
                polygon=polygon, bbox=bbox,
                text=text, confidence=conf,
            ))

        return regions
