# Import required utils
import torch
from PIL import Image
from typing import Any, List, Dict, Optional, ClassVar, Set, Tuple, Type, Union
import json
import numpy as np
import re
# Detector Libraries

from facenet_pytorch import MTCNN     # Face Detection
from facenet_pytorch import InceptionResnetV1  # Face Embeddings (for Future Consent Agent)
import easyocr
from ultralytics import YOLO         # Object Detection (YOLO)
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from utils.models import (
    BoundingBox,
)
from agents.tools.common import _parse_tool_input
from agents.tools.text_backends import DBNetEasyOCRBackend
from agents.tools.text_backends import EasyOCRBackend


class FaceDetectionTool(BaseTool):
    """
    Tool for detecting faces in images using MTCNN + Getting embeddings using FaceNet for identity matching

    Tool is called by the Detection Agent to find all faces in an image.
    Embeddings will be used by Consent Agent for face recognition.
    """
    name: str = "detect_faces"
    description: str = (
        "Detects human faces in the image. "
        "Returns list of face locations with confidence scores, and embeddings for identity matching. "
        "Use this tool when you need to detect and locate all faces in an image during the detection phase."
    )
    detector: Any = None
    embedding_model: Any = None
    device: torch.device = None
    config: Any = None

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.device = torch.device(config.system.device)
        self._init_detector()

    def _init_detector(self):
        """
        Initialize MTCNN face detector and FaceNet embedding model.

        MTCNN detects faces, FaceNet generates 512-dimensional embeddings for face recognition.
        """
        try:
            # Use CPU for MTCNN - MPS has compatibility issues with adaptive pooling
            detector_device = torch.device("cpu") if self.device.type == "mps" else self.device
            self.detector = MTCNN(
                image_size = 160,    # Output face size
                margin = 20,          # Extra pixels around face
                min_face_size = 40,  # Ignore tiny faces (increased from 20 to reduce false positives)
                thresholds = [0.7, 0.8, 0.8], # Higher thresholds to reduce false positives
                factor = 0.709,               # Image pyramid scale factor
                device = detector_device,     # CPU for MPS, otherwise use config device
                keep_all = True,             # Return ALL faces, not just the best one,
                post_process = True           # Normalize output for FaceNet
            )

            self.embedding_model = InceptionResnetV1(
                pretrained = "vggface2",
                device = detector_device
            ).eval()    # Set to evaluation mode (no training)
            print(f"Face detection tool ready (device: {detector_device})")

        except Exception as e:
            print(f"Face detector failed: {e}")
            self.detector = None
            self.embedding_model = None

    def _classify_face_size(self, width: int, height: int, img_width: int, img_height: int) -> str:
        """
        Classify face size relative to the image.

        Helps streamline Risk Assessment Agent determine importance:
        - Large faces are likely main subjects
        - Small faces are likely bystanders

        Args:
            width: Face bounding box width
            height: Face bounding box height
            img_width: Full image width
            img_height: Full image height

        Returns:
            "large", "medium", or "small"
        """
        # Calculate the face area as percentage of image area
        face_area  = width * height
        image_area = img_width * img_height
        if image_area == 0:
            return "medium"
        ratio = face_area / image_area

        # Classify based on area ratio
        if ratio > 0.1:     # Face is >10% of image
            return "large"
        elif ratio > 0.02:  # Face is 2-10% of image
            return "medium"
        else:                # Face is <2% of image
            return "small"

    def _get_embedding(self, face_tensor: torch.Tensor) -> Optional[List[float]]:
        """
        Generate 512-dimensional face embedding for identity matching

        Used by the Consent Agent to:
        1. Compare detected faces against known people in FaceDatabase
        2. Store new face embeddings when user identifies someone

        Args:
            face_tensor: Preprocessed face tensor from MTCNN
        Returns:
            List of 512 floats (embedding vector) or None if failed
        """
        # Check if embedding model is available
        if self.embedding_model is None or face_tensor is None:
            return None

        try:
            # Disable gradient computation
            with torch.no_grad():
                if face_tensor.dim() == 3:
                    face_tensor = face_tensor.unsqueeze(0)   # Ensure tensor has batch dimension

                # Generate embedding
                embedding = self.embedding_model(face_tensor)

                # Convert to Python list of native floats for JSON serialization
                return [float(x) for x in embedding.squeeze().cpu().numpy()]

        except Exception as e:
            # Return None if embedding fails (face might be too blurry, etc)
            return None

    def _run(self, image_path: str) -> str:
        """
        Run face detection on an image.

        Main method that is to be run.

        Args:
            image_path: Path to the image file

        Returns:
            JSON string with detection results.
        """
        if self.detector is None:
            return json.dumps({"error": "Face detector not available", "faces": [], "count": 0})

        try:
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size

            try:
                # Single MTCNN pass: detect() returns boxes, probs, landmarks
                # without running the full pipeline twice (previously called detector(image) again for tensors)
                boxes, probs, landmarks = self.detector.detect(image, landmarks=True)

            except RuntimeError as e:
                # MPS fallback: if MPS fails, retry on CPU
                if "MPS" in str(e) or "divisible" in str(e):
                    print("MPS error detected, falling back to CPU for face detection...")
                    cpu_detector = MTCNN(
                        image_size=160,
                        margin=20,
                        min_face_size=40,
                        thresholds=[0.7, 0.8, 0.8],
                        factor=0.709,
                        device=torch.device("cpu"),
                        keep_all=True,
                        post_process = True
                    )
                    boxes, probs, landmarks = cpu_detector.detect(image, landmarks=True)
                else:
                    raise e
            # Handling No Detections
            if boxes is None:
                return json.dumps({"faces": [], "count": 0})

            # Processing Each Individual Face
            faces = []
            for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                # Skipping low confidence detections (0.8 threshold — for privacy, missed faces are costlier than false positives)
                if prob < 0.8:
                    continue

                # Extract Bounding Box
                # MTCNN returns [x1, y1, x2, y2] -> Convert to [x, y, width, height]
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1
                height = y2 - y1

                # Getting Face Embeddings (manual crop+align — avoids second MTCNN pass)
                embedding = None
                try:
                    margin = 20
                    cx1 = max(0, x1 - margin)
                    cy1 = max(0, y1 - margin)
                    cx2 = min(img_width, x2 + margin)
                    cy2 = min(img_height, y2 + margin)
                    face_crop = image.crop((cx1, cy1, cx2, cy2)).resize((160, 160), Image.BILINEAR)
                    face_tensor = torch.from_numpy(np.array(face_crop)).permute(2, 0, 1).float()
                    # Normalize to [-1, 1] (same as MTCNN post_process=True)
                    face_tensor = (face_tensor - 127.5) / 128.0
                    embedding = self._get_embedding(face_tensor)
                except Exception:
                    pass

                # Building Face Result
                faces.append({
                    "id": f"face_{i}",
                    "bbox": [x1, y1, width, height],
                    "confidence": float(prob),
                    "size": self._classify_face_size(width, height, img_width, img_height),
                    "has_landmarks": landmark is not None,
                    "has_embedding": embedding is not None,
                    "embedding": embedding
                })
            return json.dumps({"faces": faces, "count": len(faces)})

        except Exception as e:
            return json.dumps({"error": str(e), "faces": [], "count": 0})


class TextDetectionTool(BaseTool):
    """Tool for detecting and recognizing text in images using EasyOCR"""
    name: str = "detect_text"
    description: str = (
        "Detects and recognizes text in the image using OCR. "
        "Returns detected text content and locations. "
        "Use this tool when you need to detect and extract all text regions in an image during the detection phase."
    )
    detector: Any = None
    backend: Any = None
    config: Any = None

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self._init_detector()

    def _init_detector(self):
        """Initialize text detection backend (EasyOCR or DBNet via config)."""
        detection_cfg = getattr(getattr(self.config, 'models', None), 'detection', None)
        backend_name = getattr(detection_cfg, 'text_detector', 'easyocr') if detection_cfg else 'easyocr'
        options_cfg = getattr(detection_cfg, 'text_detector_options', None) if detection_cfg else None
        dbnet_model = getattr(options_cfg, 'dbnet_model', 'db_resnet50') if options_cfg else 'db_resnet50'
        gpu = self.config.system.device == "cuda"
        try:
            if backend_name == "dbnet":
                try:
                    self.backend = DBNetEasyOCRBackend(gpu=gpu, dbnet_model=dbnet_model)
                except ImportError as ie:
                    print(f"  DBNet import failed ({ie}), falling back to EasyOCR")
                    self.backend = EasyOCRBackend(gpu=gpu)
                    backend_name = "easyocr"
            else:
                self.backend = EasyOCRBackend(gpu=gpu)
            self.detector = self.backend.reader  # Alias for _find_missing_values
            print(f"  Text detection tool ready (backend: {backend_name}, model: {dbnet_model})")
        except Exception as e:
            print(f"  Text detector failed: {e}")
            self.backend = None
            self.detector = None

    def _classify_text_type(self, text: str) -> Dict[str, Any]:
        """
        Classify text for privacy risk assessment.

        Distinguishes between:
        - Actual sensitive values (SSN numbers, passwords, etc.) → CRITICAL/HIGH
        - Labels only ("Password:", "Bank Account:") → LOW (context indicator, not actual data)

        Returns:
            Dict with type, is_sensitive, is_pii, is_critical, is_label_only flags
        """
        text_clean = text.strip()

        # Normalize common OCR misreads: _ often misread from ":"
        # Trailing: "PIN_" → "PIN:"
        # Mid-text: "PIN_ 3902" → "PIN: 3902" (underscore followed by space+data)
        text_clean = re.sub(r'[_]+\s*$', ':', text_clean)
        text_clean = re.sub(r'[_]+\s+', ': ', text_clean)

        # ----- CRITICAL: Social Security Number (actual digits) -----
        if re.search(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', text_clean):
            return {
                "type": "ssn",
                "is_sensitive": True,
                "is_pii": True,
                "is_critical": True,
                "is_label_only": False
            }

        # ----- CRITICAL: Credit Card Number (actual digits) -----
        if re.search(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b', text_clean):
            return {
                "type": "credit_card",
                "is_sensitive": True,
                "is_pii": True,
                "is_critical": True,
                "is_label_only": False
            }

        # ----- Password/PIN: Separate label vs value -----
        pw_match = re.search(r'\b(password|pwd|pin|passcode)\b', text_clean, re.IGNORECASE)
        if pw_match:
            # Extract everything after the keyword
            after_keyword = text_clean[pw_match.end():].strip()
            # Remove separator (colon, equals)
            after_keyword = re.sub(r'^[:=]\s*', '', after_keyword).strip()

            if after_keyword:
                # Has actual value content (e.g., "PIN: 4821", "password=Secret123")
                return {
                    "type": "password",
                    "is_sensitive": True,
                    "is_pii": True,
                    "is_critical": True,
                    "is_label_only": False
                }
            else:
                # Label only (e.g., "Password:", "PIN:")
                return {
                    "type": "password_label",
                    "is_sensitive": True,
                    "is_pii": False,
                    "is_critical": False,
                    "is_label_only": True
                }

        # ----- Bank/Financial: Composite (keyword + actual numbers) -----
        # Broad keywords OK here because digits are required
        if re.search(r'\b(bank|account|routing|iban)\s*[:=]?\s*\d{4,}', text_clean, re.IGNORECASE):
            return {
                "type": "bank_account",
                "is_sensitive": True,
                "is_pii": True,
                "is_critical": True,
                "is_label_only": False
            }

        # ----- Bank/Financial: Label only (stricter multi-word keywords to avoid false positives) -----
        bank_label_match = re.search(r'\b(bank\s*account|account\s*number|routing\s*number|iban)\b', text_clean, re.IGNORECASE)
        if bank_label_match:
            return {
                "type": "bank_label",
                "is_sensitive": True,
                "is_pii": False,
                "is_critical": False,
                "is_label_only": True
            }

        # ----- SSN/Credit Card Label (keyword without actual numbers) -----
        ssn_label_match = re.search(r'\b(social\s*security|ssn)\b', text_clean, re.IGNORECASE)
        if ssn_label_match:
            return {
                "type": "ssn_label",
                "is_sensitive": True,
                "is_pii": False,
                "is_critical": False,
                "is_label_only": True
            }

        credit_label_match = re.search(r'\b(credit\s*card)\b', text_clean, re.IGNORECASE)
        if credit_label_match:
            return {
                "type": "credit_card_label",
                "is_sensitive": True,
                "is_pii": False,
                "is_critical": False,
                "is_label_only": True
            }

        # ----- PII: Phone Number -----
        phone_patterns = [
            r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
        ]
        for pattern in phone_patterns:
            if re.search(pattern, text_clean):
                return {
                    "type": "phone_number",
                    "is_sensitive": True,
                    "is_pii": True,
                    "is_critical": False,
                    "is_label_only": False
                }

        # ----- PII: Email Address -----
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_clean):
            return {
                "type": "email",
                "is_sensitive": True,
                "is_pii": True,
                "is_critical": False,
                "is_label_only": False
            }

        # ----- PII: Physical Address -----
        address_keywords = [
            'street', 'st.', 'st ', 'avenue', 'ave.', 'ave ',
            'road', 'rd.', 'rd ', 'drive', 'dr.', 'dr ',
            'lane', 'ln.', 'ln ', 'boulevard', 'blvd',
            'apt', 'apartment', 'suite', 'unit', 'floor'
        ]
        if any(kw in text_clean.lower() for kw in address_keywords):
            return {
                "type": "address",
                "is_sensitive": True,
                "is_pii": True,
                "is_critical": False,
                "is_label_only": False
            }

        # ----- Numeric fragments: isolated 3-8 digit numbers or partial numeric PII patterns -----
        # These could be fragments of SSN, PIN, account numbers split by OCR
        if re.match(r'^\d{3,8}$', text_clean) or re.match(r'^\d{2,3}-\d{2,4}$', text_clean):
            return {
                "type": "numeric_fragment",
                "is_sensitive": True,
                "is_pii": False,
                "is_critical": False,
                "is_label_only": False
            }

        # ----- Default: General Text -----
        return {
            "type": "general_text",
            "is_sensitive": False,
            "is_pii": False,
            "is_critical": False,
            "is_label_only": False
        }

    def _bbox_distance(self, bbox1: List, bbox2: List) -> float:
        """Calculate Euclidean distance between two bbox centers. Bboxes are [x, y, w, h]."""
        c1x = bbox1[0] + bbox1[2] / 2
        c1y = bbox1[1] + bbox1[3] / 2
        c2x = bbox2[0] + bbox2[2] / 2
        c2y = bbox2[1] + bbox2[3] / 2
        return ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5

    def _propagate_labels_to_values(self, texts: List[Dict], img_width: int = 2000, img_height: int = 2000) -> None:
        """
        Second pass: Propagate label classifications to nearby unclassified values.

        When OCR splits "Password:" and "Magic123!" into separate blocks,
        the label gets classified but the value doesn't. This pass links them
        by spatial proximity so the value inherits the label's risk type.
        """
        # Map label types to the value classification they propagate
        LABEL_TO_VALUE = {
            "password_label": ("password", True),       # (value_type, is_critical)
            "bank_label": ("bank_account", True),
            "ssn_label": ("ssn", True),
            "credit_card_label": ("credit_card", True),
        }

        labels = [t for t in texts if t.get("is_label_only", False)]
        if not labels:
            return

        for label in labels:
            propagation = LABEL_TO_VALUE.get(label["text_type"])
            if not propagation:
                continue

            value_type, is_critical = propagation

            for text in texts:
                # Propagate to unclassified general text or numeric fragments (likely PII values)
                if text["text_type"] not in ("general_text", "numeric_fragment") or text.get("is_label_only", False):
                    continue

                # Skip known label words that OCR split from their parent label
                # e.g., "Bank Account:" split into "Bank" + "Account:" by OCR
                # But do NOT skip unknown alphabetic strings — those may be passwords
                # e.g., "HaurerWvifi" near "Password:" IS a password value
                LABEL_WORDS = {
                    "bank", "account", "social", "security", "credit", "card",
                    "password", "pin", "routing", "number", "ssn", "iban",
                    "name", "address", "email", "phone", "date", "birth",
                    "expiry", "cvv", "code", "type", "id", "no",
                }
                text_content = text.get("text_content", "").strip()
                text_clean = re.sub(r'[:\s\-_.,;]', '', text_content).lower()
                if text_clean in LABEL_WORDS:
                    continue

                # Check spatial proximity
                # Use tighter vertical threshold (same row) with wider horizontal reach
                # to handle text spread across the same line in grid images
                dist = self._bbox_distance(label["bbox"], text["bbox"])
                label_cy = label["bbox"][1] + label["bbox"][3] / 2
                text_cy = text["bbox"][1] + text["bbox"][3] / 2
                vertical_dist = abs(label_cy - text_cy)

                # Same row (within 50px vertically): allow 20% of image width
                # Different row: allow 10% of image height
                same_row_threshold = int(img_width * 0.20)
                diff_row_threshold = int(img_height * 0.10)
                threshold = same_row_threshold if vertical_dist < 50 else diff_row_threshold

                if dist < threshold:
                    text["text_type"] = value_type
                    text["is_pii"] = True
                    text["is_critical"] = is_critical
                    text["is_sensitive"] = True

    def _merge_and_classify_numeric_fragments(self, texts: List[Dict]) -> None:
        """
        Merge adjacent numeric fragments and check for PII patterns.

        When OCR splits "233-45-6789" into "233" and "45-6789", and the label
        "Bank Account:" is missed by OCR, these fragments stay as unescalated
        numeric_fragment items. This method groups spatially adjacent numeric
        fragments, concatenates their text, and checks whether the combined
        string matches a known PII format (SSN, bank account, credit card).

        Only escalates when the merged pattern is a clear PII match.
        Mutates the text dicts in-place.
        """
        # Collect unescalated numeric fragments
        fragments = [t for t in texts
                     if t["text_type"] == "numeric_fragment"
                     and not t.get("is_critical", False)
                     and not t.get("is_pii", False)]

        if len(fragments) < 2:
            return

        # Sort by y-center then x position
        fragments.sort(key=lambda t: (t["bbox"][1] + t["bbox"][3] / 2, t["bbox"][0]))

        # Group fragments by spatial proximity (same row, close horizontally)
        groups: List[List[Dict]] = []
        used: set = set()

        for i, frag in enumerate(fragments):
            if i in used:
                continue
            group = [frag]
            used.add(i)

            # Iteratively extend the group with nearby fragments
            changed = True
            while changed:
                changed = False
                for j, other in enumerate(fragments):
                    if j in used:
                        continue
                    # Check proximity against any member of the group
                    for member in group:
                        mcy = member["bbox"][1] + member["bbox"][3] / 2
                        ocy = other["bbox"][1] + other["bbox"][3] / 2
                        if abs(mcy - ocy) > 50:
                            continue
                        # Horizontal: gap between right edge of member and left edge of other
                        m_right = member["bbox"][0] + member["bbox"][2]
                        o_left = other["bbox"][0]
                        o_right = other["bbox"][0] + other["bbox"][2]
                        m_left = member["bbox"][0]
                        gap = min(abs(o_left - m_right), abs(m_left - o_right))
                        if gap < 80:
                            group.append(other)
                            used.add(j)
                            changed = True
                            break

            if len(group) >= 2:
                groups.append(group)

        # PII patterns for merged text
        PII_PATTERNS = [
            (re.compile(r'^\d{3}-?\d{2}-?\d{4}$'), "ssn"),
            (re.compile(r'^\d{8,12}$'), "bank_account"),
            (re.compile(r'^\d{4}-?\d{4}-?\d{4}-?\d{4}$'), "credit_card"),
            (re.compile(r'^\d{13,19}$'), "credit_card"),
        ]

        for group in groups:
            # Sort by x position and merge text
            group.sort(key=lambda t: t["bbox"][0])
            merged = "".join(t.get("text_content", "").strip() for t in group)
            # Also try with dashes between fragments
            merged_dashed = "-".join(t.get("text_content", "").strip() for t in group)
            # Strip internal whitespace/dashes for digit-only check
            digits_only = re.sub(r'[^0-9]', '', merged)

            for pattern, pii_type in PII_PATTERNS:
                if pattern.match(merged) or pattern.match(merged_dashed) or pattern.match(digits_only):
                    for frag in group:
                        frag["text_type"] = pii_type
                        frag["is_pii"] = True
                        frag["is_critical"] = True
                        frag["is_sensitive"] = True
                    break

    def _find_missing_values(self, texts: List[Dict], image_path: str) -> List[Dict]:
        """
        Third pass: find values near labels that have no nearby classified value.

        When OCR detects a label like "PIN:" but misses the value next to it,
        re-crop the region to the right of the label and run OCR to recover
        the missing value.

        Returns:
            List of new text dicts to append to the texts list.
        """
        LABEL_TO_VALUE = {
            "password_label": ("password", True),
            "bank_label": ("bank_account", True),
            "ssn_label": ("ssn", True),
            "credit_card_label": ("credit_card", True),
        }

        labels = [t for t in texts
                  if t.get("is_label_only", False) and t["text_type"] in LABEL_TO_VALUE]
        if not labels:
            return []

        try:
            image = Image.open(image_path)
            img_w, img_h = image.size
        except Exception:
            return []

        # Find labels that have no nearby classified value
        same_row_thresh = int(img_w * 0.20)
        diff_row_thresh = int(img_h * 0.10)
        labels_without_values = []
        for label in labels:
            has_value = False
            for t in texts:
                if t is label or t.get("is_label_only", False):
                    continue
                if not (t.get("is_pii", False) or t.get("is_critical", False)):
                    continue
                label_cy = label["bbox"][1] + label["bbox"][3] / 2
                text_cy = t["bbox"][1] + t["bbox"][3] / 2
                dist = self._bbox_distance(label["bbox"], t["bbox"])
                threshold = same_row_thresh if abs(label_cy - text_cy) < 50 else diff_row_thresh
                if dist < threshold:
                    has_value = True
                    break
            if not has_value:
                labels_without_values.append(label)

        if not labels_without_values:
            return []

        LABEL_WORDS = {
            "bank", "account", "social", "security", "credit", "card",
            "password", "pin", "routing", "number", "ssn", "iban",
        }

        new_texts = []
        next_id = len(texts)

        for label in labels_without_values:
            lx, ly, lw, lh = label["bbox"]

            # Search in all 4 directions: right, below, left, above
            search_regions = [
                # Right of label
                (lx + lw, max(0, ly - int(lh * 0.5)),
                 min(lx + lw + lw * 4, img_w), min(ly + int(lh * 1.5), img_h)),
                # Below label
                (max(0, lx - int(lw * 0.5)), ly + lh,
                 min(lx + int(lw * 1.5), img_w), min(ly + lh + lh * 4, img_h)),
                # Left of label
                (max(0, lx - lw * 4), max(0, ly - int(lh * 0.5)),
                 lx, min(ly + int(lh * 1.5), img_h)),
                # Above label
                (max(0, lx - int(lw * 0.5)), max(0, ly - lh * 4),
                 min(lx + int(lw * 1.5), img_w), ly),
            ]

            found_value_for_label = False
            for search_x, search_y, search_x2, search_y2 in search_regions:
                if found_value_for_label:
                    break

                if search_x2 - search_x < 15 or search_y2 - search_y < 10:
                    continue

                try:
                    crop = image.crop((search_x, search_y, search_x2, search_y2))
                    crop_array = np.array(crop)

                    ocr_results = self.detector.readtext(
                        crop_array, width_ths=0.1, low_text=0.15,
                        paragraph=False
                    )

                    for (box_pts, text_content, conf) in ocr_results:
                        if conf < 0.25 or len(text_content.strip()) < 2:
                            continue

                        # Skip label words
                        clean = re.sub(r'[:\s\-_.,;]', '', text_content).lower()
                        if clean in LABEL_WORDS:
                            continue

                        # Convert local coords back to image coords
                        pts = np.array(box_pts)
                        min_x = int(pts[:, 0].min()) + search_x
                        min_y = int(pts[:, 1].min()) + search_y
                        max_x = int(pts[:, 0].max()) + search_x
                        max_y = int(pts[:, 1].max()) + search_y
                        new_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

                        # Check for overlap with existing detections
                        overlaps = False
                        for existing in texts + new_texts:
                            ex, ey, ew, eh = existing["bbox"]
                            if (min_x < ex + ew and max_x > ex and
                                    min_y < ey + eh and max_y > ey):
                                overlaps = True
                                break
                        if overlaps:
                            continue

                        value_type, is_critical = LABEL_TO_VALUE[label["text_type"]]

                        new_texts.append({
                            "id": f"text_recrop_{next_id}",
                            "text_content": text_content,
                            "bbox": new_bbox,
                            "confidence": float(conf),
                            "text_type": value_type,
                            "is_sensitive": True,
                            "is_pii": True,
                            "is_critical": is_critical,
                            "is_label_only": False,
                        })
                        next_id += 1
                        found_value_for_label = True
                        print(f"    RECROP: Found '{text_content}' near "
                              f"'{label.get('text_content', '')}' → {value_type}")
                        break  # One value per label per direction search

                except Exception as e:
                    print(f"    RECROP ERROR near '{label.get('text_content', '')}': {e}")

        return new_texts

    def _run(self, image_path: str) -> str:
        """Run text detection on an image"""
        if self.backend is None:
            return json.dumps({"error": "Text detector not available", "texts": [], "count": 0})

        try:
            image = Image.open(image_path)
            img_array = np.array(image)
            detections = self.backend.detect(img_array)
            texts = []

            for idx, region in enumerate(detections):
                classification = self._classify_text_type(region.text)

                texts.append({
                    "id": f"text_{idx}",
                    "text_content": region.text,
                    "bbox": region.bbox,
                    "polygon": region.polygon,
                    "confidence": region.confidence,
                    "text_type": classification["type"],
                    "is_sensitive": classification["is_sensitive"],
                    "is_pii": classification["is_pii"],
                    "is_critical": classification["is_critical"],
                    "is_label_only": classification["is_label_only"]
                })

            # Second pass: propagate label classifications to nearby unclassified values
            img_width, img_height = image.size
            self._propagate_labels_to_values(texts, img_width, img_height)

            # Pass 2.5: merge adjacent numeric fragments into PII patterns
            # Catches cases where OCR missed the label (e.g., "233" + "45-6789" = SSN)
            self._merge_and_classify_numeric_fragments(texts)

            # Third pass: find missing values near labels via targeted re-crop
            new_values = self._find_missing_values(texts, image_path)
            if new_values:
                texts.extend(new_values)

            return json.dumps({
                "texts": texts,
                "count": len(texts),
                "pii_count": sum(1 for t in texts if t["is_pii"]),
                "critical_count": sum(1 for t in texts if t["is_critical"])
            })

        except Exception as e:
            return json.dumps({"error": str(e), "texts": [], "count": 0})


class ObjectDetectionTool(BaseTool):
    """Tool for detecting objects in images using YOLO"""
    name: str = "detect_objects"
    description: str = (
        "Detects objects in the image like cars, laptops, phones, screens, etc. "
        "Returns list of detected objects with locations. "
        "Use this tool when you need to detect privacy-relevant objects in an image during the detection phase."
    )

    detector: Any = None
    device: Any = None
    config: Any = None

    # Privacy-relevant object classes from COCO dataset
    PRIVACY_OBJECTS: ClassVar[Set[str]] = {
        # Electronics with screens
        "laptop", "tv", "cell phone", "remote", "keyboard", "mouse",
        # Vehicles (may have license plates)
        "car", "truck", "bus", "motorcycle", "bicycle",
        # Personal items
        "backpack", "handbag", "suitcase"
    }

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.device = config.system.device
        self._init_detector()

    def _init_detector(self):
        """Initialize YOLO object detector"""
        try:
            self.detector = YOLO("yolov8n.pt")
            print(f"Object detection tool ready (YOLO on {self.device})")
        except Exception as e:
            print(f"Object detector failed: {e}")
            self.detector = None

    def _is_privacy_relevant(self, class_name: str) -> bool:
        """Check if object class is privacy-relevant"""
        return class_name.lower() in self.PRIVACY_OBJECTS

    def _get_risk_category(self, class_name: str) -> str:
        """Categorize object by privacy risk type"""
        class_lower = class_name.lower()

        # Screen devices
        if class_lower in {"laptop", "tv", "cell phone", "remote"}:
            return "screen_device"

        # Vehicles
        if class_lower in {"car", "truck", "bus", "motorcycle", "bicycle"}:
            return "vehicle"

        # Personal items
        if class_lower in {"backpack", "handbag", "suitcase"}:
            return "personal_item"

        # Input devices
        if class_lower in {"keyboard", "mouse"}:
            return "input_device"

        return "other"

    def _run(self, image_path: str) -> str:
        """Run object detection on an image"""
        if self.detector is None:
            return json.dumps({"error": "Object detector not available", "objects": [], "count": 0})

        try:
            # Run YOLO detection with lower confidence threshold to catch more objects
            results = self.detector.predict(
                image_path,
                conf = 0.25,  # Lower threshold to detect more objects (was 0.5)
                verbose = False,
                save = False,
                device = self.device
            )

            objects = []
            person_bboxes = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])

                    # Collect person bboxes separately for face cross-referencing
                    if class_name.lower() == "person":
                        person_bboxes.append([x1, y1, x2 - x1, y2 - y1])
                        continue

                    # Check privacy relevance
                    is_relevant = self._is_privacy_relevant(class_name)
                    risk_category = self._get_risk_category(class_name)

                    # Only include privacy-relevant objects
                    if is_relevant:
                        objects.append({
                            "id": f"obj_{len(objects)}",
                            "label": class_name,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "confidence": confidence,
                            "is_privacy_relevant": is_relevant,
                            "risk_category": risk_category,
                            "contains_screen": risk_category == "screen_device"
                        })

            return json.dumps({
                "objects": objects,
                "count": len(objects),
                "privacy_relevant_count": sum(1 for o in objects if o["is_privacy_relevant"]),
                "screen_count": sum(1 for o in objects if o.get("contains_screen", False)),
                "person_bboxes": person_bboxes,
                "person_count": len(person_bboxes)
            })

        except Exception as e:
            return json.dumps({"error": str(e), "objects": [], "count": 0})
