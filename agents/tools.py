# Import required utils
import torch
from PIL import Image
from typing import Any, List, Dict, Optional, ClassVar, Set, Tuple, Union
import json
import numpy as np
import re
# Detector Libraries

from facenet_pytorch import MTCNN     # Face Detection
from facenet_pytorch import InceptionResnetV1  # Face Embeddings (for Future Consent Agent)
import easyocr
from ultralytics import YOLO         # Object Detection (YOLO)
from langchain.tools import BaseTool
from utils.models import (
    RiskLevel,
    RiskType,
    PrivacyProfile,
    BoundingBox,
    RiskAssessment,
    FaceDetection,
    TextDetection,
    ObjectDetection,
    DetectionResults,
    ConsentStatus,
    PersonClassification
)
from utils.config import get_risk_color
## Detection Tools for Detection Agent

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
                # Run MTCNN Detection -> detect() returns: boxes, probabilities, landmarks
                boxes, probs, landmarks = self.detector.detect(image, landmarks=True)
                face_tensors = self.detector(image)   # Get aligned face tensors for embedding extraction

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
                    face_tensors = cpu_detector(image)
                else:
                    raise e
            # Handling No Detections
            if boxes is None:
                return json.dumps({"faces": [], "count": 0})

            # Processing Each Individual Face
            faces = []
            for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                # Skipping low confidence detections (higher threshold to reduce false positives)
                if prob < 0.9:
                    continue

                # Extract Bounding Box
                # MTCNN returns [x1, y1, x2, y2] -> Convert to [x, y, width, height]
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1
                height = y2 - y1

                # Getting Face Embeddings
                embedding = None
                if face_tensors is not None:
                    # Single tensor or batch tensor
                    if isinstance(face_tensors, torch.Tensor):
                        if face_tensors.dim() == 4 and i < face_tensors.shape[0]:
                            # Batch tensor: [N, 3, 160, 160]
                            embedding = self._get_embedding(face_tensors[i])
                        elif face_tensors.dim() == 3 and i == 0:
                            # Single tensor: [3, 160, 160]
                            embedding = self._get_embedding(face_tensors)

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
    config: Any = None

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self._init_detector()

    def _init_detector(self):
        """Initialize EasyOCR text detector"""
        try:
            self.detector = easyocr.Reader(
                ["en"],
                gpu = self.config.system.device == "cuda",
                verbose = False
            )
            print("Text detection tool ready")

        except Exception as e:
            print(f"Text detector failed: {e}")
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

    def _propagate_labels_to_values(self, texts: List[Dict]) -> None:
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
                # Only propagate to unclassified general text that isn't a label
                if text["text_type"] != "general_text" or text.get("is_label_only", False):
                    continue

                # Check spatial proximity (threshold: 200 pixels center-to-center)
                if self._bbox_distance(label["bbox"], text["bbox"]) < 200:
                    text["text_type"] = value_type
                    text["is_pii"] = True
                    text["is_critical"] = is_critical
                    text["is_sensitive"] = True

    def _run(self, image_path: str) -> str:
        """Run text detection on an image"""
        if self.detector is None:
            return json.dumps({"error": "Text detector not available", "texts": [], "count": 0})

        try:
            image = Image.open(image_path)
            img_array = np.array(image)
            results = self.detector.readtext(img_array)
            texts = []

            for idx, detection in enumerate(results):
                bbox_points, text, confidence = detection

                if confidence < 0.3:  # Lower threshold to catch more text
                    continue

                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]

                # Classify text type for privacy assessment
                classification = self._classify_text_type(text)

                texts.append({
                    "id": f"text_{idx}",
                    "text_content": text,
                    "bbox": [
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords) - min(x_coords)),
                        int(max(y_coords) - min(y_coords))
                    ],
                    "confidence": float(confidence),
                    "text_type": classification["type"],
                    "is_sensitive": classification["is_sensitive"],
                    "is_pii": classification["is_pii"],
                    "is_critical": classification["is_critical"],
                    "is_label_only": classification["is_label_only"]
                })

            # Second pass: propagate label classifications to nearby unclassified values
            self._propagate_labels_to_values(texts)

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
        # Documents/books
        "book",
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
                device = self.device
            )

            objects = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])

                    # Check privacy relevance
                    is_relevant = self._is_privacy_relevant(class_name)
                    risk_category = self._get_risk_category(class_name)

                    # Only include privacy-relevant objects
                    if is_relevant:
                        objects.append({
                            "id": f"obj_{len(objects)}",
                            "label": class_name,  # Changed from "class" to "label"
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
                "screen_count": sum(1 for o in objects if o.get("contains_screen", False))
            })

        except Exception as e:
            return json.dumps({"error": str(e), "objects": [], "count": 0})


## Risk Assessment Tools
"""
Risk Assessment Tools - Specialized tools for comprehensive privacy risk analysis
The agent uses these factors to generate VLM-based reasoning.

  Architecture:
      Tools = Pure calculation (fast, deterministic)
      Agent = VLM reasoning (contextual, batched)

  Phase 1: Individual Element Assessment (rule-based calculation)
      - FaceRiskAssessmentTool
      - TextRiskAssessmentTool
      - ObjectRiskAssessmentTool

  Phase 2: Contextual Enhancement (spatial reasoning)
      - SpatialRelationshipTool
      - ConsentInferenceTool
      - RiskEscalationTool

  Phase 3: Validation & Filtering (quality control)
      - FalsePositiveFilterTool
      - ConsistencyValidationTool
"""

class FaceRiskAssessmentTool(BaseTool):
    """
    Advance face privacy risk assessment with multi-factor scoring
    Returns structured data for agent to use in VLM reasoning. 

    Factors considered:
    1. Base risk: User sensitivity + consent status settings
    2. Face size (large = more identifiable)
    3. Face clarity (clear = more identifiable)
    4. Face position (center = likely subject, edge = likely bystander)
    5. Confidence Adjustment (low confidence = uncertain)
    6. Final risk level (after all escalations)
    """
    name: str = "assess_face_risk"
    description: str = (
        "Performs comprehensive privacy risk assessment for detected faces using "
        "multi-factor analysis including size, clarity, position, and user sensitivity. " \
        "Use this tool when doing risk assessments on detected faces"
    )
    config: Any = None
    privacy_profile: PrivacyProfile = None

    # Risk escalation mappings
    SENSITIVITY_TO_RISK: ClassVar[Dict] = {
        "critical": RiskLevel.CRITICAL,     
        "high": RiskLevel.HIGH,             
        "medium": RiskLevel.MEDIUM,         
        "low": RiskLevel.LOW        
    }

    SIZE_ESCALATION: ClassVar[Dict] = {
        "large": 2,                         # Large faces are highly identifiable
        "medium": 1,                        # Medium faces moderately identifiable
        "small": 0                          # Small faces less identifiable
    }

    CLARITY_ESCALATION: ClassVar[Dict] = {
        "high": 1,                          # Clear faces are easily identifiable
        "medium": 0,                        # Normal clarity
        "low": -1                           # Blurry/obscured faces harder to identify
    }

    def __init__(self, config, privacy_profile: PrivacyProfile = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.privacy_profile = privacy_profile if privacy_profile else PrivacyProfile()

    def _get_base_risk(self, consent_status: str) -> RiskLevel:
        """Get base risk from user's privacy profile based on consent status. (User's privacy preferences)"""
        key_map = {
            "explicit": "own_face",
            "assumed": "friend_faces",
            "none": "bystander_faces"
        }
        key = key_map.get(consent_status, "bystander_faces")
        sensitivity = self.privacy_profile.identity_sensitivity.get(key, "critical")
        return self.SENSITIVITY_TO_RISK.get(sensitivity, RiskLevel.HIGH)
    
    def _escalate_risk(self, base_risk: RiskLevel, escalation: int) -> RiskLevel:
        """Escalate or de-escalate risk level by specified amount."""
        risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        current_idx = risk_order.index(base_risk)
        new_idx = max(0, min(current_idx + escalation, len(risk_order) - 1))
        return risk_order[new_idx]
    
    def _calculate_position_factor(self, bbox: BoundingBox, image_width: int, image_height: int) -> Tuple[int, str]:
        """
        Calculate position-based risk factor.
        Center faces = likely subjects (lower risk if user photo)
        Edge faces = likely bystanders (higher risk)
        """
        center_x = bbox.x + bbox.width / 2
        center_y = bbox.y + bbox.height / 2
        rel_x = center_x / image_width
        rel_y = center_y / image_height

        # Check if face is in central region (30% - 70% on both axes)
        if 0.3 <= rel_x <= 0.7 and 0.3 <= rel_y <= 0.7:
            return (0, "centered (likely subject)")
        elif rel_x < 0.2 or rel_x > 0.8 or rel_y < 0.2 or rel_y > 0.8:
            return (1, "edge position (likely bystander)")
        else:
            return (0, "intermediate position")
    
    def _explain_consent_status(self, consent_status: str) -> str:
        """Get human-readable consent explanation."""
        if consent_status == "none":
            return "unknown person, likely bystander without consent"
        elif consent_status == "explicit":
            return "person with explicit consent"
        elif consent_status == "assumed":
            return "known person with assumed consent"
        else:
            return f"person with {consent_status} consent status"
    
    def _explain_size(self, size: str) -> str:
        "Get human-readable size explanation."
        return {
            "large": "prominent and highly identifiable",
            "medium": "moderately visible",
            "small": "background presence, less identifiable"
        }.get(size, "normal_visibility")

    def _explain_clarity(self, clarity: str) -> str:
        """Get human-readable clarity explanation."""
        return {
            "high": "clear and easily identifiable",
            "medium": "normal clarity",
            "low": "blurry or obscured, harder to identify"
        }.get(clarity, "normal clarity")
    
    def _run(self, face_json: str) -> str:
        """
        Calculate all risk factors for a detected face.

        Args:
            face_json: JSON with face data including id, bbox, size, clarity, confidence, attributes

        Returns:
            JSON with calculated factors 
        """
        try:
            data = json.loads(face_json)

            # Extract face properties
            face_id = data.get("id", "unknown")
            bbox_raw = data.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox_raw, dict):
                bbox = BoundingBox(**bbox_raw)
            elif isinstance(bbox_raw, list):
                bbox = BoundingBox(x=bbox_raw[0], y=bbox_raw[1], width=bbox_raw[2], height=bbox_raw[3])
            else:
                bbox = BoundingBox(x=0, y=0, width=0, height=0)

            size = data.get("size", "medium")
            confidence = data.get("confidence", 0.0)

            consent_status = data.get("attributes", {}).get("consent_status", "none")
            image_width = data.get("image_width", 1920)
            image_height = data.get("image_height", 1080)

            # Infer clarity from quality indicators
            clarity = data.get("clarity", "medium")

            # Calculate all factors

            # Step 1: Get base risk from user sensitivity
            base_risk = self._get_base_risk(consent_status)

            # Step 2: Calculate escalation factors
            size_esc = self.SIZE_ESCALATION.get(size, 0)
            clarity_esc = self.CLARITY_ESCALATION.get(clarity, 0)
            position_esc, position_desc = self._calculate_position_factor(bbox, image_width, image_height)

            # Confidence adjustment: very low confidence reduces risk
            confidence_esc = -1 if confidence < 0.85 else 0

            # Total escalation
            total_escalation = size_esc + clarity_esc + position_esc + confidence_esc

            # Step 3: Calculate final risk
            final_risk = self._escalate_risk(base_risk, total_escalation)

            # Step 4: Determine protection requirement
            requires_protection = final_risk in [RiskLevel.CRITICAL, RiskLevel.HIGH]

            # Get sensitivity applied
            sensitivity_applied = {
                "none": "bystander_faces",
                "explicit": "own_face",
                "assumed": "friend_faces"
            }.get(consent_status, "bystander_faces")

            # Return structured factors
            return json.dumps({
                "detection_id": face_id,
                "element_type": "face",
                "element_description": f"Face ({size}, {clarity} clarity)",
                "risk_type": RiskType.IDENTITY_EXPOSURE.value,
                "severity": final_risk.value,
                "color_code": get_risk_color(self.config, final_risk.value),
                "user_sensitivity_applied": sensitivity_applied,
                "bbox": bbox.to_list(),
                "requires_protection": requires_protection,
                # Structured factors for VLM reasoning
                "factors": {
                    "base_risk": base_risk.value,
                    "size": size,
                    "size_explanation": self._explain_size(size),
                    "size_escalation": size_esc,
                    "escalation_applied": total_escalation,
                    "clarity": clarity,
                    "clarity_explanation": self._explain_clarity(clarity),
                    "clarity_escalation": clarity_esc,
                    "position": position_desc,
                    "position_escalation": position_esc,
                    "detection_confidence": confidence,
                    "confidence_adjustment": confidence_esc,
                    "image_dimensions": f"{image_width}x{image_height}"
                },
                # Consent metadata
                "consent_status": consent_status,
                "consent_explanation": self._explain_consent_status(consent_status),
                "consent_confidence": 0.0
            })

        except Exception as e:
            return json.dumps({
                "error": str(e),
                "detection_id": "unknown",
                "severity": RiskLevel.HIGH.value,
                "requires_protection": True
            })

class TextRiskAssessmentTool(BaseTool):
    """
    Pure text risk calculation tool.

    Calculates PII classification and risk factors WITHOUT generating reasoning.
    Returns structured data for agent to use in VLM reasoning.

    Features:
    1. Multi-pattern PII detection (SSN, credit cards, phone, email, etc.)
    2. Risk level calculation based on text type
    3. Confidence-based adjustments
    """
    name: str = "assess_text_risk"
    description: str = (
        "Calculates comprehensive privacy risk factors for detected text using "
        "pattern matching, PII classification, and sensitivity analysis."
        "Use this tool when doing risk assessments on detected texts."
    )
    config: Any = None
    privacy_profile: PrivacyProfile = None

    RISK_TYPES: ClassVar[Dict] = {
        "critical_types": {
            "ssn", "social_security", "credit_card", "bank_account",
            "password", "pin", "routing_number", "cvv", "api_key", "secret"
        },
        "high_risk_types": {
            "phone", "phone_number", "email", "address",
            "date_of_birth", "dob", "license", "passport",
            "id_number", "medical_record", "employee_id"
        },
        "medium_risk_types": {
            "name", "username", "account_number", "student_id"
        }
    }

    # Sensitivity mapping
    TYPE_TO_SENSITIVITY: ClassVar[Dict] = {
        "ssn": "personal_numbers",
        "credit_card": "financial_data",
        "bank_account": "financial_data",
        "phone": "personal_numbers",
        "phone_number": "personal_numbers",
        "email": "personal_numbers",
        "address": "personal_numbers",
        "password": "personal_numbers",
        "general_text": "documents"
    }

    def __init__(self, config, privacy_profile: PrivacyProfile = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.privacy_profile = privacy_profile if privacy_profile else PrivacyProfile()
    
    def _get_risk_level(
        self,
        text_type: str,
        is_pii: bool,
        is_critical: bool,
        confidence: float
    ) -> RiskLevel:
        """
        Determine risk level based on text classification.

        Args:
            text_classification: (text_type, is_pii, is_critical) from _classify_text
            confidence: OCR confidence score
        
        Returns:
            RiskLevel enum
        """
        text_type = text_type.lower() if text_type else ""
        # CRITICAL: Already identified as critical by pattern matching
        if is_critical or text_type in self.RISK_TYPES['critical_types']:
            return RiskLevel.CRITICAL
        elif is_pii or text_type in self.RISK_TYPES['high_risk_types']:
            # High confidence PII = HIGH risk
            # Low confidence PII = Medium risk (might be OCR error)
            return RiskLevel.HIGH if confidence > 0.6 else RiskLevel.MEDIUM
        # CRITICAL: Double-check specific critical types as fallback
        elif text_type in self.RISK_TYPES['medium_risk_types']:
            return RiskLevel.MEDIUM if confidence > 0.5 else RiskLevel.LOW
        else:
            return RiskLevel.LOW
        
    def _explain_text_type(self, text_type: str, is_pii: bool, is_critical: bool) -> str:
        """Get human-readable text type explanation."""
        if is_critical:
            return f"critical data type: {text_type}"
        elif is_pii:
            return f"personally identifiable information: {text_type}"
        else:
            return f"general text: {text_type}"
    
    def _run(self, text_json: str) -> str:
        """
        Calculate risk factors using text data.

        Args:
            text_json: JSON with text data from TextDetectionTool

        Returns:
            JSON with calculated factors
        """
        try:
            data = json.loads(text_json)

            # Extract data
            text_id = data.get("id", "unknown")
            text_content = data.get("text_content", "")
            bbox_raw = data.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox_raw, dict):
                bbox = BoundingBox(**bbox_raw)
            elif isinstance(bbox_raw, list):
                bbox = BoundingBox(x=bbox_raw[0], y=bbox_raw[1], width=bbox_raw[2], height=bbox_raw[3])
            else:
                bbox = BoundingBox(x=0, y=0, width=0, height=0)
            confidence = data.get("confidence", 0)
            text_type = data.get("text_type", "general_text")
            is_pii = data.get("attributes", {}).get("is_pii", False)
            is_critical = data.get("attributes", {}).get("is_critical", False)

            # Calculate risk level
            risk_level = self._get_risk_level(text_type, is_pii, is_critical, confidence)

            # Get user sensitivity
            sensitivity_category = self.TYPE_TO_SENSITIVITY.get(text_type, "documents")
            user_sensitivity = self.privacy_profile.information_sensitivity.get(sensitivity_category, "high")
            
            # Content preview (masked if sensitive)
            if is_critical or is_pii:
                content_preview = f"[{text_type.upper()}_REDACTED]"
            else:
                content_preview = text_content[:30] + "..." if len(text_content) > 30 else text_content
            
            requires_protection = risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]

            return json.dumps({
                "detection_id": text_id,
                "element_type": "text",
                "element_description": f"Text: {text_content}",
                "risk_type": RiskType.INFORMATION_DISCLOSURE.value,
                "severity": risk_level.value,
                "color_code": get_risk_color(self.config, risk_level.value),
                "user_sensitivity_applied": user_sensitivity,
                "bbox": bbox.to_list(),
                "requires_protection": requires_protection,
                "factors": {
                    "text_type": text_type, 
                    "text_type_explanation": self._explain_text_type(text_type, is_pii, is_critical),
                    "is_pii": is_pii,     
                    "is_critical": is_critical,
                    "content_preview": content_preview,
                    "ocr_confidence": confidence,
                    "sensitivity_category": sensitivity_category,
                    "user_sensitivity": user_sensitivity
                }
            })
        
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "detection_id": "unknown",
                "severity": RiskLevel.HIGH.value,
                "requires_protection": True
            })

class ObjectRiskAssessmentTool(BaseTool):
    name: str = "assess_object_risk"
    description: str = (
        "Calculates privacy risk for pre-classified objects from ObjectDetectionTool. "
        "Use this tool when doing risk assessments on detected privacy-relevant objects."
    )
    config: Any = None
    privacy_profile: PrivacyProfile = None

    def __init__(self, config, privacy_profile: PrivacyProfile = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.privacy_profile = privacy_profile if privacy_profile else PrivacyProfile()

    def _get_risk_level(self, risk_category: str, contains_screen: bool) -> RiskLevel:
        """
        Calculate risk level from pre-classified risk category.
        
        ObjectDetectionTool already classified objects into risk categories.
        """

        # Screen devices with visible content
        if contains_screen or risk_category == "screen_device":
            return RiskLevel.HIGH
        
        # Vehicles (licsense plate risk)
        if risk_category == "vehicle":
            return RiskLevel.MEDIUM
        
        # Personal items, input devices
        if risk_category in ["personal_items", "input_devices"]:
            return RiskLevel.LOW

        # Other privacy-relevant objects
        return RiskLevel.LOW

    def _get_risk_type(self, risk_category: str, contains_screen: bool) -> RiskType:
        """Determine risk type from risk_category"""        
        if contains_screen or risk_category == "screen_device":
            return RiskType.INFORMATION_DISCLOSURE
        
        if risk_category == "vehicle":
            return RiskType.LOCATION_EXPOSURE
        
        return RiskType.CONTEXT_EXPOSURE
    
    def _explain_object_category(self, risk_category: str, label: str) -> str:
        """Human-readable object category explanation"""
        explanations = {
            "screen_device": f"{label} with potentially visible screen content",
            "vehicle": f"{label} that may show license plate",
            "personal_item": f"{label} (personal belonging)",
            "input_device": f"{label} (computer peripheral)",
            "other": f"{label} (privacy-relevant object)"
        }
        return explanations.get(risk_category, f"{label}")
    
    def _run(self, object_json: str) -> str:
        """
        Calculate risk factors using pre-classified object data.

        Args:
            object_json: JSON with object data from ObjectDetectionTool
        
        Returns:
            JSON with calculated factors or filtered status
        """
        try:
            data = json.loads(object_json)

            # Extract pre-classified data from ObjectDetectionTool
            obj_id = data.get("id", "unknown")
            obj_label = data.get("object_class", "unknown")
            bbox_raw = data.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox_raw, dict):
                bbox = BoundingBox(**bbox_raw)
            elif isinstance(bbox_raw, list):
                bbox = BoundingBox(x=bbox_raw[0], y=bbox_raw[1], width=bbox_raw[2], height=bbox_raw[3])
            else:
                bbox = BoundingBox(x=0, y=0, width=0, height=0)
            confidence = data.get("confidence", 0.0)

            # Use existing classification
            is_privacy_relevant = data.get("attributes", {}).get("is_privacy_relevant", True)
            risk_category = data.get("attributes", {}).get("risk_category", "other")
            contains_screen = data.get("contains_screen", False) 

            # Objects already filtered by ObjectDetectionTool
            if not is_privacy_relevant:
                return json.dumps({
                    "detection_id": obj_id,
                    "filtered": True,
                    "reason": f"Object '{obj_label}' marked as not privacy-relevant"
                })
            
            # Calculate risk from existing classification
            risk_level = self._get_risk_level(risk_category, contains_screen)
            risk_type = self._get_risk_type(risk_category, contains_screen)

            # Get user sensitivity
            if risk_category == "screen_device":
                user_sensitivity = self.privacy_profile.information_sensitivity.get("screens", "high")
            elif risk_category == "vehicle":
                user_sensitivity = self.privacy_profile.location_sensitivity.get("license_plates", "high")
            else:
                user_sensitivity = self.privacy_profile.context_sensitivity.get("background_items", "medium")
            
            requires_protection = contains_screen or risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]

            return json.dumps({
                "detection_id": obj_id,
                "element_type": "object",
                "element_description": f"Object: {obj_label}",
                "risk_type": risk_type.value,
                "severity": risk_level.value,
                "color_code": get_risk_color(self.config, risk_level.value),
                "user_sensitivity_applied": user_sensitivity,
                "bbox": bbox.to_list(),
                "requires_protection": requires_protection,
                "filtered": False,
                "factors": {
                    "object_category": self._explain_object_category(risk_category, obj_label),
                    "risk_category": risk_category,     # From ObjectDetectionTool
                    "contains_screen": contains_screen,
                    "detection_confidence": confidence,
                }
            })
        
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "detection_id": "unknown",
                "filtered": True,
                "reason": f"Assessment error: {str(e)}"
            })
        
# Contextual Enhancement Tools
class SpatialRelationshipTool(BaseTool):
    """Analyze spatial relationships between detected elements."""
    name: str = "analyze_spatial_relationship"
    description: str = (
        "Analyzes spatial proximity and relationships between elements "
        "to identify compounded privacy risks. "
        "Use this tool when you need to find spatial relationships between detected elements (e.g., face near PII text, face near screen)."
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calculate_distance(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate Euclidean distance between bbox centers."""
        if isinstance(bbox1, list):
            center1_x = bbox1[0] + bbox1[2] / 2
            center1_y = bbox1[1] + bbox1[3] / 2
        else:
            center1_x = bbox1["x"] + bbox1["width"] / 2
            center1_y = bbox1["y"] + bbox1["height"] / 2

        if isinstance(bbox2, list):
            center2_x = bbox2[0] + bbox2[2] / 2
            center2_y = bbox2[1] + bbox2[3] / 2
        else:
            center2_x = bbox2["x"] + bbox2["width"] / 2
            center2_y = bbox2["y"] + bbox2["height"] / 2

        return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
    
    def _is_near(self, bbox1: Dict, bbox2: Dict, threshold: float = 150) -> bool:
        """Check if bboxes are within threshold pixels."""
        return self._calculate_distance(bbox1, bbox2) < threshold
    
    def _run(self, detections_json: str) -> str:
        """Analyze spatial relationships."""
        try:
            data = json.loads(detections_json)
            faces = data.get("faces", [])
            texts = data.get("texts", [])
            objects = data.get("objects", [])

            escalations = []

            # Rule 1: Face near PII text (identity linkage)
            for face in faces:
                for text in texts:
                    if self._is_near(face["bbox"], text["bbox"], threshold = 200):
                        if text.get("attributes", {}).get("is_pii", False):
                            escalations.append({
                                "elements": [face["id"], text["id"]],
                                "escalation_amount": 2,
                                "reason": f"Identity linkage: face near {text.get('text_type', 'PII')}",
                                "relationship_type": "identity_linkage"
                            })

            # Rule 2: Face near screen
            for face in faces:
                for obj in objects:
                    if obj.get("contains_screen", False):
                        if self._is_near(face["bbox"], obj["bbox"], threshold = 300):
                            escalations.append({
                                "elements": [face["id"], obj["id"]],
                                "escalation_amount": 1,
                                "reason": "Face near screen with visible content",
                                "relationship_type": "content_association"
                            })
            
            # Rule 3: Multiple small faces (group bystanders)
            small_faces = [f for f in faces if f.get("size") == "small"]
            if len(faces) >= 3 and len(small_faces) >= 2:
                for face in small_faces:
                    escalations.append({
                        "elements": [face["id"]],
                        "escalation_amount": 1,
                        "reason": "Background face in group photo",
                        "relationship_type": "group_bystander"
                    })
            
            return json.dumps({
                "escalations": escalations,
                "total_escalations": len(escalations)
            })
        
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "escalations": [],
                "total_escalations": 0
            })
        
class ConsentInferenceTool(BaseTool):
    """Infers consent likelihood from visual cues."""
    name: str = "infer_consent_likelihood"
    description: str = (
        "Infers consent status using size, position, and context. "
        "Use this tool when you need to estimate consent likelihood for detected faces based on visual cues without face recognition."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_relative_position(self, bbox: Dict, image_width: int, image_height: int) -> str:
        """Classify face position."""
        if isinstance(bbox, list):
            center_x = (bbox[0] + bbox[2] / 2) / image_width
            center_y = (bbox[1] + bbox[3] / 2) / image_height

        else:
            center_x = (bbox["x"] + bbox["width"] / 2) / image_width
            center_y = (bbox["y"] + bbox["height"] / 2) / image_height

        if 0.3 <= center_x <= 0.7 and 0.3 <= center_y <= 0.7:
            return "center"
        elif center_x < 0.2 or center_x > 0.8 or center_y < 0.2 or center_y > 0.8:
            return "edge"
        else:
            return "intermediate"
    
    def _run(self, face_context_json: str) -> str:
        """Infer consent likelihood"""
        try:
            data = json.loads(face_context_json)
            face = data.get("face", {})
            image_context = data.get("image_context", {})

            # Use pre-classified size from FaceDetectionTool
            size = face.get("size", "medium")
            bbox = face.get("bbox", [0, 0, 0, 0])

            image_width = image_context.get("width", 1920)
            image_height = image_context.get("height", 1080)
            total_faces = image_context.get("total_faces", 1)

            consent_score = 0.0
            reasoning_parts = []

            # Size (from FaceDetectionTool)
            if size == "large":
                consent_score += 0.3
                reasoning_parts.append("large face (likely subject)")
            elif size == "small":
                consent_score -= 0.2
                reasoning_parts.append("small face (likely background)")
            
            # Position
            position = self._get_relative_position(bbox, image_width, image_height)
            if position == "center":
                consent_score += 0.3
                reasoning_parts.append("centered")
            elif position == "edge":
                consent_score -= 0.2
                reasoning_parts.append("edge position")
            
            # Count
            if total_faces == 1:
                consent_score += 0.2
                reasoning_parts.append("only face")
            elif total_faces > 3:
                consent_score -= 0.1
                reasoning_parts.append("group photo")
            
            # Map to consent status
            if consent_score > 0.5:
                status = ConsentStatus.ASSUMED.value
                confidence = min(consent_score, 0.85)
            elif consent_score > 0:
                status = ConsentStatus.UNCLEAR.value
                confidence = 0.5
            else:
                status = ConsentStatus.NONE.value
                confidence = min(abs(consent_score) + 0.3, 0.8)
            
            return json.dumps({
                "face_id": face.get("id", "unknown"),
                "inferred_consent_status": status,
                "consent_confidence": confidence,
                "consent_score": consent_score,
                "reasoning": " | ".join(reasoning_parts)
            })
        
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "face_id": "unknown",
                "inferred_consent_status": ConsentStatus.UNCLEAR.value,
                "consent_confidence": 0.5
            })

class RiskEscalationTool(BaseTool):
    """Applies risk escalations from spatial analysis"""
    name: str = "apply_risk_escalations"
    description: str = (
        "Applies risk escalations based on spatial relationships. "
        "Use this tool after spatial analysis to escalate risk severity for elements with compounded privacy risks."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _escalate_severity(self, current_severity: str, escalation_amount: int) -> str:
        """Escalate risk severity."""
        risk_order = ["low", "medium", "high", "critical"]
        try:
            current_idx = risk_order.index(current_severity)
        except ValueError:
            current_idx = 1
        
        new_idx = min(current_idx + escalation_amount, len(risk_order) - 1)
        return risk_order[new_idx]

    def _run(self, escalation_data_json: str) -> str:
        """Apply escalations."""
        try:
            data = json.loads(escalation_data_json)
            
            assessments = data.get("assessments", [])
            escalations = data.get("escalations", [])

            assessment_map = {a["detection_id"]: a for a in assessments}
            escalations_applied = 0

            for escalation in escalations:
                elements = escalation.get("elements", [])
                escalation_amount = escalation.get("escalation_amount", 1)
                reason = escalation.get("reason", "Contextual escalation")

                for element_id in elements:
                    if element_id in assessment_map:
                        assessment = assessment_map[element_id]
                        old_severity = assessment.get("severity", "medium")
                        new_severity = self._escalate_severity(old_severity, escalation_amount)

                        if new_severity != old_severity:
                            assessment["severity"] = new_severity
                            assessment["factors"]["final_risk"] = new_severity
                            assessment["factors"]["escalation_reason"] = reason
                            assessment["requires_protection"] = new_severity in ["high", "critical"]
                            escalations_applied += 1
            
            return json.dumps({
                "assessments": list(assessment_map.values()),
                "escalations_applied": escalations_applied
            })
        
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "assessments": data.get("assessments", []),
                "escalations_applied": 0
            })

# Validation and Filtering Tools
class FalsePositiveFilterTool(BaseTool):
    """Filters false positive detections."""
    name: str = "filter_false_positives"
    description: str = (
        "Filters false positives to reduce noise. "
        "Use this tool to remove low-confidence detections and tiny elements that are likely false positives."
    )
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _run(self, assessments_json: str) -> str:
        """Filter false positives"""
        try:
            data = json.loads(assessments_json)
            assessments = data.get("assessments", [])

            filtered = []
            filter_stats = {
                "low_confidence": 0,
                "too_small": 0,
                "already_filtered": 0
            }

            for assessment in assessments:
                if assessment.get("filtered", False):
                    filter_stats["already_filtered"] += 1
                    continue

                # Get confidence from factors based on element type
                factors = assessment.get("factors", {})
                element_type = assessment.get("element_type", "")

                if element_type == "text":
                    confidence = factors.get("ocr_confidence", 1.0)
                elif element_type in ["face", "object"]:
                    confidence = factors.get("detection_confidence", 1.0)
                else:
                    confidence = 1.0

                # Filter low confidence
                if element_type == "text" and confidence < 0.3:
                    filter_stats["low_confidence"] += 1
                    continue
                elif element_type == "face" and confidence < 0.90:
                    filter_stats["low_confidence"] += 1
                    continue
            
                # Filter tiny elements
                bbox = assessment.get("bbox", [0, 0, 0, 0])
                width = bbox[2] if isinstance(bbox, list) else bbox.get("width", 0)
                height = bbox[3] if isinstance(bbox, list) else bbox.get("height", 0)

                if width < 20 or height < 20:
                    filter_stats["too_small"] += 1
                    continue

                filtered.append(assessment)

            return json.dumps({
                "assessments": filtered,
                "original_count": len(assessments),
                "filtered_count": len(filtered),
                "removed_count": len(assessments) - len(filtered),
                "filter_stats": filter_stats
            })
        
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "assessments": data.get("assessments", []),
                "original_count": len(data.get("assessments", [])),
                "filtered_count": len(data.get("assessments", []))
            })

class ConsistencyValidationTool(BaseTool):
    """Validate consistency across assessments."""
    name: str = "validate_consistency"
    description: str = (
        "Validates logical consistency across risk assessments. "
        "Use this tool to ensure severity levels match protection requirements (e.g., critical/high must require protection)."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, assessments_json: str) -> str:
        """Validate consistency"""
        try:
            data = json.loads(assessments_json)
            assessments = data.get("assessments", [])

            corrections = 0
            for assessment in assessments:
                severity = assessment.get("severity", "low")
                requires_protection = assessment.get("requires_protection", False)

                # Critical/High must require protection
                if severity in ["critical", "high"] and not requires_protection:
                    assessment["requires_protection"] = True
                    corrections += 1

                # Low shouldn't require protection
                if severity == "low" and requires_protection:
                    assessment["requires_protection"] = False
                    corrections += 1
                
            return json.dumps({
                "assessments": assessments,
                "validated_count": len(assessments),
                "corrections_made": corrections
            })
        
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "assessments": data.get("assessments", []),
                "validated_count": 0,
                "corrections_made": 0
            })



            

            
        





