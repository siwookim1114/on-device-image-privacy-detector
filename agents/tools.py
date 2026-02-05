# Import required utils
import torch
from PIL import Image
from typing import Any, List, Dict, Optional, ClassVar, Set, Tuple
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

        Returns:
            Dict with type, is_sensitive, is_pii, is_critical flags
        """
        text_clean = text.strip()

        # ----- CRITICAL: Social Security Number -----
        if re.search(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', text_clean):
            return {
                "type": "ssn",
                "is_sensitive": True,
                "is_pii": True,
                "is_critical": True
            }

        # ----- CRITICAL: Credit Card Number -----
        if re.search(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b', text_clean):
            return {
                "type": "credit_card",
                "is_sensitive": True,
                "is_pii": True,
                "is_critical": True
            }

        # ----- CRITICAL: Password/PIN Labels -----
        if re.search(r'\b(password|pwd|pin|passcode)\s*[:=]?\s*\S+', text_clean, re.IGNORECASE):
            return {
                "type": "password",
                "is_sensitive": True,
                "is_pii": True,
                "is_critical": True
            }

        # ----- CRITICAL: Bank Account -----
        if re.search(r'\b(bank|account|routing|iban)\s*[:=]?\s*\d+', text_clean, re.IGNORECASE):
            return {
                "type": "bank_account",
                "is_sensitive": True,
                "is_pii": True,
                "is_critical": True
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
                    "is_critical": False
                }

        # ----- PII: Email Address -----
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_clean):
            return {
                "type": "email",
                "is_sensitive": True,
                "is_pii": True,
                "is_critical": False
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
                "is_critical": False
            }

        # ----- Default: General Text -----
        return {
            "type": "general_text",
            "is_sensitive": False,
            "is_pii": False,
            "is_critical": False
        }

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
                    "text_content": text,  # Changed from "text" to "text_content"
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
                    "is_critical": classification["is_critical"]
                })

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
    SENSITIVITY_TO_RISK = {
        "critical": RiskLevel.CRITICAL,     
        "high": RiskLevel.HIGH,             
        "medium": RiskLevel.MEDIUM,         
        "low": RiskLevel.LOW        
    }

    SIZE_ESCALATION = {
        "large": 2,                         # Large faces are highly identifiable
        "medium": 1,                        # Medium faces moderately identifiable
        "low": -1                           # Small faces less identifiable
    }

    CLARITY_ESCALATION = {
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
        if consent_status == "explicit":    # If the person in the photo has explicitly given permission (user's own face or someone who has clearly consented)
            sensitivity = self.privacy_profile.identity_sensitivity.get("own_face", "low")
        elif consent_status == "assumed":   # Likely a known contact like friend or family from embeddings -> People who have a history of appearing in photos with consent
            sensitivity = self.privacy_profile.identity_sensitivity.get("friend_faces", "medium")
        else:
            sensitivity = self.privacy_profile.identity_sensitivity.get("bystander_faces", "critical")
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
            bbox_list = data.get("bbox", [0, 0, 0, 0])
            bbox = BoundingBox(x=bbox_list[0], y=bbox_list[1], width=bbox_list[2], height=bbox_list[3])

            size = data.get("size", "medium")
            clarity = data.get("clarity", "medium")
            confidence = data.get("confidence", 0.0)
            consent_status = data.get("attributes", {}).get("consent_status", "none")
            image_width = data.get("image_width", 1920)
            image_height = data.get("image_height", 1080)

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
                "assessment_confidence": confidence,
                # Structured factors for VLM reasoning
                "factors": {
                    "consent_status": consent_status,
                    "consent_explanation": self._explain_consent_status(consent_status),
                    "base_risk": base_risk.value,
                    "size": size,
                    "size_explanation": self._explain_size(size),
                    "size_escalation": size_esc,

                    "consent_confidence": 0.0,
                    "assessment_confidence": confidence,
                    "escalation_applied": total_escalation,
                    "clarity": clarity,
                    "clarity_explanation": self._explain_clarity(clarity),
                    "clarity_escalation": clarity_esc,
                    "position": position_desc,
                    "position_escalation": position_esc,
                    "detection_confidence": confidence,
                    "confidence_adjustment": confidence_esc,
                    "total_escalation": total_escalation,
                    "final_risk": final_risk.value,
                    "image_dimensions": f"{image_width}x{image_height}"
                },
                # Consent metadata
                "consent_status": consent_status,
                "consent_confidence": 0.0
            })

        except Exception as e:
            return json.dumps({
                "error": str(e),
                "detection_id": "unknown",
                "severity": RiskLevel.HIGH.value,
                "requires_protection": True
            })