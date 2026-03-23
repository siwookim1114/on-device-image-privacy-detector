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

import json
import re
import numpy as np
from typing import Any, List, Dict, Optional, ClassVar, Set, Tuple, Type, Union

from PIL import Image
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from utils.models import (
    RiskLevel,
    RiskType,
    PrivacyProfile,
    BoundingBox,
    ConsentStatus,
    ReclassifyAssessmentInput,
    ReclassifyItem,
    BatchReclassifyInput,
    SplitPart,
    SplitAssessmentInput,
)
from agents.tools.common import _parse_tool_input, get_risk_color


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
        "Performs comprehensive privacy risk assessment for a single detected face. "
        "Input: a JSON string of ONE face dict with keys: id, bbox, size, clarity, "
        "confidence, attributes, image_width, image_height. "
        "Pass each face from the Face Data array individually. "
        "Example: {\"id\": \"abc\", \"bbox\": {\"x\":0,\"y\":0,\"width\":100,\"height\":100}, "
        "\"size\": \"medium\", \"clarity\": \"high\", \"confidence\": 0.99, "
        "\"image_width\": 1024, \"image_height\": 768}"
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
            data = _parse_tool_input(face_json)

            # Unwrap if LLM sent wrapped format like {"face_data": [...]} or [...]
            if "face_data" in data and isinstance(data.get("face_data"), list) and data["face_data"]:
                wrapper = data
                data = data["face_data"][0]
                # Carry over image dimensions from wrapper level
                data.setdefault("image_width", wrapper.get("image_width", 1920))
                data.setdefault("image_height", wrapper.get("image_height", 1080))
            elif isinstance(data, list) and data:
                data = data[0]

            # Extract face properties
            face_id = data.get("id", "unknown")
            bbox = BoundingBox.from_raw(data.get("bbox", [0, 0, 0, 0]))

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
        "Calculates privacy risk for a single detected text region. "
        "Input: a JSON string of ONE text dict with keys: id, text_content, text_type, "
        "bbox, confidence, attributes. "
        "Pass each text from the Text Data array individually. "
        "Example: {\"id\": \"abc\", \"text_content\": \"Password:\", \"text_type\": \"password_label\", "
        "\"bbox\": {\"x\":0,\"y\":0,\"width\":100,\"height\":30}, \"confidence\": 0.95, "
        "\"attributes\": {\"is_pii\": true, \"is_critical\": false}}"
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
        elif text_type == "numeric_fragment":
            return RiskLevel.MEDIUM
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
            data = _parse_tool_input(text_json)

            # Unwrap if LLM sent wrapped format like {"text_data": [...]} or [...]
            if "text_data" in data and isinstance(data.get("text_data"), list) and data["text_data"]:
                data = data["text_data"][0]
            elif isinstance(data, list) and data:
                data = data[0]

            # Extract data
            text_id = data.get("id", "unknown")
            text_content = data.get("text_content", "")
            bbox = BoundingBox.from_raw(data.get("bbox", [0, 0, 0, 0]))
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
        "Calculates privacy risk for a single detected object. "
        "Input: a JSON string of ONE object dict with keys: id, object_class, bbox, "
        "confidence, attributes, contains_screen. "
        "Pass each object from the Object Data array individually. "
        "Example: {\"id\": \"abc\", \"object_class\": \"laptop\", "
        "\"bbox\": {\"x\":0,\"y\":0,\"width\":200,\"height\":150}, \"confidence\": 0.9, "
        "\"attributes\": {\"is_privacy_relevant\": true, \"risk_category\": \"screen_device\"}, "
        "\"contains_screen\": true}"
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

        # Screen devices: default LOW, VLM Phase 2 can escalate if sensitive content visible
        if contains_screen or risk_category == "screen_device":
            return RiskLevel.LOW

        # Vehicles (licsense plate risk)
        if risk_category == "vehicle":
            return RiskLevel.MEDIUM

        # Personal items, input devices
        if risk_category in ["personal_item", "input_device"]:
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
            data = _parse_tool_input(object_json)

            # Unwrap if LLM sent wrapped format like {"object_data": [...]} or [...]
            if "object_data" in data and isinstance(data.get("object_data"), list) and data["object_data"]:
                data = data["object_data"][0]
            elif isinstance(data, list) and data:
                data = data[0]

            # Extract pre-classified data from ObjectDetectionTool
            obj_id = data.get("id", "unknown")
            obj_label = data.get("object_class", "unknown")
            bbox = BoundingBox.from_raw(data.get("bbox", [0, 0, 0, 0]))
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

            requires_protection = risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]

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
        b1 = BoundingBox.from_raw(bbox1)
        b2 = BoundingBox.from_raw(bbox2)
        center1_x = b1.x + b1.width / 2
        center1_y = b1.y + b1.height / 2
        center2_x = b2.x + b2.width / 2
        center2_y = b2.y + b2.height / 2

        return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5

    def _is_near(self, bbox1: Dict, bbox2: Dict, threshold: float = 150) -> bool:
        """Check if bboxes are within threshold pixels."""
        return self._calculate_distance(bbox1, bbox2) < threshold

    def _run(self, detections_json: str) -> str:
        """Analyze spatial relationships."""
        try:
            data = _parse_tool_input(detections_json)
            faces = data.get("faces", [])
            texts = data.get("texts", [])
            objects = data.get("objects", [])

            # Adaptive threshold: 20% of image diagonal, clamped to [150, 400] px
            image_width = data.get("image_width", 1024)
            image_height = data.get("image_height", 768)
            image_diag = (image_width**2 + image_height**2) ** 0.5
            adaptive_threshold = max(150, min(400, image_diag * 0.20))

            escalations = []

            # Rule 1: Face near PII text (identity linkage)
            for face in faces:
                for text in texts:
                    if self._is_near(face["bbox"], text["bbox"], threshold=adaptive_threshold):
                        if text.get("attributes", {}).get("is_pii", False):
                            escalations.append({
                                "elements": [face["id"], text["id"]],
                                "escalation_amount": 2,
                                "reason": f"Identity linkage: face near {text.get('text_type', 'PII')}",
                                "relationship_type": "identity_linkage"
                            })

            # Rule 2: Face near screen — flag for VLM review but do NOT auto-escalate.
            # Screen devices default to LOW in Phase 1. VLM decides severity based
            # on whether the screen is actually on/showing content.
            for face in faces:
                for obj in objects:
                    if obj.get("contains_screen", False):
                        if self._is_near(face["bbox"], obj["bbox"], threshold=adaptive_threshold):
                            escalations.append({
                                "elements": [obj["id"]],  # Only the screen device
                                "escalation_amount": 0,   # Flag only, no auto-escalation
                                "reason": "Face near screen device (VLM: verify screen state)",
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
        b = BoundingBox.from_raw(bbox)
        center_x = (b.x + b.width / 2) / image_width
        center_y = (b.y + b.height / 2) / image_height

        if 0.3 <= center_x <= 0.7 and 0.3 <= center_y <= 0.7:
            return "center"
        elif center_x < 0.2 or center_x > 0.8 or center_y < 0.2 or center_y > 0.8:
            return "edge"
        else:
            return "intermediate"

    def _run(self, face_context_json: str) -> str:
        """Infer consent likelihood"""
        try:
            data = _parse_tool_input(face_context_json)
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
    config: Any = None

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def _escalate_severity(self, current_severity: str, escalation_amount: int) -> str:
        """Escalate risk severity."""
        risk_order = ["low", "medium", "high", "critical"]
        try:
            current_idx = risk_order.index(current_severity)
        except ValueError:
            current_idx = 1

        new_idx = max(0, min(current_idx + escalation_amount, len(risk_order) - 1))
        return risk_order[new_idx]

    def _run(self, escalation_data_json: str) -> str:
        """Apply escalations."""
        try:
            data = _parse_tool_input(escalation_data_json)

            assessments = data.get("assessments", [])
            escalations = data.get("escalations", [])

            assessment_map = {a["detection_id"]: a for a in assessments}
            escalations_applied = 0
            applied_to = set()  # (element_id, relationship_type) dedup

            for escalation in escalations:
                elements = escalation.get("elements", [])
                escalation_amount = escalation.get("escalation_amount", 1)
                reason = escalation.get("reason", "Contextual escalation")
                rel_type = escalation.get("relationship_type", "unknown")

                for element_id in elements:
                    key = (element_id, rel_type)
                    if key in applied_to:
                        continue
                    applied_to.add(key)

                    if element_id in assessment_map:
                        assessment = assessment_map[element_id]
                        old_severity = assessment.get("severity", "medium")
                        new_severity = self._escalate_severity(old_severity, escalation_amount)

                        if new_severity != old_severity:
                            assessment["severity"] = new_severity
                            if self.config is not None:
                                assessment["color_code"] = get_risk_color(self.config, new_severity)
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
            data = _parse_tool_input(assessments_json)
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
                bbox = BoundingBox.from_raw(assessment.get("bbox", [0, 0, 0, 0]))
                width = bbox.width
                height = bbox.height

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
            fallback = data.get("assessments", []) if 'data' in locals() else []
            return json.dumps({
                "error": str(e),
                "assessments": fallback,
                "original_count": len(fallback),
                "filtered_count": len(fallback)
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
            data = _parse_tool_input(assessments_json)
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
            fallback = data.get("assessments", []) if 'data' in locals() else []
            return json.dumps({
                "error": str(e),
                "assessments": fallback,
                "validated_count": 0,
                "corrections_made": 0
            })


#  Phase 2: VLM Review Tools
"""
Phase 2 tools are used by the VLM agent to review and modify Phase 1 assessments.
Each tool receives a reference to the shared assessments list and modifies it in-place.
The agent dynamically decides which tools to call based on visual evidence.
"""

class ReclassifyAssessmentTool(BaseTool):
    """Reclassify assessment severity based on visual evidence. Actual modifications in-place."""
    name: str = "reclassify_assessment"
    description: str = (
        "Reclassify the severity of an assessment based on visual evidence. "
        "Actually changes the severity, color code, and protection status in-place."
    )
    args_schema: Type[BaseModel] = ReclassifyAssessmentInput
    handle_tool_error: bool = True
    assessments: Any = None
    config: Any = None

    def __init__(self, assessments: List[Dict], config, **kwargs):
        super().__init__(**kwargs)
        self.assessments = assessments
        self.config = config

    def _run(self, index: int, severity: str, reason: str = "VLM visual review") -> str:
        """Reclassify an assessment's severity and update derived fields."""
        if index < 0 or index >= len(self.assessments):
            return json.dumps({"error": f"Invalid index {index}, valid range: 0-{len(self.assessments)-1}"})

        new_severity_lower = severity.lower()
        valid = {"critical", "high", "medium", "low"}
        if new_severity_lower not in valid:
            return json.dumps({"error": f"Invalid severity '{severity}', must be one of: {sorted(valid)}"})

        assessment = self.assessments[index]
        old_severity = assessment.get("severity", "low")

        # Guard: screen devices are pre-verified by focused VLM crop — do not override
        if assessment.get("element_type") == "object":
            factors = assessment.get("factors", {})
            if factors.get("contains_screen", False) or factors.get("risk_category") == "screen_device":
                return json.dumps({
                    "status": "blocked",
                    "index": index,
                    "element": assessment.get("element_description", "?"),
                    "message": f"Screen device [{index}] was verified by focused VLM crop. Do not reclassify."
                })

        # Guard: never downgrade faces with consent=none (bystanders are always CRITICAL)
        severity_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        if assessment.get("element_type") == "face":
            consent = assessment.get("consent_status", "none")
            if hasattr(consent, 'value'):
                consent = consent.value
            if consent == "none":
                new_rank = severity_rank.get(new_severity_lower, 0)
                old_rank = severity_rank.get(old_severity, 0)
                if new_rank < old_rank:
                    return json.dumps({
                        "status": "blocked",
                        "index": index,
                        "element": assessment.get("element_description", "?"),
                        "message": f"Cannot downgrade face [{index}] with consent=none. Bystander faces are always CRITICAL."
                    })

        # Guard: never downgrade CRITICAL/HIGH text items
        if assessment.get("element_type") == "text":
            old_rank = severity_rank.get(old_severity, 0)
            new_rank = severity_rank.get(new_severity_lower, 0)
            if new_rank < old_rank and old_rank >= 2:
                return json.dumps({
                    "status": "blocked",
                    "index": index,
                    "element": assessment.get("element_description", "?"),
                    "message": f"Cannot downgrade text [{index}] from {old_severity}. Phase 1 PII pattern match is authoritative."
                })

        # No-op: already at target severity
        if old_severity == new_severity_lower:
            return json.dumps({
                "status": "no_change",
                "index": index,
                "element": assessment.get("element_description", "?"),
                "message": f"Already at severity '{new_severity_lower}'. Do NOT repeat this action."
            })

        # Actually modify the assessment in-place
        assessment["severity"] = new_severity_lower
        assessment["color_code"] = get_risk_color(self.config, new_severity_lower)
        assessment["requires_protection"] = new_severity_lower in ("critical", "high")

        # Update reasoning with VLM evidence
        original_reasoning = assessment.get("reasoning", "Tool-based assessment")
        assessment["reasoning"] = f"{original_reasoning} -> VLM: {reason}"

        # Add VLM metadata to factors
        factors = assessment.get("factors", {})
        factors["vlm_reclassified"] = {
            "from": old_severity,
            "to": new_severity_lower,
            "reason": reason
        }
        assessment["factors"] = factors

        print(f"    RECLASSIFY: [{index}] {assessment.get('element_description', '?')} "
              f"{old_severity} -> {new_severity_lower} ({reason})")

        return json.dumps({
            "status": "success",
            "index": index,
            "element": assessment.get("element_description", "?"),
            "changed": f"{old_severity} -> {new_severity_lower}",
            "requires_protection": assessment["requires_protection"]
        })


class BatchReclassifyTool(BaseTool):
    """Reclassify multiple assessments in a single call. Does not shift indices."""
    name: str = "batch_reclassify"
    description: str = (
        "Reclassify the severity of MULTIPLE assessments at once. "
        "Pass a list of reclassifications, each with index, severity, and reason. "
        "This is more efficient than calling reclassify_assessment multiple times. "
        "Does NOT shift indices (unlike split_assessment)."
    )
    args_schema: Type[BaseModel] = BatchReclassifyInput
    handle_tool_error: bool = True
    assessments: Any = None
    config: Any = None

    def __init__(self, assessments: List[Dict], config, **kwargs):
        super().__init__(**kwargs)
        self.assessments = assessments
        self.config = config

    def _run(self, reclassifications: List[Dict]) -> str:
        """Apply multiple reclassifications in one call."""
        valid_severities = {"critical", "high", "medium", "low"}
        results = []

        for item in reclassifications:
            # Handle both dict and ReclassifyItem
            if hasattr(item, 'index'):
                index, severity, reason = item.index, item.severity, item.reason
            else:
                index = item.get("index", -1)
                severity = item.get("severity", "")
                reason = item.get("reason", "VLM visual review")

            # Validate index
            if index < 0 or index >= len(self.assessments):
                results.append(f"[{index}] ERROR: invalid index (0-{len(self.assessments)-1})")
                continue

            new_severity = severity.lower()
            if new_severity not in valid_severities:
                results.append(f"[{index}] ERROR: invalid severity '{severity}'")
                continue

            assessment = self.assessments[index]
            old_severity = assessment.get("severity", "low")

            # Skip no-ops
            if old_severity == new_severity:
                results.append(f"[{index}] SKIP: already {new_severity}")
                continue

            # Guard: screen devices are pre-verified by focused VLM crop
            if assessment.get("element_type") == "object":
                factors = assessment.get("factors", {})
                if factors.get("contains_screen", False) or factors.get("risk_category") == "screen_device":
                    results.append(
                        f"[{index}] BLOCKED: screen device verified by focused VLM crop. Do not reclassify."
                    )
                    continue

            # Guard: never downgrade faces with consent=none
            severity_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            if assessment.get("element_type") == "face":
                consent = assessment.get("consent_status", "none")
                if hasattr(consent, 'value'):
                    consent = consent.value
                if consent == "none":
                    new_rank = severity_rank.get(new_severity, 0)
                    old_rank = severity_rank.get(old_severity, 0)
                    if new_rank < old_rank:
                        results.append(
                            f"[{index}] BLOCKED: cannot downgrade face with consent=none. "
                            f"Bystander faces are always CRITICAL."
                        )
                        continue

            # Guard: never downgrade CRITICAL/HIGH text items
            severity_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            if assessment.get("element_type") == "text":
                old_rank = severity_rank.get(old_severity, 0)
                new_rank = severity_rank.get(new_severity, 0)
                if new_rank < old_rank and old_rank >= 2:
                    results.append(
                        f"[{index}] BLOCKED: cannot downgrade text from {old_severity}. "
                        f"Phase 1 PII match is authoritative."
                    )
                    continue

            # Guard: protect split-created value items from accidental downgrade
            det_id = str(assessment.get("detection_id", ""))
            desc = assessment.get("element_description", "")
            if "_split_" in det_id and desc.startswith("Text value:"):
                if new_severity in ("low", "medium") and old_severity in ("critical", "high"):
                    results.append(
                        f"[{index}] BLOCKED: '{desc}' is a split-created sensitive value. "
                        f"Do NOT downgrade from {old_severity}."
                    )
                    continue

            # Apply reclassification in-place
            assessment["severity"] = new_severity
            assessment["color_code"] = get_risk_color(self.config, new_severity)
            assessment["requires_protection"] = new_severity in ("critical", "high")

            original_reasoning = assessment.get("reasoning", "Tool-based assessment")
            assessment["reasoning"] = f"{original_reasoning} -> VLM: {reason}"

            factors = assessment.get("factors", {})
            factors["vlm_reclassified"] = {
                "from": old_severity,
                "to": new_severity,
                "reason": reason
            }
            assessment["factors"] = factors

            desc = assessment.get("element_description", "?")
            print(f"    RECLASSIFY: [{index}] {desc} {old_severity} -> {new_severity} ({reason})")
            results.append(f"[{index}] {old_severity} -> {new_severity}")

        return json.dumps({
            "status": "success",
            "applied": len([r for r in results if "->" in r]),
            "skipped": len([r for r in results if "SKIP" in r]),
            "errors": len([r for r in results if "ERROR" in r]),
            "details": results
        })


class SplitAssessmentTool(BaseTool):
    """Split a text assessment into separate parts. Actually replaces in the list."""
    name: str = "split_assessment"
    description: str = (
        "Split a text assessment into separate parts (e.g., label vs sensitive value). "
        "Actually replaces the original assessment with multiple new assessments. "
        "Uses OCR re-crop for precise bounding boxes on each split part. "
        "WARNING: This shifts all indices after the split point."
    )
    args_schema: Type[BaseModel] = SplitAssessmentInput
    handle_tool_error: bool = True
    assessments: Any = None
    config: Any = None
    ocr_reader: Any = None
    image_path: Optional[str] = None

    def __init__(self, assessments: List[Dict], config, ocr_reader=None, image_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.assessments = assessments
        self.config = config
        self.ocr_reader = ocr_reader
        self.image_path = image_path

    def _ocr_recrop_bboxes(self, original_bbox: List, part_descriptions: List[str]) -> List[List]:
        """
        Re-run OCR on a cropped region to get precise sub-bboxes for split parts.

        Args:
            original_bbox: [x, y, w, h] of the original assessment
            part_descriptions: List of text descriptions for each split part

        Returns:
            List of [x, y, w, h] bboxes (absolute coords), one per part.
            Falls back to original bbox for parts that can't be matched.
        """
        if not self.ocr_reader or not self.image_path:
            return [list(original_bbox)] * len(part_descriptions)

        try:
            image = Image.open(self.image_path)
            ox, oy, ow, oh = original_bbox

            # Add padding around crop for better OCR (10% each side, clamped to image)
            img_w, img_h = image.size
            pad_x = max(int(ow * 0.1), 5)
            pad_y = max(int(oh * 0.1), 5)
            crop_x1 = max(0, ox - pad_x)
            crop_y1 = max(0, oy - pad_y)
            crop_x2 = min(img_w, ox + ow + pad_x)
            crop_y2 = min(img_h, oy + oh + pad_y)

            crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            crop_array = np.array(crop)

            # Run OCR with tight word separation for granular detection
            ocr_results = self.ocr_reader.readtext(
                crop_array,
                width_ths=0.1,      # Tight word separation
                paragraph=False,
            )

            if not ocr_results:
                return [list(original_bbox)] * len(part_descriptions)

            # Convert OCR results to absolute coordinates
            ocr_words = []
            for bbox_points, text, conf in ocr_results:
                if conf < 0.2:
                    continue
                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]
                # Convert crop-relative to absolute image coordinates
                abs_bbox = [
                    int(min(x_coords) + crop_x1),
                    int(min(y_coords) + crop_y1),
                    int(max(x_coords) - min(x_coords)),
                    int(max(y_coords) - min(y_coords))
                ]
                ocr_words.append({"text": text.strip(), "bbox": abs_bbox})

            # Match each part description to OCR words
            result_bboxes = []
            used_indices = set()

            for desc in part_descriptions:
                desc_lower = desc.lower().strip()

                # Multi-word descriptions: try merge match FIRST to avoid
                # partial single-word matches (e.g., "bank" matching "bank account")
                if ' ' in desc_lower:
                    merged = self._try_merge_match(desc_lower, ocr_words, used_indices)
                    if merged is not None:
                        for idx in merged["used"]:
                            used_indices.add(idx)
                        result_bboxes.append(merged["bbox"])
                        continue

                # Single-word match (or multi-word fallback)
                best_match_idx = None
                best_score = 0

                for wi, word in enumerate(ocr_words):
                    if wi in used_indices:
                        continue
                    word_lower = word["text"].lower().strip()

                    # Exact match
                    if word_lower == desc_lower or desc_lower in word_lower or word_lower in desc_lower:
                        score = len(word_lower) / max(len(desc_lower), 1)
                        # Prefer closer length matches
                        if score > best_score:
                            best_score = score
                            best_match_idx = wi

                if best_match_idx is not None:
                    used_indices.add(best_match_idx)
                    result_bboxes.append(ocr_words[best_match_idx]["bbox"])
                else:
                    # No match found — try merging multiple OCR words (single-word fallback)
                    merged = self._try_merge_match(desc_lower, ocr_words, used_indices)
                    if merged:
                        for idx in merged["used"]:
                            used_indices.add(idx)
                        result_bboxes.append(merged["bbox"])
                    else:
                        result_bboxes.append(list(original_bbox))

            return result_bboxes

        except Exception as e:
            print(f"    [OCR re-crop] Failed: {e}, using original bbox")
            return [list(original_bbox)] * len(part_descriptions)

    def _try_merge_match(self, target: str, ocr_words: List[Dict], used: set) -> Optional[Dict]:
        """
        Try to match a description by merging consecutive OCR words.
        E.g., target="8765 4321 0987" might match OCR words ["8765", "4321", "0987"].

        Finds the LONGEST matching merge (most OCR words covered) to avoid
        partial matches that miss trailing words (e.g., missing "0987").

        Returns:
            Dict with "bbox" (merged) and "used" (set of indices), or None.
        """
        available = [(i, w) for i, w in enumerate(ocr_words) if i not in used]
        best_match = None
        best_length = 0

        for start in range(len(available)):
            for length in range(2, min(6, len(available) - start + 1)):
                group = available[start:start + length]
                merged_text = " ".join(w["text"].lower().strip() for _, w in group)
                # Check if merged text contains target or vice versa
                if target in merged_text or merged_text in target:
                    # Prefer longest match (covers more OCR words)
                    if length > best_length:
                        all_x1 = min(w["bbox"][0] for _, w in group)
                        all_y1 = min(w["bbox"][1] for _, w in group)
                        all_x2 = max(w["bbox"][0] + w["bbox"][2] for _, w in group)
                        all_y2 = max(w["bbox"][1] + w["bbox"][3] for _, w in group)
                        best_match = {
                            "bbox": [all_x1, all_y1, all_x2 - all_x1, all_y2 - all_y1],
                            "used": set(i for i, _ in group)
                        }
                        best_length = length

        return best_match

    def _spatial_split_bbox(self, original_bbox, num_parts, original_text=""):
        """
        Split bbox proportionally based on text content.
        For "Label: Value" patterns, estimate split point from colon position.
        Fallback: equal width division.
        """
        ox, oy, ow, oh = original_bbox

        if num_parts == 2 and ":" in original_text:
            colon_pos = original_text.index(":")
            ratio = (colon_pos + 1) / max(len(original_text), 1)
            ratio = max(0.2, min(ratio, 0.6))
            split_x = int(ox + ow * ratio)
            return [
                [ox, oy, split_x - ox, oh],
                [split_x, oy, ox + ow - split_x, oh]
            ]

        part_w = ow // max(num_parts, 1)
        return [[ox + i * part_w, oy, part_w, oh] for i in range(num_parts)]

    def _run(self, index: int, parts: List[Dict]) -> str:
        """Split an assessment into multiple parts and replace in the list."""
        if index < 0 or index >= len(self.assessments):
            return json.dumps({"error": f"Invalid index {index}, valid range: 0-{len(self.assessments)-1}"})
        if not parts or len(parts) < 2:
            return json.dumps({"error": "Need at least 2 parts for a split"})

        original = self.assessments[index]

        # Guard: only text assessments should be split
        if original.get("element_type") != "text":
            return json.dumps({
                "error": f"Cannot split {original.get('element_type', 'unknown')} assessment. "
                         f"Only text assessments can be split. Use reclassify_assessment instead."
            })

        # Guard: block cascading splits (items already created by a previous split)
        original_id = original.get("detection_id", "unknown")
        if "_split_" in str(original_id):
            return json.dumps({
                "error": f"Item [{index}] was already created by a previous split. "
                         f"Do NOT re-split. Only split original Phase 1 items."
            })

        # Guard: only split text that contains a "Label: Value" or "Label:Value" pattern
        # (colon followed by meaningful text/digits after it)
        desc = original.get("element_description", "")
        # Strip "Text: " prefix to get raw content
        raw_text = desc
        for prefix in ["Text: ", "Text label: ", "Text value: "]:
            if raw_text.startswith(prefix):
                raw_text = raw_text[len(prefix):]
                break
        # Check for "Label: Value" (with space) or "Label:Digits" (no space, digits after colon)
        has_colon_space = ": " in raw_text and len(raw_text.split(": ", 1)[1].strip()) >= 2
        has_colon_digits = bool(re.search(r':\s*\d{2,}', raw_text))  # colon + optional space + 2+ digits
        if not has_colon_space and not has_colon_digits:
            return json.dumps({
                "error": f"Item [{index}] '{desc}' is not a composite 'Label: Value' text. "
                         f"Use batch_reclassify to change its severity instead."
            })
        original_bbox = original.get("bbox", [0, 0, 0, 0])

        # Convert parts for processing
        processed_parts = []
        for part in parts:
            if isinstance(part, BaseModel):
                part = part.model_dump()
            processed_parts.append(part)

        # Get precise sub-bboxes via OCR re-crop, with spatial split fallback
        original_text = original.get("element_description", "").replace("Text: ", "", 1)
        part_descriptions = [
            p.get("element_description", original.get("element_description", ""))
            for p in processed_parts
        ]
        sub_bboxes = self._ocr_recrop_bboxes(original_bbox, part_descriptions)

        # If OCR matching failed (all bboxes same as original), use spatial split
        all_same = all(b == list(original_bbox) for b in sub_bboxes)
        if all_same:
            sub_bboxes = self._spatial_split_bbox(
                original_bbox, len(processed_parts), original_text
            )

        new_parts = []
        for j, part in enumerate(processed_parts):
            new_assessment = dict(original)
            new_assessment["detection_id"] = f"{original_id}_split_{j}"

            desc = part.get("element_description", original.get("element_description", "Unknown"))
            new_sev = part.get("severity", original.get("severity", "low")).lower()

            # Normalize description: ensure consistent "Text label:" / "Text value:" prefix
            # Handle VLM-provided "Label: X" / "Value: X" → convert to "Text label: X" / "Text value: X"
            if desc.startswith("Label: "):
                desc = f"Text label: {desc[7:]}"
            elif desc.startswith("Value: "):
                desc = f"Text value: {desc[7:]}"
            elif not any(desc.startswith(p) for p in ["Text label: ", "Text value: ", "Text: "]):
                if new_sev in ("critical", "high"):
                    desc = f"Text value: {desc}"
                else:
                    desc = f"Text label: {desc}"
            new_assessment["element_description"] = desc
            new_assessment["severity"] = new_sev
            new_assessment["color_code"] = get_risk_color(self.config, new_sev)
            new_assessment["requires_protection"] = part.get(
                "requires_protection", new_sev in ("critical", "high"))

            # Assign precise sub-bbox from OCR re-crop
            new_assessment["bbox"] = sub_bboxes[j]

            visual_reason = part.get("visual_reason", "")
            if not visual_reason or visual_reason.lower() == "vlm split":
                if new_sev in ("critical", "high"):
                    visual_reason = "Sensitive value requiring protection"
                else:
                    visual_reason = "Label/context only, low risk"
            new_assessment["reasoning"] = f"VLM review: {visual_reason}"

            factors = new_assessment.get("factors", {})
            factors["vlm_split"] = {
                "original_id": original_id,
                "part_index": j,
                "total_parts": len(processed_parts),
                "original_bbox": list(original_bbox),
                "sub_bbox": sub_bboxes[j],
                "bbox_precise": sub_bboxes[j] != list(original_bbox)
            }
            new_assessment["factors"] = factors
            new_parts.append(new_assessment)

        # Actually replace in the list (modifies in-place)
        self.assessments[index:index + 1] = new_parts

        descriptions = [p.get("element_description", "?") for p in processed_parts]
        bbox_info = ["precise" if sub_bboxes[i] != list(original_bbox) else "original"
                     for i in range(len(processed_parts))]
        print(f"    SPLIT: [{index}] {original.get('element_description', '?')} -> {descriptions}")
        print(f"    BBOX:  {bbox_info}")

        return json.dumps({
            "status": "success",
            "original_index": index,
            "split_into": descriptions,
            "new_count": len(new_parts),
            "total_assessments": len(self.assessments),
            "bbox_precision": bbox_info,
            "new_indices": {
                descriptions[i]: index + i for i in range(len(new_parts))
            },
            "note": f"Indices {index+len(new_parts)} onwards have shifted by {len(new_parts)-1}."
        })


class GetCurrentAssessmentsTool(BaseTool):
    """Get the current state of all assessments with their indices."""
    name: str = "get_current_assessments"
    description: str = (
        "Get the current state of all assessments with their indices. "
        "Call this after splits to see updated indices, or at any time to review."
    )
    handle_tool_error: bool = True
    assessments: Any = None

    def __init__(self, assessments: List[Dict], **kwargs):
        super().__init__(**kwargs)
        self.assessments = assessments

    def _run(self, tool_input: str = "") -> str:
        """Return compact summary of all current assessments."""
        lines = []
        for i, a in enumerate(self.assessments):
            lines.append(
                f"[{i}] {a.get('element_type', '?')} | "
                f"{a.get('element_description', '?')} | "
                f"{a.get('severity', 'low')}"
            )
        return f"Total: {len(self.assessments)}\n" + "\n".join(lines)


class ValidateAssessmentsTool(BaseTool):
    """Validate and finalize all assessments."""
    name: str = "validate_assessments"
    description: str = (
        "Validate and finalize all assessments. "
        "Ensures severity levels match protection requirements, then finalizes the review. "
        "Call this ONCE after all reclassifications and splits are done."
    )
    handle_tool_error: bool = True
    assessments: Any = None
    already_validated: bool = False

    def __init__(self, assessments: List[Dict], **kwargs):
        super().__init__(**kwargs)
        self.assessments = assessments

    def _run(self, tool_input: str = "") -> str:
        """Fix consistency and finalize: critical/high must require protection, low must not."""
        if self.already_validated:
            return json.dumps({
                "status": "already_finalized",
                "message": "Already validated and finalized. Do NOT call any more tools."
            })

        corrections = 0
        for a in self.assessments:
            sev = a.get("severity", "low")
            prot = a.get("requires_protection", False)
            if sev in ("critical", "high") and not prot:
                a["requires_protection"] = True
                corrections += 1
            if sev == "low" and prot:
                a["requires_protection"] = False
                corrections += 1

        self.already_validated = True

        # Auto-finalize: generate summary
        summary = {"critical": 0, "high": 0, "medium": 0, "low": 0, "requiring_protection": 0}
        for a in self.assessments:
            sev = a.get("severity", "low").lower()
            if sev in summary:
                summary[sev] += 1
            if a.get("requires_protection", False):
                summary["requiring_protection"] += 1

        return json.dumps({
            "status": "validated_and_finalized",
            "corrections": corrections,
            "total_assessments": len(self.assessments),
            "summary": summary,
            "message": "Review complete. Do NOT call any more tools."
        })
