"""
Agent 2: Risk Assessment Agent (Privacy Analyzer)

Purpose: Evaluate privacy risk based on detection results and user privacy profile
Input: DetectionResults from Agent 1
Output: RiskAnalysisResult (JSON) -> feeds into Agent 2.5 and Risk Analysis Update

Key Responsibilities:
1. Filter false positives - Remove non-privacy-relevant detections
2. Apply user's PrivacyProfile sensitivities
3. Assign RiskLevel (critical/high/medium/low) and RiskType to each detection
4. Use local VLM for contextual analysis
5. Output RiskAnalysisResult with all assessed risks
"""
import time
import json
from PIL import Image
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from utils.models import (
    DetectionResults,
    FaceDetection,
    TextDetection,
    ObjectDetection,
    RiskAssessment,
    RiskAnalysisResult,
    RiskLevel,
    RiskType,
    PrivacyProfile,
    BoundingBox,
    PersonClassification,
    ConsentStatus
)
from agents.local_wrapper import VisionLLM


class RiskAssessmentAgent:
    """
    Agent 2: Risk Assessment Agent (Privacy Analyzer)

    Evaluates privacy risks based on:
    - Detection results from Agent 1
    - User's privacy profile sensitivities
    - Contextual analysis using local VLM

    Architecture:
    - Rule-based risk classification for known patterns
    - VLM-assisted contextual analysis for ambiguous cases
    - Privacy profile-aware risk level adjustment
    """

    # Objects that are typically NOT privacy-relevant (false positive filter)
    NON_PRIVACY_OBJECTS = {
        "mouse", "keyboard", "book", "cup", "bottle", "chair", "couch",
        "dining table", "potted plant", "vase", "clock", "scissors",
        "teddy bear", "toothbrush", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "orange", "broccoli", "carrot", "pizza",
        "donut", "cake", "bed", "toilet", "sink", "refrigerator",
        "oven", "microwave", "toaster"
    }

    # Objects that ARE privacy-relevant
    PRIVACY_RELEVANT_OBJECTS = {
        # Screen devices - may show sensitive information
        "laptop": {"risk_type": RiskType.INFORMATION_DISCLOSURE, "base_level": RiskLevel.MEDIUM},
        "tv": {"risk_type": RiskType.INFORMATION_DISCLOSURE, "base_level": RiskLevel.LOW},
        "cell phone": {"risk_type": RiskType.INFORMATION_DISCLOSURE, "base_level": RiskLevel.MEDIUM},
        "monitor": {"risk_type": RiskType.INFORMATION_DISCLOSURE, "base_level": RiskLevel.MEDIUM},

        # Vehicles - license plates
        "car": {"risk_type": RiskType.LOCATION_EXPOSURE, "base_level": RiskLevel.MEDIUM},
        "truck": {"risk_type": RiskType.LOCATION_EXPOSURE, "base_level": RiskLevel.MEDIUM},
        "bus": {"risk_type": RiskType.LOCATION_EXPOSURE, "base_level": RiskLevel.MEDIUM},
        "motorcycle": {"risk_type": RiskType.LOCATION_EXPOSURE, "base_level": RiskLevel.MEDIUM},

        # Personal items - may reveal identity or location
        "backpack": {"risk_type": RiskType.CONTEXT_EXPOSURE, "base_level": RiskLevel.LOW},
        "handbag": {"risk_type": RiskType.CONTEXT_EXPOSURE, "base_level": RiskLevel.LOW},
        "suitcase": {"risk_type": RiskType.CONTEXT_EXPOSURE, "base_level": RiskLevel.LOW},
    }

    # Text type to risk mapping
    TEXT_RISK_MAPPING = {
        # Critical - must always be protected
        "ssn": {"risk_type": RiskType.INFORMATION_DISCLOSURE, "base_level": RiskLevel.CRITICAL},
        "credit_card": {"risk_type": RiskType.INFORMATION_DISCLOSURE, "base_level": RiskLevel.CRITICAL},
        "password": {"risk_type": RiskType.INFORMATION_DISCLOSURE, "base_level": RiskLevel.CRITICAL},
        "bank_account": {"risk_type": RiskType.INFORMATION_DISCLOSURE, "base_level": RiskLevel.CRITICAL},

        # PII - high risk
        "phone_number": {"risk_type": RiskType.INFORMATION_DISCLOSURE, "base_level": RiskLevel.HIGH},
        "email": {"risk_type": RiskType.INFORMATION_DISCLOSURE, "base_level": RiskLevel.HIGH},
        "address": {"risk_type": RiskType.LOCATION_EXPOSURE, "base_level": RiskLevel.HIGH},

        # General text - context dependent
        "general_text": {"risk_type": RiskType.INFORMATION_DISCLOSURE, "base_level": RiskLevel.LOW},
    }

    # Risk level colors (from config)
    RISK_COLORS = {
        RiskLevel.CRITICAL: "#FF0000",  # Red
        RiskLevel.HIGH: "#FF6600",      # Orange
        RiskLevel.MEDIUM: "#FFD700",    # Gold
        RiskLevel.LOW: "#90EE90"        # Light green
    }

    def __init__(self, config, use_vlm: bool = True, vlm_model: str = "llava-phi3"):
        """
        Initialize Risk Assessment Agent

        Args:
            config: Configuration object
            use_vlm: Whether to use VLM for contextual analysis (default True)
            vlm_model: Ollama model name for vision tasks
        """
        self.config = config
        self.use_vlm = use_vlm
        self.vlm = None

        print(f"Initializing Risk Assessment Agent")

        # Initialize VLM for contextual analysis if enabled
        if use_vlm:
            try:
                self.vlm = VisionLLM(model=vlm_model)
                print(f"  VLM initialized: {vlm_model}")
            except Exception as e:
                print(f"  Warning: Failed to initialize VLM: {e}")
                print(f"  Falling back to rule-based analysis only")
                self.vlm = None

        # Load risk level config
        self._load_risk_config()

    def _load_risk_config(self):
        """Load risk level configuration from config"""
        try:
            self.risk_config = {
                RiskLevel.CRITICAL: {
                    "color": self.config.risk_levels.critical.color,
                    "requires_protection": self.config.risk_levels.critical.requires_protection,
                    "user_can_override": self.config.risk_levels.critical.user_can_override
                },
                RiskLevel.HIGH: {
                    "color": self.config.risk_levels.high.color,
                    "requires_protection": self.config.risk_levels.high.requires_protection,
                    "user_can_override": self.config.risk_levels.high.user_can_override
                },
                RiskLevel.MEDIUM: {
                    "color": self.config.risk_levels.medium.color,
                    "requires_protection": self.config.risk_levels.medium.requires_protection,
                    "user_can_override": self.config.risk_levels.medium.user_can_override
                },
                RiskLevel.LOW: {
                    "color": self.config.risk_levels.low.color,
                    "requires_protection": self.config.risk_levels.low.requires_protection,
                    "user_can_override": self.config.risk_levels.low.user_can_override
                }
            }
            # Update colors from config
            self.RISK_COLORS = {
                RiskLevel.CRITICAL: self.config.risk_levels.critical.color,
                RiskLevel.HIGH: self.config.risk_levels.high.color,
                RiskLevel.MEDIUM: self.config.risk_levels.medium.color,
                RiskLevel.LOW: self.config.risk_levels.low.color
            }
        except Exception as e:
            print(f"  Warning: Could not load risk config: {e}")
            # Use defaults

    def run(
        self,
        detection_results: DetectionResults,
        privacy_profile: PrivacyProfile
    ) -> RiskAnalysisResult:
        """
        Main risk assessment pipeline

        Args:
            detection_results: Detection results from Agent 1
            privacy_profile: User's privacy profile with sensitivity settings

        Returns:
            RiskAnalysisResult with all assessed risks
        """
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Risk Assessment Agent - Processing")
        print(f"{'='*60}")
        print(f"Input: {detection_results.total_detections} detections")
        print(f"  - Faces: {len(detection_results.faces)}")
        print(f"  - Text regions: {len(detection_results.text_regions)}")
        print(f"  - Objects: {len(detection_results.objects)}")

        # Initialize result
        risk_assessments: List[RiskAssessment] = []

        # Load image for VLM context analysis if needed
        if self.vlm and Path(detection_results.image_path).exists():
            try:
                image = Image.open(detection_results.image_path).convert("RGB")
                self.vlm.set_image(image)
            except Exception as e:
                print(f"  Warning: Could not load image for VLM: {e}")

        # Stage 1: Assess Face Risks
        print(f"\n--- Stage 1: Face Risk Assessment ---")
        face_assessments = self._assess_faces(
            detection_results.faces,
            privacy_profile
        )
        risk_assessments.extend(face_assessments)
        print(f"  Assessed {len(face_assessments)} face risks")

        # Stage 2: Assess Text Risks
        print(f"\n--- Stage 2: Text Risk Assessment ---")
        text_assessments = self._assess_text(
            detection_results.text_regions,
            privacy_profile
        )
        risk_assessments.extend(text_assessments)
        print(f"  Assessed {len(text_assessments)} text risks")

        # Stage 3: Assess Object Risks (with filtering)
        print(f"\n--- Stage 3: Object Risk Assessment ---")
        object_assessments = self._assess_objects(
            detection_results.objects,
            privacy_profile
        )
        risk_assessments.extend(object_assessments)
        print(f"  Assessed {len(object_assessments)} object risks (filtered)")

        # Stage 4: VLM Contextual Analysis (if enabled)
        if self.vlm:
            print(f"\n--- Stage 4: VLM Contextual Analysis ---")
            risk_assessments = self._vlm_contextual_analysis(
                risk_assessments,
                detection_results
            )

        # Calculate overall risk level
        overall_risk = self._calculate_overall_risk(risk_assessments)

        # Count faces pending identity verification
        faces_pending = sum(
            1 for r in risk_assessments
            if r.element_type == "face" and r.consent_status == ConsentStatus.UNCLEAR
        )

        # Count confirmed risks (high or critical)
        confirmed_risks = sum(
            1 for r in risk_assessments
            if r.severity in [RiskLevel.CRITICAL, RiskLevel.HIGH]
        )

        # Build result
        processing_time = (time.time() - start_time) * 1000

        result = RiskAnalysisResult(
            image_path=detection_results.image_path,
            risk_assessments=risk_assessments,
            overall_risk_level=overall_risk,
            faces_pending_identity=faces_pending,
            confirmed_risks=confirmed_risks,
            processimg_time_ms=processing_time
        )

        # Print summary
        self._print_summary(result)

        return result

    def _assess_faces(
        self,
        faces: List[FaceDetection],
        profile: PrivacyProfile
    ) -> List[RiskAssessment]:
        """
        Assess privacy risks for detected faces

        Face risk depends on:
        - Face size (large = likely main subject, small = likely bystander)
        - User's identity sensitivity settings
        - Whether the face can be identified
        """
        assessments = []

        for face in faces:
            # Determine face classification based on size
            # Note: Agent 2.5 will refine this with identity matching
            if face.size == "large":
                # Large face - likely the main subject
                classification = PersonClassification.PRIMARY_SUBJECT
                sensitivity_key = "own_face"
            elif face.size == "medium":
                # Medium face - could be friend/family or bystander
                classification = PersonClassification.KNOWN_CONTACT
                sensitivity_key = "friend_faces"
            else:
                # Small face - likely a bystander
                classification = PersonClassification.BYSTANDER
                sensitivity_key = "bystander_faces"

            # Get user's sensitivity for this type of face
            sensitivity = profile.identity_sensitivity.get(sensitivity_key, "medium")
            base_level = self._sensitivity_to_risk_level(sensitivity)

            # Create assessment
            assessment = RiskAssessment(
                detection_id=face.id,
                element_type="face",
                element_description=f"Face ({face.size}, {face.clarity or 'unknown'} clarity)",
                risk_type=RiskType.IDENTITY_EXPOSURE,
                severity=base_level,
                color_code=self.RISK_COLORS[base_level],
                reasoning=f"Face detected with {face.confidence:.1%} confidence. "
                         f"Size: {face.size}. User sensitivity for {sensitivity_key}: {sensitivity}.",
                user_sensitivity_applied=sensitivity,
                bbox=face.bbox,
                requires_protection=base_level in [RiskLevel.CRITICAL, RiskLevel.HIGH],
                legal_requirement=classification == PersonClassification.BYSTANDER,
                classification=classification,
                consent_status=ConsentStatus.UNCLEAR,  # Will be resolved by Agent 2.5
                consent_confidence=0.0
            )
            assessments.append(assessment)

        return assessments

    def _assess_text(
        self,
        text_regions: List[TextDetection],
        profile: PrivacyProfile
    ) -> List[RiskAssessment]:
        """
        Assess privacy risks for detected text

        Text risk depends on:
        - Text type (SSN, credit card, phone, email, address, etc.)
        - User's information/location sensitivity settings
        - Whether text contains critical/PII data
        """
        assessments = []

        for text in text_regions:
            # Get text type risk mapping
            text_type = text.text_type or "general_text"
            risk_mapping = self.TEXT_RISK_MAPPING.get(
                text_type,
                self.TEXT_RISK_MAPPING["general_text"]
            )

            risk_type = risk_mapping["risk_type"]
            base_level = risk_mapping["base_level"]

            # Apply user sensitivity
            sensitivity_key = self._get_text_sensitivity_key(text_type)
            sensitivity_dict = (
                profile.location_sensitivity
                if risk_type == RiskType.LOCATION_EXPOSURE
                else profile.information_sensitivity
            )
            sensitivity = sensitivity_dict.get(sensitivity_key, "medium")

            # Adjust risk level based on sensitivity
            adjusted_level = self._adjust_risk_level(base_level, sensitivity)

            # Check attributes for additional context
            is_critical = text.attributes.get("is_critical", False)
            is_pii = text.attributes.get("is_pii", False)

            # Critical/PII items should never go below HIGH
            if is_critical and adjusted_level not in [RiskLevel.CRITICAL]:
                adjusted_level = RiskLevel.CRITICAL
            elif is_pii and adjusted_level == RiskLevel.LOW:
                adjusted_level = RiskLevel.MEDIUM

            # Truncate text content for display
            content_preview = text.text_content[:30] + "..." if len(text.text_content) > 30 else text.text_content

            assessment = RiskAssessment(
                detection_id=text.id,
                element_type="text",
                element_description=f"Text: '{content_preview}' ({text_type})",
                risk_type=risk_type,
                severity=adjusted_level,
                color_code=self.RISK_COLORS[adjusted_level],
                reasoning=f"Text type: {text_type}. "
                         f"{'Critical data detected. ' if is_critical else ''}"
                         f"{'PII detected. ' if is_pii else ''}"
                         f"User sensitivity for {sensitivity_key}: {sensitivity}.",
                user_sensitivity_applied=sensitivity,
                bbox=text.bbox,
                requires_protection=adjusted_level in [RiskLevel.CRITICAL, RiskLevel.HIGH] or is_critical,
                legal_requirement=is_critical
            )
            assessments.append(assessment)

        return assessments

    def _assess_objects(
        self,
        objects: List[ObjectDetection],
        profile: PrivacyProfile
    ) -> List[RiskAssessment]:
        """
        Assess privacy risks for detected objects

        Filters out non-privacy-relevant objects and assesses the rest
        """
        assessments = []
        filtered_count = 0

        for obj in objects:
            obj_class = obj.object_class.lower()

            # Filter out non-privacy-relevant objects
            if obj_class in self.NON_PRIVACY_OBJECTS:
                filtered_count += 1
                continue

            # Check if object is privacy-relevant
            if obj_class in self.PRIVACY_RELEVANT_OBJECTS:
                risk_info = self.PRIVACY_RELEVANT_OBJECTS[obj_class]
                risk_type = risk_info["risk_type"]
                base_level = risk_info["base_level"]
            else:
                # Unknown object - check attributes
                if obj.attributes.get("is_privacy_relevant", False):
                    risk_type = RiskType.CONTEXT_EXPOSURE
                    base_level = RiskLevel.LOW
                else:
                    filtered_count += 1
                    continue

            # Get appropriate sensitivity
            sensitivity_key = self._get_object_sensitivity_key(obj_class, risk_type)
            sensitivity_dict = self._get_sensitivity_dict(profile, risk_type)
            sensitivity = sensitivity_dict.get(sensitivity_key, "medium")

            # Adjust based on user sensitivity
            adjusted_level = self._adjust_risk_level(base_level, sensitivity)

            # Check for screen content
            if obj.contains_screen:
                # Screens should be at least medium risk
                if adjusted_level == RiskLevel.LOW:
                    adjusted_level = RiskLevel.MEDIUM

            assessment = RiskAssessment(
                detection_id=obj.id,
                element_type="object",
                element_description=f"Object: {obj.object_class} "
                                   f"({'screen' if obj.contains_screen else 'no screen'})",
                risk_type=risk_type,
                severity=adjusted_level,
                color_code=self.RISK_COLORS[adjusted_level],
                reasoning=f"Object class: {obj.object_class}. "
                         f"{'Contains screen. ' if obj.contains_screen else ''}"
                         f"Risk category: {obj.attributes.get('risk_category', 'other')}. "
                         f"User sensitivity for {sensitivity_key}: {sensitivity}.",
                user_sensitivity_applied=sensitivity,
                bbox=obj.bbox,
                requires_protection=adjusted_level in [RiskLevel.CRITICAL, RiskLevel.HIGH],
                legal_requirement=False
            )
            assessments.append(assessment)

        if filtered_count > 0:
            print(f"    Filtered {filtered_count} non-privacy-relevant objects")

        return assessments

    def _vlm_contextual_analysis(
        self,
        assessments: List[RiskAssessment],
        detection_results: DetectionResults
    ) -> List[RiskAssessment]:
        """
        Use VLM to perform contextual analysis and refine risk assessments

        The VLM can:
        - Identify context that increases/decreases risk
        - Detect additional privacy concerns not caught by detectors
        - Provide reasoning for risk level adjustments
        """
        if not self.vlm:
            return assessments

        try:
            # Build prompt for VLM
            prompt = self._build_vlm_prompt(assessments)

            # Get VLM analysis
            response = self.vlm.invoke(prompt)

            # Parse and apply VLM insights
            updated_assessments = self._apply_vlm_insights(assessments, response)

            print(f"  VLM contextual analysis complete")

            return updated_assessments

        except Exception as e:
            print(f"  Warning: VLM analysis failed: {e}")
            return assessments

    def _build_vlm_prompt(self, assessments: List[RiskAssessment]) -> str:
        """Build prompt for VLM contextual analysis"""
        prompt = """Analyze this image for privacy risks. I have detected the following elements:

"""
        for i, assessment in enumerate(assessments, 1):
            prompt += f"{i}. {assessment.element_type}: {assessment.element_description}\n"
            prompt += f"   Current risk: {assessment.severity.value}\n"

        prompt += """
Please analyze the image context and answer:
1. Are there any additional privacy concerns not listed above? (e.g., reflective surfaces showing sensitive info, background elements revealing location)
2. Should any of the current risk levels be adjusted based on context? (e.g., face on ID card = higher risk)
3. Is there any text visible that was not detected?

Respond in JSON format:
{
    "additional_concerns": ["concern1", "concern2"],
    "risk_adjustments": [{"index": 1, "new_level": "high", "reason": "..."}],
    "missed_text": ["text1", "text2"],
    "overall_context": "brief description of the scene context"
}
"""
        return prompt

    def _apply_vlm_insights(
        self,
        assessments: List[RiskAssessment],
        vlm_response: str
    ) -> List[RiskAssessment]:
        """Parse VLM response and apply insights to assessments"""
        try:
            # Try to extract JSON from response
            json_start = vlm_response.find("{")
            json_end = vlm_response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = vlm_response[json_start:json_end]
                insights = json.loads(json_str)

                # Apply risk adjustments
                if "risk_adjustments" in insights:
                    for adjustment in insights["risk_adjustments"]:
                        idx = adjustment.get("index", 0) - 1
                        if 0 <= idx < len(assessments):
                            new_level_str = adjustment.get("new_level", "").lower()
                            reason = adjustment.get("reason", "VLM contextual analysis")

                            # Map string to RiskLevel
                            level_map = {
                                "critical": RiskLevel.CRITICAL,
                                "high": RiskLevel.HIGH,
                                "medium": RiskLevel.MEDIUM,
                                "low": RiskLevel.LOW
                            }

                            if new_level_str in level_map:
                                old_level = assessments[idx].severity
                                new_level = level_map[new_level_str]

                                # Only escalate, never reduce from VLM
                                if self._risk_level_priority(new_level) < self._risk_level_priority(old_level):
                                    assessments[idx].severity = new_level
                                    assessments[idx].color_code = self.RISK_COLORS[new_level]
                                    assessments[idx].reasoning += f" VLM adjustment: {reason}"
                                    print(f"    Adjusted assessment {idx+1}: {old_level.value} -> {new_level.value}")

                # Log additional concerns
                if insights.get("additional_concerns"):
                    print(f"    VLM identified additional concerns: {insights['additional_concerns']}")

        except (json.JSONDecodeError, KeyError) as e:
            # VLM response wasn't valid JSON - that's okay, use as-is
            print(f"    VLM response not in expected format, using rule-based results")

        return assessments

    def _calculate_overall_risk(self, assessments: List[RiskAssessment]) -> RiskLevel:
        """Calculate overall risk level based on all assessments"""
        if not assessments:
            return RiskLevel.LOW

        # Overall risk is the highest individual risk
        priorities = {
            RiskLevel.CRITICAL: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.MEDIUM: 3,
            RiskLevel.LOW: 4
        }

        highest_risk = RiskLevel.LOW
        highest_priority = 4

        for assessment in assessments:
            priority = priorities.get(assessment.severity, 4)
            if priority < highest_priority:
                highest_priority = priority
                highest_risk = assessment.severity

        return highest_risk

    def _sensitivity_to_risk_level(self, sensitivity: str) -> RiskLevel:
        """Convert sensitivity setting to base risk level"""
        mapping = {
            "critical": RiskLevel.CRITICAL,
            "high": RiskLevel.HIGH,
            "medium": RiskLevel.MEDIUM,
            "low": RiskLevel.LOW
        }
        return mapping.get(sensitivity.lower(), RiskLevel.MEDIUM)

    def _adjust_risk_level(self, base_level: RiskLevel, sensitivity: str) -> RiskLevel:
        """Adjust risk level based on user sensitivity"""
        # If sensitivity is higher than base level, escalate
        sensitivity_level = self._sensitivity_to_risk_level(sensitivity)

        # Use the higher of base level or sensitivity level
        if self._risk_level_priority(sensitivity_level) < self._risk_level_priority(base_level):
            return sensitivity_level
        return base_level

    def _risk_level_priority(self, level: RiskLevel) -> int:
        """Get priority value for risk level (lower = more severe)"""
        priorities = {
            RiskLevel.CRITICAL: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.MEDIUM: 3,
            RiskLevel.LOW: 4
        }
        return priorities.get(level, 4)

    def _get_text_sensitivity_key(self, text_type: str) -> str:
        """Get the appropriate sensitivity key for text type"""
        mapping = {
            "ssn": "personal_numbers",
            "credit_card": "financial_data",
            "password": "documents",
            "bank_account": "financial_data",
            "phone_number": "personal_numbers",
            "email": "personal_numbers",
            "address": "home_address",
            "general_text": "documents"
        }
        return mapping.get(text_type, "documents")

    def _get_object_sensitivity_key(self, obj_class: str, risk_type: RiskType) -> str:
        """Get the appropriate sensitivity key for object class"""
        obj_lower = obj_class.lower()

        if obj_lower in ["laptop", "tv", "cell phone", "monitor"]:
            return "screens"
        elif obj_lower in ["car", "truck", "bus", "motorcycle"]:
            return "license_plates"
        elif obj_lower in ["backpack", "handbag", "suitcase"]:
            return "background_items"
        else:
            return "background_items"

    def _get_sensitivity_dict(self, profile: PrivacyProfile, risk_type: RiskType) -> Dict[str, str]:
        """Get the appropriate sensitivity dictionary based on risk type"""
        if risk_type == RiskType.IDENTITY_EXPOSURE:
            return profile.identity_sensitivity
        elif risk_type == RiskType.INFORMATION_DISCLOSURE:
            return profile.information_sensitivity
        elif risk_type == RiskType.LOCATION_EXPOSURE:
            return profile.location_sensitivity
        else:
            return profile.context_sensitivity

    def _print_summary(self, result: RiskAnalysisResult):
        """Print summary of risk assessment"""
        print(f"\n{'='*60}")
        print(f"Risk Assessment Complete")
        print(f"{'='*60}")
        print(f"  Overall Risk Level: {result.overall_risk_level.value.upper()}")
        print(f"  Total Assessments: {len(result.risk_assessments)}")
        print(f"  Confirmed Risks (High/Critical): {result.confirmed_risks}")
        print(f"  Faces Pending Identity: {result.faces_pending_identity}")
        print(f"  Processing Time: {result.processimg_time_ms:.2f}ms")

        # Breakdown by severity
        for level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]:
            count = len(result.get_by_serverity(level))
            if count > 0:
                print(f"    - {level.value.capitalize()}: {count}")

        print(f"{'='*60}\n")
