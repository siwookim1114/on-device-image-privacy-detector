"""
Data Models 
All data structures used throughout the system
Uses Pydantic for validation and serialization
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import time
import uuid

# Enumerations 

class RiskLevel(str, Enum):
    """Risk Severity Levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RiskType(str, Enum):
    """Types of Privacy Risks"""
    IDENTITY_EXPOSURE = "identity_exposure"
    INFORMATION_DISCLOSURE = "information_disclosure"
    LOCATION_EXPOSURE = "location_exposure"
    CONTEXT_EXPOSURE = "context_exposure"

class ProcessingMode(str, Enum):
    """User processing modes"""
    MANUAL = "manual"
    HYBRID = "hybrid"
    AUTO = "auto"

class EthicalMode(str, Enum):
    """Ethical editing modes"""
    STRICT = "strict"
    BALANCED = "balanced"
    CREATIVE = "creative"

class ObfuscationMethod(str, Enum):
    """Available obfuscation methods"""
    BLUR = "blur"
    PIXELATE = "pixelate"
    SOLID_OVERLAY = "solid_overlay"
    SILHOUETTE = "silhouette"
    INPAINT = "inpaint"
    AVATAR_REPLACE = "avatar_replace"
    GENERATIVE_REPLACE = "generative_replace"
    NONE = "none"

class PersonClassification(str, Enum):
    """Person identity classifications"""
    PRIMARY_SUBJECT = "primary_subject"            # User themselves
    KNOWN_CONTACT = "known_contact"                # Recognized friend/family
    BYSTANDER = "bystander"                        # Unknown person

class ConsentStatus(str, Enum):
    """Consent status for people in images"""
    EXPLICIT = "explicit"                          # User themselves
    ASSUMED = "assumed"                            # Known contact with approval history
    NONE = "none"                                  # Unknown/bystander
    UNCLEAR = "unclear"                            # Requires confirmation


# Privacy Profile Sub-Models

class FaceSensitivitySettings(BaseModel):
    """Per-identity-class sensitivity thresholds for face detection."""
    bystander_sensitivity: str = "critical"      # RiskLevel value
    known_contact_sensitivity: str = "high"
    self_sensitivity: str = "medium"
    min_face_size_px: int = Field(default=30, ge=10, le=100)


class TextSensitivitySettings(BaseModel):
    """Boolean flags controlling which text categories trigger protection."""
    protect_ssn: bool = True                     # BLOCK-guarded — cannot be False
    protect_credit_card: bool = True             # BLOCK-guarded
    protect_passwords: bool = True               # BLOCK-guarded
    protect_phone_numbers: bool = True
    protect_email_addresses: bool = True
    protect_addresses: bool = False
    protect_names: bool = False
    protect_numeric_fragments: bool = True
    protect_generic_text: bool = False


class ScreenSensitivitySettings(BaseModel):
    """Controls when screen devices are protected."""
    protect_screens_when_on: bool = True
    protect_screens_when_off: bool = False
    own_devices_unprotected: bool = True


class ObjectSensitivitySettings(BaseModel):
    """Controls which object categories receive protection."""
    protect_license_plates: bool = True
    protect_personal_documents: bool = True
    protect_other_objects: bool = False


class ContactEntry(BaseModel):
    """A single known contact entry in the user's contact list."""
    person_id: str
    display_name: str
    relationship: str = "friend"                 # family, friend, colleague, other
    consent_level: str = "assumed"              # explicit, assumed, none


# Privacy Profile Models
class PrivacyProfile(BaseModel):
    """User's privacy preferences and sensitivity settings"""
    user_id: str = Field(default_factory = lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory = datetime.now)
    updated_at: datetime = Field(default_factory = datetime.now)

    # Legacy sensitivity dicts — preserved for backward compatibility.
    # New code reads from the structured sub-models below; these fields are
    # accepted if present in existing MongoDB documents and ignored.
    identity_sensitivity: Optional[Dict[str, str]] = Field(
        default_factory = lambda: {
            "own_face": "medium",
            "family_faces": "high",
            "friend_faces": "medium",
            "bystander_faces": "critical"
        }
    )

    information_sensitivity: Optional[Dict[str, str]] = Field(
        default_factory = lambda: {
            "personal_numbers": "critical",
            "documents": "high",
            "screens": "high",
            "financial_data": "critical"
        }
    )

    location_sensitivity: Optional[Dict[str, str]] = Field(
        default_factory = lambda: {
            "home_address": "critical",
            "license_plates": "high",
            "workplace": "medium",
            "landmarks": "low"
        }
    )

    context_sensitivity: Optional[Dict[str, str]] = Field(
        default_factory = lambda: {
            "background_items": "medium",
            "reflections": "medium",
            "metadata": "high"
        }
    )

    # Default preferences (preserved)
    default_mode: str = "hybrid"     # manual, hybrid, auto
    ethical_mode: str = "balanced"   # strict, balanced, creative
    display_name: Optional[str] = None
    self_person_id: Optional[str] = None
    face_enrollment_count: int = 0
    known_contacts: List[ContactEntry] = Field(default_factory=list)
    face_settings: FaceSensitivitySettings = Field(
        default_factory=FaceSensitivitySettings
    )
    text_settings: TextSensitivitySettings = Field(
        default_factory=TextSensitivitySettings
    )
    screen_settings: ScreenSensitivitySettings = Field(
        default_factory=ScreenSensitivitySettings
    )
    object_settings: ObjectSensitivitySettings = Field(
        default_factory=ObjectSensitivitySettings
    )
    preferred_face_method: str = "blur"
    preferred_text_method: str = "solid_overlay"
    preferred_screen_method: str = "blur"
    preferred_object_method: str = "blur"
    auto_advance_threshold: str = "medium"   # never, low, medium, high
    pause_on_critical: bool = True
    pause_on_new_faces: bool = True
    require_confirm_on_bystander_unprotect: bool = True
    # Keys: "face_bystander", "face_known", "face_self", "text_ssn", etc.
    # Values: RiskLevel string (critical / high / medium / low)
    threshold_overrides: Dict[str, str] = Field(default_factory=dict)
    onboarding_complete: bool = False
    profile_version: int = 1

    class Config:
        use_enum_values = True

# Detection Models
class BoundingBox(BaseModel):
    """Bounding box cordinates"""
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_raw(cls, data) -> "BoundingBox":
        """Parse a BoundingBox from dict, list, or existing BoundingBox.

        Accepts:
          - dict with keys x, y, width, height
          - list/tuple of [x, y, width, height]
          - an existing BoundingBox instance (returned as-is)
          - anything else returns a zero-area bbox
        """
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return cls(**data)
        if isinstance(data, (list, tuple)) and len(data) >= 4:
            return cls(x=data[0], y=data[1], width=data[2], height=data[3])
        return cls(x=0, y=0, width=0, height=0)

    def to_list(self) -> List[int]:
        """Convert to [x, y, w, h] list"""
        return [self.x, self.y, self.width, self.height]
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Conver to (x1, y1, x2, y2) format"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
class Detection(BaseModel):
    """Base Detection Model"""
    id: str = Field(default_factory = lambda: str(uuid.uuid4()))
    category: str
    bbox: BoundingBox
    confidence: float = Field(ge = 0.0, le= 1.0)      # Confidence always 0 <= x <= 1
    attributes: Dict[str, Any] = Field(default_factory = dict)

class FaceDetection(Detection):
    """Face detection result"""
    # id: str = Field(default_factory = lambda: str(uuid.uuid4()))
    # category: str
    # bbox: BoundingBox
    # confidence: float = Field(ge = 0.0, le= 1.0)      # Confidence always 0 <= x <= 1
    # attributes: Dict[str, Any] = Field(default_factory = dict)
    category: str = "face"
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None
    angle: Optional[str] = None    # frontal, side_profile, etc
    size: Optional[str] = None     # large, medium, small
    clarity: Optional[str] = None   # high, medium, low

class TextDetection(Detection):
    """Text/OCR detection result"""
    # id: str = Field(default_factory = lambda: str(uuid.uuid4()))
    # category: str
    # bbox: BoundingBox
    # confidence: float = Field(ge = 0.0, le= 1.0)      # Confidence always 0 <= x <= 1
    # attributes: Dict[str, Any] = Field(default_factory = dict)
    category: str = "text"
    text_content: str
    text_type: Optional[str] = None   # phone_number, name, etc
    language: Optional[str] = "en"
    polygon: Optional[List[List[int]]] = None  # EasyOCR 4-point quadrilateral

class ObjectDetection(Detection):
    """Object detection result"""
    # id: str = Field(default_factory = lambda: str(uuid.uuid4()))
    # category: str
    # bbox: BoundingBox
    # confidence: float = Field(ge = 0.0, le= 1.0)      # Confidence always 0 <= x <= 1
    # attributes: Dict[str, Any] = Field(default_factory = dict)
    category: str = "object"
    object_class: str
    contains_text: bool = False
    contains_screen: bool = False

class DetectionResults(BaseModel):
    """Complete detection results"""
    image_path: str
    annotated_image_path: Optional[str] = None  # Path to visualized image with bounding boxes
    faces: List[FaceDetection] = Field(default_factory = list)
    text_regions: List[TextDetection] = Field(default_factory = list)
    objects: List[ObjectDetection] = Field(default_factory = list)
    scene_context: Dict[str, Any] = Field(default_factory = dict)
    processing_time_ms: float = 0.0

    @property
    def total_detections(self) -> int:
        """Calculate total detections"""
        return len(self.faces) + len(self.text_regions) + len(self.objects)

# Risk Assessment Models
class RiskAssessment(BaseModel):
    """Risk assessment for a detected element"""
    detection_id: str
    element_type: str    # face, text, object
    element_description: str
    risk_type: RiskType
    severity: RiskLevel
    color_code: str
    reasoning: str
    user_sensitivity_applied: str
    bbox: BoundingBox
    requires_protection: bool = True
    legal_requirement: bool = False

    # Screen-only bbox for screen devices (top portion of device bbox, excludes keyboard/trackpad)
    screen_bbox: Optional[BoundingBox] = None

    # Phase 1.5a screen state: "verified_on", "verified_off", or None (not a screen device)
    screen_state: Optional[str] = None

    # Consent-related (for faces)
    person_id: Optional[str] = None
    person_label: Optional[str] = None
    classification: Optional[PersonClassification] = None
    consent_status: Optional[ConsentStatus] = None
    consent_confidence: float = 0.0

    # Text-specific: EasyOCR 4-point polygon for tighter overlay rendering
    text_polygon: Optional[List[List[int]]] = None

class RiskAnalysisResult(BaseModel):
    """Complete risk analysis results"""
    image_path: str
    risk_assessments: List[RiskAssessment] = Field(default_factory = list)
    overall_risk_level: RiskLevel
    faces_pending_identity: int = 0
    confirmed_risks: int = 0
    processing_time_ms: float = 0.0

    def get_by_severity(self, severity: RiskLevel) -> List[RiskAssessment]:
        """Get risks by severity level"""
        return [r for r in self.risk_assessments if r.severity == severity]

    def get_critical_risks(self) -> List[RiskAssessment]:
        """Get all critical risks""" 
        return self.get_by_severity(RiskLevel.CRITICAL)
    
    def get_high_risks(self) -> List[RiskAssessment]:
        """Get all high risks"""
        return self.get_by_severity(RiskLevel.HIGH)

# Face Recognition & Consent Models
class FaceEmbedding(BaseModel):
    """Face embedding vector"""
    embedding: List[float] = Field(min_length = 512, max_length = 512)
    source_image: Optional[str] = None
    timestamp: datetime = Field(default_factory = datetime.now)

class ConsentHistory(BaseModel):
    """Consent history for a person"""
    times_appeared: int = 0
    times_approved: int = 0
    times_protected: int = 0
    contexts: List[str] = Field(default_factory = list)
    last_consent_decision: Optional[str] = None
    consent_confidence: float = 0.0

    @property
    def approval_rate(self) -> float:
        """Calculate approval rate"""
        if self.times_appeared == 0:
            return 0.0
        return self.times_approved / self.times_appeared

    @property
    def protection_rate(self) -> float:
        """Calculate protection rate""" 
        if self.times_appeared == 0:
            return 0.0
        return self.times_protected / self.times_appeared

class PersonEntry(BaseModel):
    """Entry in face database"""
    person_id: str = Field(default_factory = lambda: str(uuid.uuid4()))
    label: str     # User-assigned label
    relationship: str   # self, family, friend, colleague, bystander
    embeddings: List[FaceEmbedding] = Field(default_factory = list, max_length = 5)
    consent_history: ConsentHistory = Field(default_factory=ConsentHistory)
    risk_decay_factor: float = Field(default = 1.0, ge = 0.0, le = 2.0)
    first_seen: datetime = Field(default_factory = datetime.now)
    last_seen: datetime = Field(default_factory = datetime.now)
    notes: Optional[str] = None

class IdentityMatch(BaseModel):
    """Face matching result"""
    detection_id: str
    person_id: Optional[str] = None
    person_label: Optional[str] = None
    classification: PersonClassification
    consent_status: ConsentStatus
    consent_confidence: float
    recognition_confidence: float
    match_type: str         # confident, probable, no_match
    history: Optional[ConsentHistory] = None
    ethical_risk: str       # None, Low, Medium, High
    risk_adjustment: str    # severity_reduced_to_X, severity_escalated_to_X

# Strategy & Obfuscation Models
class ObfuscationParameters(BaseModel):
    """Parameters for obfuscation methods"""
    method: ObfuscationMethod
    parameters: Dict[str, Any] = Field(default_factory = dict)

class AlternativeMethod(BaseModel):
    """Alternative obfuscation method"""
    method: ObfuscationMethod
    parameters: Dict[str, Any]
    reasoning: str
    score: int = Field(ge = 0, le = 10)

class ProtectionStrategy(BaseModel):
    """Protection strategy for a detected risk"""
    detection_id: str
    element: str
    severity: RiskLevel
    recommended_action: str      # Protect, None, None with Confirmation
    recommended_method: Optional[ObfuscationMethod] = None
    parameters: Dict[str, Any] = Field(default_factory = dict)
    reasoning: str
    alternative_options: List[AlternativeMethod] = Field(default_factory = list)
    ethical_compliance: str = "COMPLIANT"
    execution_priority: int = Field(ge = 1, le = 5)
    optional: bool = False
    requires_user_decision: bool = False
    user_can_override: bool = True
    # SAM segmentation mask path (set by PrecisionSegmenter after Agent 3)
    segmentation_mask_path: Optional[str] = None

class StrategyRecommendations(BaseModel):
    """Complete strategy recommendations"""
    image_path: str
    strategies: List[ProtectionStrategy] = Field(default_factory = list)
    total_protections_recommended: int = 0
    requires_user_confirmation: int = 0
    estimated_processing_time_ms: float = 0.0

    def get_by_priority(self, priority: int) -> List[ProtectionStrategy]:
        """Get strategies by execution priority"""
        return [p for p in self.strategies if p.execution_priority == priority]

# Execution Models
class TransformationResult(BaseModel):
    """Result of a transformation"""
    detection_id: str
    element: str
    method: ObfuscationMethod
    parameters: Dict[str, Any]
    status: str    # success, failed, skipped
    execution_time_ms: float
    error_message: Optional[str] = None

class ExecutionReport(BaseModel):
    """Report of execution agent"""
    image_path: str
    status: str      # completed, partial, failed
    transformations_applied: List[TransformationResult] = Field(default_factory = list)
    elements_unchanged: List[Dict[str, str]] = Field(default_factory = list)
    total_execution_time_ms: float = 0.0
    protected_image_path: Optional[str] = None

# Provenance Models

class ProvenanceEventType(str, Enum):
    PIPELINE_START = "pipeline_start"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_FAILED = "pipeline_failed"
    STAGE_START = "stage_start"
    STAGE_COMPLETE = "stage_complete"
    STAGE_ERROR = "stage_error"
    FACE_DETECTED = "face_detected"
    TEXT_DETECTED = "text_detected"
    OBJECT_DETECTED = "object_detected"
    RISK_ASSESSED_P1 = "risk_assessed_phase1"
    RISK_ASSESSED_P2 = "risk_assessed_phase2"
    RISK_ESCALATED = "risk_escalated"
    SCREEN_STATE_VERIFIED = "screen_state_verified"
    FACE_MATCH_HIT = "face_match_hit"
    FACE_MATCH_MISS = "face_match_miss"
    CONSENT_APPLIED = "consent_applied"
    STRATEGY_ASSIGNED_P1 = "strategy_assigned_phase1"
    STRATEGY_ASSIGNED_P2 = "strategy_assigned_phase2"
    CHALLENGE_TRIGGERED = "challenge_triggered"
    CHALLENGE_RESOLVED = "challenge_resolved"
    SAM_MASK_GENERATED = "sam_mask_generated"
    SAM_SKIPPED = "sam_skipped"
    OBFUSCATION_APPLIED = "obfuscation_applied"
    OBFUSCATION_FAILED = "obfuscation_failed"
    VLM_PATCH_APPLIED = "vlm_patch_applied"
    VLM_COVERAGE_ADDED = "vlm_coverage_added"
    SAFETY_ALLOW = "safety_allow"
    SAFETY_BLOCK = "safety_block"
    SAFETY_CHALLENGE = "safety_challenge"
    HITL_PAUSED = "hitl_paused"
    HITL_APPROVED = "hitl_approved"
    HITL_OVERRIDE_APPLIED = "hitl_override_applied"
    HITL_REJECTED = "hitl_rejected"


class ProvenanceEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: float = Field(default_factory=time.time)
    event_type: ProvenanceEventType
    phase: str
    detection_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class EditRecord(BaseModel):
    """Record of a single edit"""
    edit_id: int
    element_type: str
    content_type: Optional[str] = None
    region: Dict[str, Any]  # bbox as dict
    technique: ObfuscationMethod
    parameters: Dict[str, Any]
    reason: str
    consent_status: Optional[str] = None
    legal_requirement: bool = False
    synthetic_content_added: bool = False
    synthetic_type: Optional[str] = None

class ProvenanceLog(BaseModel):
    """Complete provenance log"""
    image_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: str
    original_hash: str
    protected_hash: str
    privacy_profile_version: str = "v1.0"
    ethical_mode: EthicalMode
    edits_applied: List[EditRecord] = Field(default_factory=list)
    people_recognized: List[Dict[str, Any]] = Field(default_factory=list)
    authenticity_score: float = Field(ge=0.0, le=1.0)
    deception_risk: str
    synthetic_content_count: int = 0
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)

# Pipeline Models
class PipelineInput(BaseModel):
    """Input to the privacy protection pipeline"""
    image_path: str
    privacy_profile: PrivacyProfile
    mode: ProcessingMode = ProcessingMode.HYBRID
    user_id: Optional[str] = None

class PipelineOutput(BaseModel):
    """Output from the privacy protection pipeline"""
    success: bool
    protected_image_path: Optional[str] = None
    provenance_log: Optional[ProvenanceLog] = None
    risk_analysis: Optional[RiskAnalysisResult] = None
    strategy_recommendations: Optional[StrategyRecommendations] = None
    execution_report: Optional[ExecutionReport] = None
    total_time_ms: float = 0.0
    error_message: Optional[str] = None
    phase_timings: Dict[str, float] = Field(default_factory=dict)

# Input Schemas for Phase 2 tools
class ReclassifyAssessmentInput(BaseModel):
    """Input schema for reclassify_assessment tool."""
    index: int = Field(description="Assessment index (0-based) from the assessment list")
    severity: str = Field(description="New severity level: critical, high, medium, or low")
    reason: str = Field(default="VLM visual review", description="Why this change is needed")


class ReclassifyItem(BaseModel):
    """Single reclassification within a batch."""
    index: int = Field(description="Assessment index (0-based)")
    severity: str = Field(description="New severity: critical, high, medium, or low")
    reason: str = Field(default="VLM visual review", description="Why this change is needed")


class BatchReclassifyInput(BaseModel):
    """Input schema for batch_reclassify tool."""
    reclassifications: List[ReclassifyItem] = Field(
        description="List of reclassifications to apply. Each has index, severity, and reason."
    )


class SplitPart(BaseModel):
    """Schema for a single part when splitting an assessment."""
    element_description: str = Field(description="Description of this part")
    severity: str = Field(description="Severity: critical, high, medium, or low")
    requires_protection: bool = Field(default=None, description="Whether this part requires protection (auto-set from severity if omitted)")
    visual_reason: str = Field(default="", description="Specific reason for this classification, e.g. 'Contains SSN digits' or 'Label text only, no actual data'")

    def model_post_init(self, __context):
        """Auto-derive requires_protection from severity if not explicitly set."""
        if self.requires_protection is None:
            self.requires_protection = self.severity.lower() in ("critical", "high")


class SplitAssessmentInput(BaseModel):
    """Input schema for split_assessment tool."""
    index: int = Field(description="Assessment index (0-based) to split")
    parts: List[SplitPart] = Field(description="List of parts to split into (minimum 2)")


# Agent 3 (Strategy Agent) tool input schemas

class ModifyStrategyInput(BaseModel):
    """Input schema for modify_strategy tool."""
    index: int = Field(description="Strategy index (0-based)")
    method: str = Field(description="New method: blur, pixelate, solid_overlay, inpaint, avatar_replace, none")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Method parameters (e.g. kernel_size, block_size, color)")
    reasoning: str = Field(default="VLM strategy review", description="Why this change improves the strategy")


class ModifyStrategyItem(BaseModel):
    """Single modification within a batch."""
    index: int = Field(description="Strategy index (0-based)")
    method: str = Field(description="New method: blur, pixelate, solid_overlay, inpaint, avatar_replace, none")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Method parameters")
    reasoning: str = Field(default="VLM strategy review", description="Why this change improves the strategy")


class BatchModifyStrategiesInput(BaseModel):
    """Input schema for batch_modify_strategies tool."""
    modifications: List[ModifyStrategyItem] = Field(
        description="List of strategy modifications to apply"
    )

# Agent 4 Phase 2: Verification tool input schemas

class PatchRegionInput(BaseModel):
    """Input schema for patch_region tool."""
    detection_id: str = Field(description="Detection ID of the element to re-protect")
    method: str = Field(description="Strengthened method: blur, pixelate, or solid_overlay")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Method parameters (e.g. kernel_size, block_size)")
    expand_px: int = Field(default=0, description="Pixels to expand bbox in all directions (0-30)")
    reasoning: str = Field(default="VLM verification", description="What leaked content was found")

class AddProtectionInput(BaseModel):
    """Input schema for add_protection tool."""
    x: int = Field(description="X coordinate of the region to protect")
    y: int = Field(description="Y coordinate of the region to protect")
    width: int = Field(description="Width of the region")
    height: int = Field(description="Height of the region")
    method: str = Field(description="Method: blur, pixelate, or solid_overlay")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Method parameters")
    reasoning: str = Field(default="VLM verification", description="What leaked content was found")
