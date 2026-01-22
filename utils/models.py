"""
Data Models 
All data structures used throughout the system
Uses Pydantic for validation and serialization
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
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


# Privacy Profile Models
class PrivacyProfile(BaseModel):
    """User's privacy preferences and sensitivity settings"""
    user_id: str = Field(default_factory = lambda: str(uuid.uuid4()))  
    created_at: datetime = Field(default_factory = datetime.now)
    updated_at: datetime = Field(default_factory = datetime.now)

    # Identity Sensitivity
    identity_sensitivity: Dict[str, str] = Field(
        default_factory = lambda: {
            "own_face": "medium",
            "family_faces": "high",
            "friend_faces": "medium",
            "bystander_faces": "critical"
        }
    )

    # Information Sensitivity
    information_sensitivity: Dict[str, str] = Field(
        default_factory = lambda: {
            "personal_numbers": "critical",
            "documents": "high",
            "screens": "high",
            "financial_data": "critical"
        }
    )

    # Location Sensitivity
    location_sensitivity: Dict[str, str] = Field(
        default_factory = lambda: {
            "home_address": "critical",
            "license_plates": "high",
            "workplace": "medium",
            "landmarks": "low"
        }
    )

    # Context Sensitivity
    context_sensitivity: Dict[str, str] = Field(
        default_factory = lambda: {
            "background_items": "medium",
            "reflections": "medium",
            "metadata": "high"
        }
    )

    # Default preferences
    default_mode: str = "hybrid"     # manual, hybrid, auto
    ethical_mode: str = "balanced"   # strict, balanced, creative

    class Config:
        use_enum_values = True

# Detection Models
class BoundingBox(BaseModel):
    """Bounding box cordinates"""
    x: int
    y: int
    width: int
    height: int

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
    category: str = "face"
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None
    angle: Optional[str] = None    # frontal, side_profile, etc
    size: Optional[str] = None     # large, medium, small
    clarity: Optional[str] = None   # high, medium, low

class TextDetection(Detection):
    """Text/OCR detection result"""
    category: str = "text"
    text_content: str
    text_type: Optional[str] = None   # phone_number, name, etc
    language: Optional[str] = "en"

class ObjectDetection(Detection):
    """Object detection result"""
    object_class: str
    contains_text: bool = False
    contains_screen: bool = False

class DetectionResults(BaseModel):
    """Complete detection results"""
    image_path: str
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

    # Consent-related (for faces)
    person_id: Optional[str] = None
    person_label: Optional[str] = None
    classification: Optional[PersonClassification] = None
    consent_status: Optional[ConsentStatus] = None
    consent_confidence: float = 0.0

class RiskAnalysisResult(BaseModel):
    """Complete risk analysis results"""
    image_path: str
    risk_assessments: List[RiskAssessment] = Field(default_factory = list)
    overall_risk_level: RiskLevel
    faces_pending_identity: int = 0
    confirmed_risks: int = 0
    processimg_time_ms: float = 0.0

    def get_by_serverity(self, severity: RiskLevel) -> List[RiskAssessment]:
        """Get risks by severity level"""
        return [r for r in self.risk_assessments if r.severity == severity]

    def get_critical_risks(self) -> List[RiskAssessment]:
        """Get all critical risks""" 
        return self.get_by_serverity(RiskLevel.CRITICAL)
    
    def get_high_risks(self) -> List[RiskAssessment]:
        """Get all high risks"""
        return self.get_by_serverity(RiskLevel.HIGH)

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
    error_message: Optional[float] = None

class ExecutionReport(BaseModel):
    """Report of execution agent"""
    image_path: str
    status: str      # completed, partial, failed
    tranformations_applied: List[TransformationResult] = Field(default_factory = list)
    elements_unchanged: List[Dict[str, str]] = Field(default_factory = list)
    total_execution_time_ms: float = 0.0
    protected_image_path: Optional[str] = None

# Provenance Models
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
    execution_report: Optional[ExecutionReport] = None
    total_time_ms: float = 0.0
    error_message: Optional[str] = None

