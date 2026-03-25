"""
profile_schemas.py — Request / response Pydantic models for the Privacy Profile API.

These schemas are intentionally separate from the core utils/models.py PrivacyProfile
so that the API surface can evolve independently of the internal data representation.
All fields are optional in update payloads (PATCH semantics via PUT endpoint).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
# Nested settings — mirrored from utils.models but kept explicit for OpenAPI

class FaceSensitivityRequest(BaseModel):
    bystander_sensitivity: Optional[str] = None   # critical | high | medium | low
    known_contact_sensitivity: Optional[str] = None
    self_sensitivity: Optional[str] = None
    min_face_size_px: Optional[int] = Field(default=None, ge=10, le=100)

    @field_validator("bystander_sensitivity", "known_contact_sensitivity", "self_sensitivity", mode="before")
    @classmethod
    def validate_risk_level(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"critical", "high", "medium", "low"}
        if v.lower() not in allowed:
            raise ValueError(f"Must be one of {sorted(allowed)}, got '{v}'")
        return v.lower()


class TextSensitivityRequest(BaseModel):
    # protect_ssn, protect_credit_card, protect_passwords are BLOCK-guarded —
    # the service layer enforces they cannot be set to False.
    protect_ssn: Optional[bool] = None
    protect_credit_card: Optional[bool] = None
    protect_passwords: Optional[bool] = None
    protect_phone_numbers: Optional[bool] = None
    protect_email_addresses: Optional[bool] = None
    protect_addresses: Optional[bool] = None
    protect_names: Optional[bool] = None
    protect_numeric_fragments: Optional[bool] = None
    protect_generic_text: Optional[bool] = None


class ScreenSensitivityRequest(BaseModel):
    protect_screens_when_on: Optional[bool] = None
    protect_screens_when_off: Optional[bool] = None
    own_devices_unprotected: Optional[bool] = None


class ObjectSensitivityRequest(BaseModel):
    protect_license_plates: Optional[bool] = None
    protect_personal_documents: Optional[bool] = None
    protect_other_objects: Optional[bool] = None


class ContactEntryRequest(BaseModel):
    person_id: str
    display_name: str = Field(..., min_length=1, max_length=100)
    relationship: str = "friend"
    consent_level: str = "assumed"

    @field_validator("relationship", mode="before")
    @classmethod
    def validate_relationship(cls, v: str) -> str:
        allowed = {"self", "family", "friend", "colleague", "other"}
        if v.lower() not in allowed:
            raise ValueError(f"relationship must be one of {sorted(allowed)}")
        return v.lower()

    @field_validator("consent_level", mode="before")
    @classmethod
    def validate_consent_level(cls, v: str) -> str:
        allowed = {"explicit", "assumed", "none"}
        if v.lower() not in allowed:
            raise ValueError(f"consent_level must be one of {sorted(allowed)}")
        return v.lower()
# Create request

class CreateProfileRequest(BaseModel):
    """Payload for POST /api/v1/profile (onboarding completion)."""

    display_name: str = Field(..., min_length=1, max_length=50)

    # Sensitivity settings — all have safe defaults if omitted
    face_settings: Optional[FaceSensitivityRequest] = None
    text_settings: Optional[TextSensitivityRequest] = None
    screen_settings: Optional[ScreenSensitivityRequest] = None
    object_settings: Optional[ObjectSensitivityRequest] = None

    # Method preferences
    preferred_face_method: Optional[str] = None
    preferred_text_method: Optional[str] = None
    preferred_screen_method: Optional[str] = None
    preferred_object_method: Optional[str] = None

    # Processing mode
    default_mode: Optional[str] = None       # manual | hybrid | auto
    ethical_mode: Optional[str] = None       # strict | balanced | creative

    # HITL prefs
    auto_advance_threshold: Optional[str] = None   # never | low | medium | high
    pause_on_critical: Optional[bool] = None
    pause_on_new_faces: Optional[bool] = None
    require_confirm_on_bystander_unprotect: Optional[bool] = None

    # Custom threshold overrides
    threshold_overrides: Optional[Dict[str, str]] = None

    # Mark onboarding done on creation
    onboarding_complete: bool = False

    @field_validator("default_mode", mode="before")
    @classmethod
    def validate_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in {"manual", "hybrid", "auto"}:
            raise ValueError("default_mode must be 'manual', 'hybrid', or 'auto'")
        return v

    @field_validator("ethical_mode", mode="before")
    @classmethod
    def validate_ethical_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in {"strict", "balanced", "creative"}:
            raise ValueError("ethical_mode must be 'strict', 'balanced', or 'creative'")
        return v

    @field_validator("auto_advance_threshold", mode="before")
    @classmethod
    def validate_threshold(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in {"never", "low", "medium", "high"}:
            raise ValueError("auto_advance_threshold must be 'never', 'low', 'medium', or 'high'")
        return v

    @field_validator("preferred_face_method", "preferred_text_method",
                     "preferred_screen_method", "preferred_object_method", mode="before")
    @classmethod
    def validate_method(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"blur", "pixelate", "solid_overlay", "inpaint", "avatar_replace", "none"}
        if v not in allowed:
            raise ValueError(f"method must be one of {sorted(allowed)}")
        return v
# Update request (all fields optional — partial update semantics)

class UpdateProfileRequest(BaseModel):
    """Payload for PUT /api/v1/profile (partial update)."""

    display_name: Optional[str] = Field(default=None, min_length=1, max_length=50)

    face_settings: Optional[FaceSensitivityRequest] = None
    text_settings: Optional[TextSensitivityRequest] = None
    screen_settings: Optional[ScreenSensitivityRequest] = None
    object_settings: Optional[ObjectSensitivityRequest] = None

    preferred_face_method: Optional[str] = None
    preferred_text_method: Optional[str] = None
    preferred_screen_method: Optional[str] = None
    preferred_object_method: Optional[str] = None

    default_mode: Optional[str] = None
    ethical_mode: Optional[str] = None

    auto_advance_threshold: Optional[str] = None
    pause_on_critical: Optional[bool] = None
    pause_on_new_faces: Optional[bool] = None
    require_confirm_on_bystander_unprotect: Optional[bool] = None

    threshold_overrides: Optional[Dict[str, str]] = None
    onboarding_complete: Optional[bool] = None

    @field_validator("default_mode", mode="before")
    @classmethod
    def validate_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in {"manual", "hybrid", "auto"}:
            raise ValueError("default_mode must be 'manual', 'hybrid', or 'auto'")
        return v

    @field_validator("ethical_mode", mode="before")
    @classmethod
    def validate_ethical_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in {"strict", "balanced", "creative"}:
            raise ValueError("ethical_mode must be 'strict', 'balanced', or 'creative'")
        return v

    @field_validator("auto_advance_threshold", mode="before")
    @classmethod
    def validate_threshold(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in {"never", "low", "medium", "high"}:
            raise ValueError("auto_advance_threshold must be 'never', 'low', 'medium', or 'high'")
        return v

    @field_validator("preferred_face_method", "preferred_text_method",
                     "preferred_screen_method", "preferred_object_method", mode="before")
    @classmethod
    def validate_method(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"blur", "pixelate", "solid_overlay", "inpaint", "avatar_replace", "none"}
        if v not in allowed:
            raise ValueError(f"method must be one of {sorted(allowed)}")
        return v
# Response models

class ContactEntryResponse(BaseModel):
    person_id: str
    display_name: str
    relationship: str
    consent_level: str


class FaceSensitivityResponse(BaseModel):
    bystander_sensitivity: str
    known_contact_sensitivity: str
    self_sensitivity: str
    min_face_size_px: int


class TextSensitivityResponse(BaseModel):
    protect_ssn: bool
    protect_credit_card: bool
    protect_passwords: bool
    protect_phone_numbers: bool
    protect_email_addresses: bool
    protect_addresses: bool
    protect_names: bool
    protect_numeric_fragments: bool
    protect_generic_text: bool


class ScreenSensitivityResponse(BaseModel):
    protect_screens_when_on: bool
    protect_screens_when_off: bool
    own_devices_unprotected: bool


class ObjectSensitivityResponse(BaseModel):
    protect_license_plates: bool
    protect_personal_documents: bool
    protect_other_objects: bool


class ProfileResponse(BaseModel):
    """Full profile as returned by GET /api/v1/profile."""

    user_id: str
    profile_version: int
    onboarding_complete: bool
    display_name: Optional[str]
    self_person_id: Optional[str]
    face_enrollment_count: int
    known_contacts: List[ContactEntryResponse]

    face_settings: FaceSensitivityResponse
    text_settings: TextSensitivityResponse
    screen_settings: ScreenSensitivityResponse
    object_settings: ObjectSensitivityResponse

    preferred_face_method: str
    preferred_text_method: str
    preferred_screen_method: str
    preferred_object_method: str

    default_mode: str
    ethical_mode: str

    auto_advance_threshold: str
    pause_on_critical: bool
    pause_on_new_faces: bool
    require_confirm_on_bystander_unprotect: bool

    threshold_overrides: Dict[str, str]

    created_at: str    # ISO-8601
    updated_at: str    # ISO-8601


class ProfileCreatedResponse(BaseModel):
    """Slim acknowledgement returned on POST /api/v1/profile."""

    user_id: str
    onboarding_complete: bool
    message: str = "Profile created successfully."


class ProfileUpdatedResponse(BaseModel):
    """Acknowledgement returned on PUT /api/v1/profile."""

    user_id: str
    updated_fields: List[str]
    message: str = "Profile updated successfully."


class ProfileDeletedResponse(BaseModel):
    """Acknowledgement returned on DELETE /api/v1/profile."""

    user_id: str
    message: str = "Profile deleted successfully."
# Questionnaire response (static structure served from YAML)

class QuestionnaireField(BaseModel):
    id: str
    type: str
    label: Optional[str] = None
    placeholder: Optional[str] = None
    required: bool = False
    accept: Optional[str] = None
    min: Optional[int] = None
    max: Optional[int] = None
    options: Optional[List[Any]] = None
    default: Optional[Any] = None


class QuestionnaireStep(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    required: bool = False
    repeatable: bool = False
    collapsed: bool = False
    type: Optional[str] = None        # "summary" for Step 5
    fields: List[QuestionnaireField] = Field(default_factory=list)


class QuestionnaireResponse(BaseModel):
    steps: List[QuestionnaireStep]
