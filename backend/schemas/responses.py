from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Error envelope — used by every error response
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Session / Auth
# ---------------------------------------------------------------------------

class SessionResponse(BaseModel):
    session_id: str
    token: str
    expires_at: str  # ISO-8601


# ---------------------------------------------------------------------------
# Pipeline status
# ---------------------------------------------------------------------------

class StageProgress(BaseModel):
    step: Optional[str] = None
    elements_processed: int = 0
    elements_total: int = 0
    elapsed_ms: Optional[int] = None


class HitlStatus(BaseModel):
    waiting: bool = False
    checkpoint: Optional[str] = None
    checkpoint_reason: Optional[str] = None
    elements_requiring_review: list[str] = Field(default_factory=list)
    actions_available: list[str] = Field(default_factory=list)


class PipelineTiming(BaseModel):
    queued_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_ms: Optional[int] = None
    stage_timings: dict[str, int] = Field(default_factory=dict)


class PipelineStatusResponse(BaseModel):
    session_id: str
    status: Literal[
        "queued",
        "running",
        "paused_hitl",
        "completed",
        "failed",
    ]
    current_stage: Optional[str] = None
    stage_progress: Optional[StageProgress] = None
    hitl: HitlStatus = Field(default_factory=HitlStatus)
    timing: PipelineTiming = Field(default_factory=PipelineTiming)
    error: Optional[ErrorDetail] = None


# ---------------------------------------------------------------------------
# Pipeline run (202 accepted)
# ---------------------------------------------------------------------------

class ImageMeta(BaseModel):
    filename: str
    width: int
    height: int
    size_bytes: int
    mime_type: str


class PipelineRunResponse(BaseModel):
    session_id: str
    status: Literal["queued"] = "queued"
    image_meta: ImageMeta


# ---------------------------------------------------------------------------
# Pipeline results
# ---------------------------------------------------------------------------

class DetectionResult(BaseModel):
    detection_id: str
    element_type: Literal["face", "text", "object"]
    bbox: list[int]  # [x, y, w, h]
    confidence: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RiskAssessmentResponse(BaseModel):
    detection_id: str
    severity: Literal["critical", "high", "medium", "low"]
    consent_status: Optional[Literal["explicit", "assumed", "none", "unclear"]] = None
    screen_state: Optional[Literal["verified_on", "verified_off"]] = None
    escalation_reasons: list[str] = Field(default_factory=list)


# Keep backward-compatible alias
RiskAssessment = RiskAssessmentResponse


class StrategyResponse(BaseModel):
    detection_id: str
    method: Literal[
        "blur",
        "pixelate",
        "solid_overlay",
        "inpaint",
        "avatar_replace",
        "generative_replace",
        "none",
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)


# Keep backward-compatible alias
Strategy = StrategyResponse


class ExecutionResult(BaseModel):
    detection_id: str
    applied: bool
    method_used: Optional[str] = None
    patch_applied: bool = False


class AuditEntry(BaseModel):
    timestamp: str
    stage: str
    action: str
    detection_id: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class PipelineResultsResponse(BaseModel):
    session_id: str
    status: str
    detections: list[DetectionResult] = Field(default_factory=list)
    risk_assessments: list[RiskAssessmentResponse] = Field(default_factory=list)
    strategies: list[StrategyResponse] = Field(default_factory=list)
    execution: list[ExecutionResult] = Field(default_factory=list)
    audit_trail: list[AuditEntry] = Field(default_factory=list)
    timing: PipelineTiming = Field(default_factory=PipelineTiming)


# ---------------------------------------------------------------------------
# Rerun
# ---------------------------------------------------------------------------

class RerunResponse(BaseModel):
    session_id: str
    status: Literal["queued"] = "queued"
    stages_to_rerun: list[str]
    stages_cached: list[str]


# ---------------------------------------------------------------------------
# HITL approve
# ---------------------------------------------------------------------------

class ApproveResponse(BaseModel):
    checkpoint: str
    next_stage: Optional[str]
    pipeline_resumed: bool


# ---------------------------------------------------------------------------
# Override
# ---------------------------------------------------------------------------

class AppliedOverride(BaseModel):
    detection_id: str
    type: str
    value: Any
    reason: str


class RejectedOverride(BaseModel):
    detection_id: str
    type: str
    error_code: str
    message: str


class OverrideResponse(BaseModel):
    applied: list[AppliedOverride] = Field(default_factory=list)
    rejected: list[RejectedOverride] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ParsedIntentResponse(BaseModel):
    action: str
    target_stage: Optional[str] = None
    target_elements: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    natural_language: str = ""


class ChatResponse(BaseModel):
    intent: ParsedIntentResponse
    response_text: str
    pipeline_action_taken: Optional[str] = None
    suggestions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class SessionSummary(BaseModel):
    session_id: str
    created_at: str
    status: str
    image_filename: Optional[str] = None
    protections_applied: int = 0


class HistoryResponse(BaseModel):
    items: list[SessionSummary]
    total: int
    page: int
    page_size: int
    has_next: bool
