from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# RunConfig — sent as a JSON string in the multipart /pipeline/run upload
# ---------------------------------------------------------------------------

class PhaseFlags(BaseModel):
    """Which VLM phases should execute."""
    run_vlm_risk: bool = True
    run_vlm_strategy: bool = True
    run_vlm_execution: bool = True


class HitlConfig(BaseModel):
    """Human-in-the-loop settings."""
    pause_on_critical: bool = True
    auto_advance_threshold: Literal["low", "medium", "high", "never"] = "medium"


class RunConfig(BaseModel):
    """Top-level pipeline run configuration."""
    mode: Literal["auto", "hybrid", "manual"] = "auto"
    ethical_mode: Literal["strict", "balanced", "creative"] = "balanced"
    phases: PhaseFlags = Field(default_factory=PhaseFlags)
    hitl: HitlConfig = Field(default_factory=HitlConfig)


# ---------------------------------------------------------------------------
# RerunRequest
# ---------------------------------------------------------------------------

VALID_STAGES = frozenset({
    "detection",
    "risk",
    "consent",
    "strategy",
    "sam",
    "execution",
    "export",
})


class RerunRequest(BaseModel):
    from_stage: str
    reason: Optional[str] = None

    @field_validator("from_stage")
    @classmethod
    def validate_stage(cls, v: str) -> str:
        if v not in VALID_STAGES:
            raise ValueError(
                f"Unknown stage '{v}'. Valid stages: {sorted(VALID_STAGES)}"
            )
        return v


# ---------------------------------------------------------------------------
# Override requests
# ---------------------------------------------------------------------------

OverrideType = Literal[
    "risk_severity",
    "strategy_method",
    "ignore_element",
    "add_protection",
]


class OverrideRequest(BaseModel):
    type: OverrideType
    detection_id: str
    value: Optional[Any] = None
    reason: str
    user_confirmed: bool = False


class BatchOverrideRequest(BaseModel):
    overrides: list[OverrideRequest] = Field(..., max_length=50)


# ---------------------------------------------------------------------------
# HITL approve
# ---------------------------------------------------------------------------

class ApproveRequest(BaseModel):
    comment: Optional[str] = None


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    context_detection_ids: Optional[list[str]] = None
