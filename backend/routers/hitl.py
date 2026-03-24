from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.middleware.auth_middleware import require_auth
from backend.schemas.requests import ApproveRequest, BatchOverrideRequest, ChatRequest
from backend.schemas.responses import ApproveResponse, ChatResponse, OverrideResponse

router = APIRouter(prefix="/pipeline", tags=["hitl"])

# Valid HITL checkpoint names per API contract
VALID_CHECKPOINTS: frozenset[str] = frozenset(
    {"risk_review", "strategy_review", "execution_verify"}
)

# Checkpoint → next stage mapping (canonical short stage names)
CHECKPOINT_NEXT_STAGE: dict[str, str] = {
    "risk_review": "consent",
    "strategy_review": "sam",
    "execution_verify": "export",
}


# ---------------------------------------------------------------------------
# POST /pipeline/{session_id}/approve/{checkpoint}
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/approve/{checkpoint}",
    response_model=ApproveResponse,
    summary="Advance past a HITL pause checkpoint.",
)
async def approve_checkpoint(
    session_id: str,
    checkpoint: str,
    body: ApproveRequest,
    request: Request,
    session: Annotated[object, Depends(require_auth)],
) -> ApproveResponse:
    _assert_session_ownership(session, session_id)

    if checkpoint not in VALID_CHECKPOINTS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": {
                    "code": "CHECKPOINT_MISMATCH",
                    "message": (
                        f"Unknown checkpoint '{checkpoint}'. "
                        f"Valid checkpoints: {sorted(VALID_CHECKPOINTS)}."
                    ),
                    "details": {"checkpoint": checkpoint},
                }
            },
        )

    pipeline_service = getattr(request.app.state, "pipeline_service", None)
    if pipeline_service is not None:
        # PipelineService.approve_checkpoint takes a SessionRecord, not a
        # session_id string — retrieve it via the session_manager.
        session_manager = getattr(request.app.state, "session_manager", None)
        session_record = (
            session_manager.get_by_id(session_id)
            if session_manager is not None
            else None
        )
        if session_record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "PIPELINE_NOT_FOUND",
                        "message": f"No active session found for '{session_id}'.",
                        "details": {"session_id": session_id},
                    }
                },
            )
        try:
            pipeline_service.approve_checkpoint(
                session=session_record,
                checkpoint=checkpoint,
            )
            resumed = True
        except (ValueError, Exception) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": {
                        "code": "CHECKPOINT_MISMATCH",
                        "message": str(exc),
                        "details": {"checkpoint": checkpoint},
                    }
                },
            ) from exc
    else:
        resumed = False

    next_stage = CHECKPOINT_NEXT_STAGE.get(checkpoint)
    return ApproveResponse(
        checkpoint=checkpoint,
        next_stage=next_stage,
        pipeline_resumed=resumed,
    )


# ---------------------------------------------------------------------------
# POST /pipeline/{session_id}/override
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/override",
    response_model=OverrideResponse,
    summary="Apply batch overrides through the Safety Kernel.",
)
async def batch_override(
    session_id: str,
    body: BatchOverrideRequest,
    request: Request,
    session: Annotated[object, Depends(require_auth)],
) -> OverrideResponse:
    """Validate and apply each override through the Safety Kernel.

    Partial success is permitted — rejected overrides are returned in the
    'rejected' list alongside any successfully applied ones.  This endpoint
    does NOT trigger a pipeline rerun; the frontend must call /rerun
    separately after batching all desired overrides.
    """
    _assert_session_ownership(session, session_id)

    safety_kernel = getattr(request.app.state, "safety_kernel", None)

    if safety_kernel is None:
        # Safety Kernel not yet wired — return all as pending/rejected with
        # a clear error so the frontend knows the service is unavailable.
        from backend.schemas.responses import RejectedOverride

        rejected = [
            RejectedOverride(
                detection_id=ov.detection_id,
                type=ov.type,
                error_code="PIPELINE_ERROR",
                message="Safety Kernel service is not available.",
            )
            for ov in body.overrides
        ]
        return OverrideResponse(applied=[], rejected=rejected)

    return safety_kernel.apply_batch(session_id=session_id, overrides=body.overrides)


# ---------------------------------------------------------------------------
# POST /pipeline/{session_id}/chat
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/chat",
    response_model=ChatResponse,
    summary="Send a natural-language query to the Coordinator Agent.",
)
async def chat(
    session_id: str,
    body: ChatRequest,
    request: Request,
    session: Annotated[object, Depends(require_auth)],
) -> ChatResponse:
    """Synchronous NL query endpoint.

    For read/query intents the response is returned directly.
    For write intents that trigger pipeline actions, progress is streamed
    via the WebSocket channel; the response indicates the action was taken.

    This implementation returns a mock response until the Coordinator Agent
    service is wired by another agent.
    """
    _assert_session_ownership(session, session_id)

    coordinator = getattr(request.app.state, "coordinator_agent", None)
    if coordinator is not None:
        return coordinator.handle(
            session_id=session_id,
            message=body.message,
            context_detection_ids=body.context_detection_ids,
        )

    # Mock response — Coordinator Agent not yet available
    from backend.schemas.responses import ParsedIntentResponse
    return ChatResponse(
        intent=ParsedIntentResponse(
            action="query",
            target_stage=None,
            target_elements=[],
            confidence=0.0,
            natural_language=body.message,
        ),
        response_text=(
            "The Coordinator Agent is not yet available. "
            "Your message has been received: "
            f'"{body.message}"'
        ),
        pipeline_action_taken=None,
        suggestions=[
            "Check /pipeline/{session_id}/status for current state.",
            "Use /pipeline/{session_id}/override to modify individual elements.",
        ],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_session_ownership(session: object, session_id: str) -> None:
    if getattr(session, "session_id", None) != session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "PIPELINE_NOT_FOUND",
                    "message": f"No pipeline found for session '{session_id}'.",
                    "details": {"session_id": session_id},
                }
            },
        )
