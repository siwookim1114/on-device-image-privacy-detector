from __future__ import annotations

import asyncio
import os
import struct
from typing import Annotated, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from backend.middleware.auth_middleware import require_auth
from backend.schemas.requests import RerunRequest, RunConfig
from backend.schemas.responses import (
    AuditEntry,
    DetectionResult,
    ErrorDetail,
    ExecutionResult,
    HitlStatus,
    ImageMeta,
    PipelineResultsResponse,
    PipelineRunResponse,
    PipelineStatusResponse,
    PipelineTiming,
    RerunResponse,
    RiskAssessmentResponse,
    StrategyResponse,
)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# MIME types accepted as valid images
ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/tiff"}
)

# Ordered stage dependency map — every stage listed after a given stage
# depends on that stage being complete.
STAGE_ORDER: list[str] = [
    "detection",
    "risk",
    "consent",
    "strategy",
    "sam",
    "execution",
    "export",
]


def _stages_from(from_stage: str) -> tuple[list[str], list[str]]:
    """Return (stages_to_rerun, stages_cached) given a starting stage name."""
    idx = STAGE_ORDER.index(from_stage)
    return STAGE_ORDER[idx:], STAGE_ORDER[:idx]
# POST /pipeline/run

@router.post(
    "/run",
    response_model=PipelineRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload an image and start the privacy pipeline.",
)
async def run_pipeline(
    request: Request,
    background_tasks: BackgroundTasks,
    session: Annotated[object, Depends(require_auth)],
    image: UploadFile = File(..., description="Image file to protect."),
    config: Optional[str] = Form(
        default=None,
        description="JSON-encoded RunConfig (optional).",
    ),
) -> PipelineRunResponse:
    """Accept a multipart upload, validate the image, persist it to the
    temp upload directory, and enqueue the pipeline as a background task.

    Returns HTTP 202 immediately with session_id and image metadata.
    """
    # -- Parse run config -------------------------------------------------------
    run_config: RunConfig
    if config:
        try:
            run_config = RunConfig.model_validate_json(config)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": f"Invalid config JSON: {exc}",
                        "details": {},
                    }
                },
            ) from exc
    else:
        run_config = RunConfig()

    # -- Validate image MIME type ------------------------------------------------
    raw_bytes = await image.read(2048)  # read magic bytes only
    try:
        import magic as _magic
        detected_mime: str = _magic.from_buffer(raw_bytes, mime=True)
    except ImportError:
        # python-magic not installed — fall back to Content-Type header
        detected_mime = (image.content_type or "application/octet-stream").split(";")[0].strip()
    if detected_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": {
                    "code": "UNSUPPORTED_FILE",
                    "message": (
                        f"File type '{detected_mime}' is not supported. "
                        f"Accepted types: {sorted(ALLOWED_MIME_TYPES)}."
                    ),
                    "details": {"detected_mime": detected_mime},
                }
            },
        )

    # Seek back and read the full file
    await image.seek(0)
    file_bytes = await image.read()
    file_size = len(file_bytes)

    # -- Persist to temp upload dir ---------------------------------------------
    upload_dir: str = request.app.state.settings.upload_dir
    os.makedirs(upload_dir, exist_ok=True)

    session_id: str = session.session_id  # type: ignore[attr-defined]
    safe_filename = f"{session_id}_{image.filename or 'upload'}"
    upload_path = os.path.join(upload_dir, safe_filename)

    with open(upload_path, "wb") as fh:
        fh.write(file_bytes)

    # -- Resolve image dimensions without importing heavy CV libs here ----------
    try:
        width, height = _fast_image_dims(file_bytes, detected_mime)
    except Exception:
        width, height = 0, 0

    # -- Update session with upload metadata ------------------------------------
    session_manager = request.app.state.session_manager
    # Store image path + dimensions on the SessionRecord directly (the real
    # SessionManager exposes SessionRecord as a mutable dataclass; the stub
    # no-ops update_image_meta for backwards compat).
    session_record = session_manager.get_by_id(session_id)
    if session_record is not None:
        session_record.image_path = upload_path
        session_record.config = run_config.model_dump()
    if hasattr(session_manager, "update_image_meta"):
        session_manager.update_image_meta(
            session_id=session_id,
            filename=image.filename or "upload",
            upload_path=upload_path,
            width=width,
            height=height,
            size_bytes=file_size,
            mime_type=detected_mime,
        )

    # -- Check for concurrent pipeline conflicts and enqueue -------------------
    pipeline_service = getattr(request.app.state, "pipeline_service", None)
    if pipeline_service is not None and session_record is not None:
        if session_record.is_running:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": {
                        "code": "PIPELINE_RUNNING",
                        "message": "A pipeline is already active for this session.",
                        "details": {"session_id": session_id},
                    }
                },
            )
        # run_pipeline() is async and needs the running event loop.
        # BackgroundTasks cannot schedule coroutines directly — use asyncio
        # task creation instead, which is safe inside an async endpoint.
        loop = asyncio.get_event_loop()
        asyncio.ensure_future(pipeline_service.run_pipeline(session_record, loop))

    return PipelineRunResponse(
        session_id=session_id,
        status="queued",
        image_meta=ImageMeta(
            filename=image.filename or "upload",
            width=width,
            height=height,
            size_bytes=file_size,
            mime_type=detected_mime,
        ),
    )
# GET /pipeline/{session_id}/status

@router.get(
    "/{session_id}/status",
    response_model=PipelineStatusResponse,
    summary="Poll the current pipeline status for a session.",
)
async def get_pipeline_status(
    session_id: str,
    request: Request,
    session: Annotated[object, Depends(require_auth)],
) -> PipelineStatusResponse:
    _assert_session_ownership(session, session_id)

    session_manager = getattr(request.app.state, "session_manager", None)
    session_record = (
        session_manager.get_by_id(session_id) if session_manager is not None else None
    )

    if session_record is None:
        # No session found — return a minimal not-found status rather than 404
        # so the frontend can poll without needing special error handling on first call.
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

    hitl = HitlStatus(
        waiting=session_record.hitl_pending_approval,
        checkpoint=session_record.hitl_checkpoint,
        checkpoint_reason=None,
        elements_requiring_review=session_record.hitl_elements_requiring_review,
        actions_available=session_record.hitl_actions_available,
    )

    # Build timing from session_record.stage_timings dict
    stage_timings_int = {k: int(v) for k, v in session_record.stage_timings.items()}
    timing = PipelineTiming(stage_timings=stage_timings_int)

    error_detail = None
    if session_record.error_code:
        error_detail = ErrorDetail(
            code=session_record.error_code,
            message=session_record.error_message or "",
        )

    # Normalize internal status values to canonical API contract values
    _STATUS_MAP: dict[str, str] = {
        "hitl_paused": "paused_hitl",
        "idle": "queued",          # treat idle as queued for external consumers
        "cancelled": "failed",     # map unsupported value to failed for safety
    }
    api_status = _STATUS_MAP.get(session_record.status, session_record.status)

    return PipelineStatusResponse(
        session_id=session_id,
        status=api_status,  # type: ignore[arg-type]
        current_stage=session_record.current_stage,
        hitl=hitl,
        timing=timing,
        error=error_detail,
    )
# GET /pipeline/{session_id}/results

@router.get(
    "/{session_id}/results",
    response_model=PipelineResultsResponse,
    summary="Retrieve full structured results after pipeline completion.",
)
async def get_pipeline_results(
    session_id: str,
    request: Request,
    session: Annotated[object, Depends(require_auth)],
) -> PipelineResultsResponse:
    _assert_session_ownership(session, session_id)

    session_manager = getattr(request.app.state, "session_manager", None)
    session_record = (
        session_manager.get_by_id(session_id) if session_manager is not None else None
    )
    if session_record is None:
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

    if session_record.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "PIPELINE_NOT_FOUND",
                    "message": f"Results not yet available; pipeline status is '{session_record.status}'.",
                    "details": {"session_id": session_id, "status": session_record.status},
                }
            },
        )

    pipeline_service = getattr(request.app.state, "pipeline_service", None)
    if pipeline_service is not None:
        raw = pipeline_service.get_results(session_record)
    else:
        raw = {
            "session_id": session_id,
            "status": session_record.status,
            "detections": session_record.detections,
            "risk_result": session_record.risk_result,
            "strategy_result": session_record.strategy_result,
            "execution_report": session_record.execution_report,
            "audit_trail": session_record.audit_trail,
            "stage_timings": session_record.stage_timings,
        }

    stage_timings_int = {k: int(v) for k, v in (raw.get("stage_timings") or {}).items()}
    timing = PipelineTiming(stage_timings=stage_timings_int)

    # Map raw detections into DetectionResult schema
    detections_out: list[DetectionResult] = []
    raw_detections = raw.get("detections")
    if raw_detections is not None:
        raw_list = []
        if hasattr(raw_detections, "faces"):
            for f in (raw_detections.faces or []):
                raw_list.append({
                    "detection_id": getattr(f, "detection_id", ""),
                    "element_type": "face",
                    "bbox": list(getattr(f, "bbox", [0, 0, 0, 0])),
                    "confidence": float(getattr(f, "confidence", 0.0)),
                    "metadata": {},
                })
            for t in (raw_detections.text_regions or []):
                raw_list.append({
                    "detection_id": getattr(t, "detection_id", ""),
                    "element_type": "text",
                    "bbox": list(getattr(t, "bbox", [0, 0, 0, 0])),
                    "confidence": float(getattr(t, "confidence", 0.0)),
                    "metadata": {"text": getattr(t, "text", "")},
                })
            for o in (raw_detections.objects or []):
                raw_list.append({
                    "detection_id": getattr(o, "detection_id", ""),
                    "element_type": "object",
                    "bbox": list(getattr(o, "bbox", [0, 0, 0, 0])),
                    "confidence": float(getattr(o, "confidence", 0.0)),
                    "metadata": {"label": getattr(o, "label", "")},
                })
        elif isinstance(raw_detections, list):
            raw_list = raw_detections
        for item in raw_list:
            try:
                detections_out.append(DetectionResult(**item) if isinstance(item, dict) else item)
            except Exception:
                pass

    # Map raw risk_result into RiskAssessmentResponse schema
    risk_assessments_out: list[RiskAssessmentResponse] = []
    raw_risk = raw.get("risk_result")
    if raw_risk is not None:
        assessments = []
        if hasattr(raw_risk, "risk_assessments"):
            assessments = raw_risk.risk_assessments or []
        elif isinstance(raw_risk, list):
            assessments = raw_risk
        for a in assessments:
            try:
                if hasattr(a, "detection_id"):
                    sev = getattr(a, "severity", None)
                    sev_val = sev.value if hasattr(sev, "value") else str(sev or "low").lower()
                    consent = getattr(a, "consent_status", None)
                    consent_val = consent.value if hasattr(consent, "value") else (str(consent).lower() if consent else None)
                    screen = getattr(a, "screen_state", None)
                    screen_val = screen.value if hasattr(screen, "value") else (str(screen).lower() if screen else None)
                    risk_assessments_out.append(RiskAssessmentResponse(
                        detection_id=a.detection_id,
                        severity=sev_val,  # type: ignore[arg-type]
                        consent_status=consent_val,  # type: ignore[arg-type]
                        screen_state=screen_val,  # type: ignore[arg-type]
                        escalation_reasons=list(getattr(a, "escalation_reasons", [])),
                    ))
                elif isinstance(a, dict):
                    risk_assessments_out.append(RiskAssessmentResponse(**a))
            except Exception:
                pass

    # Map raw strategy_result into StrategyResponse schema
    strategies_out: list[StrategyResponse] = []
    raw_strategy = raw.get("strategy_result")
    if raw_strategy is not None:
        strats = []
        if hasattr(raw_strategy, "strategies"):
            strats = raw_strategy.strategies or []
        elif isinstance(raw_strategy, list):
            strats = raw_strategy
        for s in strats:
            try:
                if hasattr(s, "detection_id"):
                    method = getattr(s, "method", None)
                    method_val = method.value if hasattr(method, "value") else str(method or "none").lower()
                    params = getattr(s, "parameters", None) or getattr(s, "params", None) or {}
                    strategies_out.append(StrategyResponse(
                        detection_id=s.detection_id,
                        method=method_val,  # type: ignore[arg-type]
                        parameters=dict(params) if params else {},
                    ))
                elif isinstance(s, dict):
                    # Normalize params -> parameters
                    if "params" in s and "parameters" not in s:
                        s = {**s, "parameters": s.pop("params")}
                    strategies_out.append(StrategyResponse(**s))
            except Exception:
                pass

    # Map raw execution_report into ExecutionResult schema
    execution_out: list[ExecutionResult] = []
    raw_execution = raw.get("execution_report")
    if raw_execution is not None:
        transforms = []
        if hasattr(raw_execution, "transformations_applied"):
            transforms = raw_execution.transformations_applied or []
        elif isinstance(raw_execution, list):
            transforms = raw_execution
        for t in transforms:
            try:
                if hasattr(t, "detection_id"):
                    execution_out.append(ExecutionResult(
                        detection_id=t.detection_id,
                        applied=bool(getattr(t, "applied", True)),
                        method_used=getattr(t, "method_used", None),
                        patch_applied=bool(getattr(t, "patch_applied", False)),
                    ))
                elif isinstance(t, dict):
                    execution_out.append(ExecutionResult(**t))
            except Exception:
                pass

    # Map audit trail
    audit_out: list[AuditEntry] = []
    for entry in (raw.get("audit_trail") or []):
        try:
            if isinstance(entry, dict):
                audit_out.append(AuditEntry(**entry))
        except Exception:
            pass

    return PipelineResultsResponse(
        session_id=session_id,
        status=raw.get("status", "completed"),
        detections=detections_out,
        risk_assessments=risk_assessments_out,
        strategies=strategies_out,
        execution=execution_out,
        audit_trail=audit_out,
        timing=timing,
    )
# POST /pipeline/{session_id}/rerun

@router.post(
    "/{session_id}/rerun",
    response_model=RerunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Re-enter the pipeline at a specific stage.",
)
async def rerun_pipeline(
    session_id: str,
    body: RerunRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    session: Annotated[object, Depends(require_auth)],
) -> RerunResponse:
    _assert_session_ownership(session, session_id)

    stages_to_rerun, stages_cached = _stages_from(body.from_stage)

    pipeline_service = getattr(request.app.state, "pipeline_service", None)
    if pipeline_service is not None:
        session_manager = getattr(request.app.state, "session_manager", None)
        session_record = (
            session_manager.get_by_id(session_id) if session_manager is not None else None
        )
        if session_record is not None:
            if session_record.is_running:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "error": {
                            "code": "PIPELINE_RUNNING",
                            "message": "Cannot rerun while pipeline is active.",
                            "details": {"session_id": session_id},
                        }
                    },
                )
            loop = asyncio.get_event_loop()
            asyncio.ensure_future(
                pipeline_service.rerun_from(session_record, body.from_stage, loop)
            )

    return RerunResponse(
        session_id=session_id,
        status="queued",
        stages_to_rerun=stages_to_rerun,
        stages_cached=stages_cached,
    )
# Helpers

def _assert_session_ownership(session: object, session_id: str) -> None:
    """Raise 404 if the authenticated session does not match the path param."""
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


def _fast_image_dims(data: bytes, mime: str) -> tuple[int, int]:
    """Extract image dimensions from raw bytes without PIL/CV2.

    Supports JPEG, PNG, and WebP via header parsing only.
    Falls back to (0, 0) if the format is unrecognised.
    """
    if mime == "image/png" and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)

    if mime == "image/jpeg":
        i = 2
        while i < len(data) - 8:
            if data[i] != 0xFF:
                break
            marker = data[i + 1]
            seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
            if marker in (0xC0, 0xC1, 0xC2):  # SOF0, SOF1, SOF2
                h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                return int(w), int(h)
            i += 2 + seg_len
        return 0, 0

    if mime == "image/webp" and len(data) >= 30:
        # RIFF....WEBPVP8 <space>
        if data[8:12] == b"WEBP" and data[12:16] == b"VP8 ":
            w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
            h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
            return int(w + 1), int(h + 1)

    return 0, 0
