"""
pipeline_service.py — Bridges async FastAPI with the synchronous PipelineOrchestrator.

Architecture
------------
PipelineOrchestrator.run() is synchronous, CPU-bound, and was designed to be
called from a script.  This service wraps it stage-by-stage so that:

1. WebSocket progress events (stage_start / stage_complete / hitl_checkpoint)
   are emitted between phases.
2. HITL gates pause execution in the worker thread and wait for the async
   event loop to signal approval via threading.Event.
3. Selective re-execution (rerun_from_stage) resumes from any cached stage
   output without repeating upstream phases.
4. Fallback-only mode (no llama-server) is gracefully handled.

Threading model
---------------
- FastAPI runs on an asyncio event loop.
- The pipeline executes in a thread-pool worker (run_in_executor).
- The worker calls websocket_manager.broadcast_from_thread() to emit WS events
  across the thread boundary using asyncio.run_coroutine_threadsafe().
- Session state mutations from the worker thread use session._lock (via
  session_manager) for atomic multi-field updates.

Imported names from the ML pipeline are resolved at call time so that the
backend module can be imported even when optional ML deps (MTCNN, EasyOCR,
YOLOv8, MobileSAM) are not installed.
"""

import asyncio
import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
# Project root on sys.path so ML imports resolve correctly.
# backend/ lives at  <project_root>/backend/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# Local imports (no ML deps at module scope)
from services.session_manager import (  # noqa: E402
    STAGE_DEPENDENCY_MAP,
    VALID_STAGES,
    SessionRecord,
    SessionManager,
)
from services.websocket_manager import WebSocketManager  # noqa: E402
# Stage metadata used for display_name / description in stage_start events
_STAGE_META: Dict[str, Dict[str, str]] = {
    "detection": {
        "display_name": "Detection",
        "description": "Detecting faces, text, and objects (MTCNN + EasyOCR + YOLOv8)",
    },
    "risk": {
        "display_name": "Risk Assessment",
        "description": "Assessing privacy risk severity for each detected element",
    },
    "consent": {
        "display_name": "Consent Identity",
        "description": "Matching faces against consent database",
    },
    "strategy": {
        "display_name": "Strategy",
        "description": "Recommending obfuscation strategies per element",
    },
    "sam": {
        "display_name": "SAM Segmentation",
        "description": "Generating precise pixel masks (MobileSAM)",
    },
    "execution": {
        "display_name": "Execution",
        "description": "Applying obfuscation to the image",
    },
    "export": {
        "display_name": "Export",
        "description": "Exporting risk map, JSON results, and protected image",
    },
}

# HITL gate configuration: which stages trigger a checkpoint pause and which
# checkpoint name they use.  The gate fires AFTER the stage completes.
_HITL_GATES: Dict[str, str] = {
    "risk":      "risk_review",
    "strategy":  "strategy_review",
    "execution": "execution_verify",
}
# PipelineService


class PipelineService:
    """
    Manages per-session pipeline execution with WebSocket progress streaming.

    One shared PipelineOrchestrator instance is reused across all sessions
    (ML model weights are loaded once).  A ThreadPoolExecutor with
    max_workers=2 limits concurrent pipeline runs.
    """

    def __init__(
        self,
        ws_manager: Optional[WebSocketManager] = None,
        session_manager: Optional[SessionManager] = None,
        # Accept kwargs passed by main.py (another agent's code) gracefully
        config_path: Optional[str] = None,
        upload_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        max_concurrent: int = 2,
        provenance_service: Optional[Any] = None,
    ) -> None:
        """
        Args:
            ws_manager:          WebSocketManager singleton.
            session_manager:     SessionManager singleton (optional).
            config_path:         Ignored — kept for compatibility.
            upload_dir:          Ignored — kept for compatibility.
            results_dir:         Ignored — the orchestrator manages its own output_dir.
            max_concurrent:      ThreadPoolExecutor worker count (default 2).
            provenance_service:  Optional ProvenanceService instance for event logging.
        """
        self.ws_manager: WebSocketManager = ws_manager or WebSocketManager()
        self.session_manager = session_manager
        self._provenance: Optional[Any] = provenance_service
        self._orchestrator: Any = None          # PipelineOrchestrator
        self._orchestrator_ready: bool = False
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="pipeline"
        )
    # Startup / shutdown

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Load ML models into the shared PipelineOrchestrator.

        Call once from the FastAPI lifespan startup handler.

        Args:
            config: Optional dict of PipelineConfig kwargs to override defaults.
                    Useful for enabling fallback_only mode on startup.
        """
        try:
            from agents.pipeline import PipelineConfig, PipelineOrchestrator

            cfg_kwargs = config or {}
            pipeline_cfg = PipelineConfig(**cfg_kwargs)
            self._orchestrator = PipelineOrchestrator(config=pipeline_cfg)
            self._orchestrator_ready = True
            logger.info("PipelineOrchestrator initialised (use_vlm=%s)", pipeline_cfg.use_vlm())
        except Exception as exc:
            logger.error(
                "PipelineOrchestrator initialisation failed — "
                "fallback_only mode will be used.  Error: %s",
                exc,
            )
            # Attempt again in fallback_only mode
            try:
                from agents.pipeline import PipelineConfig, PipelineOrchestrator

                pipeline_cfg = PipelineConfig(fallback_only=True)
                self._orchestrator = PipelineOrchestrator(config=pipeline_cfg)
                self._orchestrator_ready = True
                logger.warning("PipelineOrchestrator running in fallback_only mode")
            except Exception as exc2:
                logger.error(
                    "Fallback initialisation also failed: %s", exc2
                )
                self._orchestrator_ready = False

    def shutdown(self) -> None:
        """Release thread pool and ML orchestrator resources."""
        self._executor.shutdown(wait=False)
        if self._orchestrator is not None:
            try:
                self._orchestrator.close()
            except Exception:
                pass
    # Primary entry point: run a full pipeline

    async def run_pipeline(
        self,
        session: SessionRecord,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Start pipeline execution for *session* in a worker thread.

        Returns immediately (202); the worker broadcasts WS events as
        each stage completes.

        Args:
            session: The validated session record with image_path and config set.
            loop:    The running event loop (captured by the router before the
                     executor call so the worker can schedule WS coroutines).
        """
        if session.status not in ("idle", "queued"):
            raise PipelineRunningError(session.session_id)

        session.status = "queued"

        loop.run_in_executor(
            self._executor,
            self._execute_sync,
            session,
            loop,
            None,   # from_stage=None means run all stages
        )
    # Selective re-execution

    async def rerun_from(
        self,
        session: SessionRecord,
        from_stage: str,
        loop: asyncio.AbstractEventLoop,
    ) -> Dict[str, Any]:
        """
        Re-execute the pipeline starting from *from_stage*, reusing cached
        upstream stage outputs.

        Returns immediately (202) with a summary of which stages will be
        re-run vs. served from cache.

        Args:
            session:    Active session (must not be currently running).
            from_stage: Stage name to restart from (must be in VALID_STAGES).
            loop:       Running event loop.

        Raises:
            InvalidStageError:    Unknown from_stage.
            PipelineRunningError: Pipeline is already active.
            CheckpointMissingError: Required upstream cache is absent.
        """
        if from_stage not in VALID_STAGES:
            raise InvalidStageError(from_stage)

        if session.is_running:
            raise PipelineRunningError(session.session_id)

        # Validate that we have the required cached data for from_stage
        _required_cache: Dict[str, str] = {
            "risk":      "detections",
            "consent":   "detections",
            "strategy":  "risk_result",
            "sam":       "strategy_result",
            "execution": "strategy_result",
            "export":    "pipeline_output",
        }
        required_attr = _required_cache.get(from_stage)
        if required_attr and getattr(session, required_attr, None) is None:
            raise CheckpointMissingError(from_stage, required_attr)

        stages_to_rerun = [from_stage] + STAGE_DEPENDENCY_MAP[from_stage]
        stages_cached = [s for s in VALID_STAGES if s not in stages_to_rerun]

        session.status = "queued"

        loop.run_in_executor(
            self._executor,
            self._execute_sync,
            session,
            loop,
            from_stage,
        )

        return {
            "stages_to_rerun": stages_to_rerun,
            "stages_cached": stages_cached,
        }
    # HITL approval — called from the REST /approve endpoint

    def approve_checkpoint(
        self,
        session: SessionRecord,
        checkpoint: str,
    ) -> None:
        """
        Release a HITL pause.

        The worker thread is blocked on session.hitl_event.  Setting it here
        from the async event loop unblocks the worker so it proceeds to the
        next stage.

        Args:
            session:    The paused session.
            checkpoint: The checkpoint name that is being approved
                        (e.g. "risk_review").  Must match session.hitl_checkpoint.

        Raises:
            CheckpointMismatchError: The checkpoint name does not match.
        """
        if session.hitl_checkpoint != checkpoint:
            raise CheckpointMismatchError(
                expected=session.hitl_checkpoint or "", got=checkpoint
            )

        session.hitl_pending_approval = False
        session.hitl_event.set()
        logger.info(
            "HITL approved: session_id=%s  checkpoint=%s",
            session.session_id,
            checkpoint,
        )
    # Provenance helper (fire-and-forget; never raises)

    def _prov(
        self,
        session_id: str,
        event_type: Any,
        phase: str,
        data: Dict[str, Any],
        detection_id: Optional[str] = None,
    ) -> None:
        """Emit one provenance event, silently ignoring all errors."""
        if self._provenance is None:
            return
        try:
            self._provenance.record(session_id, event_type, phase, data, detection_id)
        except Exception as exc:
            logger.debug("Provenance record skipped: %s", exc)
    # Synchronous execution (runs in worker thread)

    def _execute_sync(
        self,
        session: SessionRecord,
        loop: asyncio.AbstractEventLoop,
        from_stage: Optional[str],
    ) -> None:
        """
        Run pipeline stages sequentially in a worker thread.

        Emits WS events via broadcast_from_thread() at stage boundaries.
        Blocks at HITL gates until the event loop signals approval.

        Args:
            session:    Mutable session record.
            loop:       The asyncio event loop for WS broadcasting.
            from_stage: If not None, skip stages that precede this stage
                        and load their outputs from session cache instead.
        """
        session.status = "running"
        pipeline_start = time.perf_counter()

        # Lazy import so the module is importable without utils on sys.path
        try:
            from utils.models import ProvenanceEventType as _PET
        except ImportError:
            _PET = None  # type: ignore[assignment]

        # Open a provenance session (fire-and-forget; no-op if service absent)
        if self._provenance is not None and _PET is not None:
            try:
                cfg_snap = session.config or {}
                self._provenance.open_session(
                    session_id=session.session_id,
                    image_path=session.image_path or "",
                    run_mode=cfg_snap.get("mode", "auto"),
                    fallback_only=cfg_snap.get("fallback_only", False),
                    user_id=getattr(session, "user_id", None),
                )
            except Exception as _pe:
                logger.debug("Provenance open_session skipped: %s", _pe)

        def emit(event_type: str, payload: dict) -> None:
            self.ws_manager.broadcast_from_thread(
                session.session_id, event_type, payload, loop
            )

        def elapsed_ms(start: float) -> float:
            return (time.perf_counter() - start) * 1000.0

        try:
            if not self._orchestrator_ready or self._orchestrator is None:
                raise RuntimeError(
                    "PipelineOrchestrator not initialised — "
                    "cannot run pipeline"
                )

            orc = self._orchestrator
            cfg = session.config or {}
            fallback_only: bool = cfg.get("fallback_only", False) or (
                not cfg.get("phases", {}).get("run_vlm_risk", True)
                and not cfg.get("phases", {}).get("run_vlm_strategy", True)
                and not cfg.get("phases", {}).get("run_vlm_execution", True)
            )

            # Resolve processing mode from RunConfig (canonical values: auto, hybrid, manual)
            # auto   → no HITL pauses, fully automated
            # hybrid → confidence-gated HITL (default: pause on critical)
            # manual → pause at every HITL checkpoint
            run_mode: str = cfg.get("mode", "auto")

            # Load user PrivacyProfile from MongoDB and pass to orchestrator when available
            user_profile = None
            if self.session_manager is not None:
                try:
                    profile_service = getattr(
                        self.session_manager, "_profile_service", None
                    )
                    # profile_service may also be available via the app state reference
                    # stored at construction time; fall back gracefully if absent.
                    if profile_service is None and hasattr(orc, "profile_service"):
                        profile_service = orc.profile_service
                    if profile_service is not None:
                        try:
                            user_profile = asyncio.run_coroutine_threadsafe(
                                profile_service.get_profile(session.session_id),
                                loop,
                            ).result(timeout=5)
                        except Exception as _pe:
                            logger.debug("Could not load user profile: %s", _pe)
                except Exception as _exc:
                    logger.debug("Profile load skipped: %s", _exc)

            if user_profile is not None and hasattr(orc, "set_user_profile"):
                try:
                    orc.set_user_profile(user_profile)
                except Exception as _exc:
                    logger.debug("set_user_profile not supported: %s", _exc)

            image_path: str = session.image_path  # type: ignore[assignment]

            # Determine which stages to run
            if from_stage is not None:
                idx = VALID_STAGES.index(from_stage)
                run_stages = VALID_STAGES[idx:]
            else:
                run_stages = VALID_STAGES[:]
            # Stage: detection
            if "detection" in run_stages:
                stage = "detection"
                meta = _STAGE_META[stage]
                emit("stage_start", {
                    "stage": stage,
                    "display_name": meta["display_name"],
                    "description": meta["description"],
                })
                if _PET is not None:
                    self._prov(session.session_id, _PET.STAGE_START, stage, {"stage": stage})
                t0 = time.perf_counter()
                session.current_stage = stage
                try:
                    detections = orc.detection_agent.run(image_path)
                    annotated_image = orc.detection_agent.get_annotated_image()
                    session.detections = detections
                    elapsed = elapsed_ms(t0)
                    session.stage_timings["detection_ms"] = elapsed
                    emit("stage_complete", {
                        "stage": stage,
                        "summary": {
                            "faces": len(detections.faces),
                            "text_regions": len(detections.text_regions),
                            "objects": len(detections.objects),
                        },
                        "elapsed_ms": elapsed,
                    })
                    if _PET is not None:
                        self._prov(session.session_id, _PET.STAGE_COMPLETE, stage, {
                            "faces": len(detections.faces),
                            "text_regions": len(detections.text_regions),
                            "objects": len(detections.objects),
                            "elapsed_ms": elapsed,
                        })
                        for face in detections.faces:
                            self._prov(session.session_id, _PET.FACE_DETECTED, stage, {
                                "confidence": face.confidence,
                                "size": getattr(face, "size", None),
                                "clarity": getattr(face, "clarity", None),
                            }, detection_id=face.id)
                        for text in detections.text_regions:
                            self._prov(session.session_id, _PET.TEXT_DETECTED, stage, {
                                "text_type": getattr(text, "text_type", None),
                                "confidence": text.confidence,
                            }, detection_id=text.id)
                        for obj in detections.objects:
                            self._prov(session.session_id, _PET.OBJECT_DETECTED, stage, {
                                "object_class": getattr(obj, "object_class", None),
                                "contains_screen": getattr(obj, "contains_screen", False),
                                "confidence": obj.confidence,
                            }, detection_id=obj.id)
                except Exception as exc:
                    if _PET is not None:
                        self._prov(session.session_id, _PET.STAGE_ERROR, stage, {"error": str(exc)})
                    self._handle_stage_error(session, stage, exc, emit)
                    return
            else:
                # Use cached
                detections = session.detections
                annotated_image = None
                if annotated_image is None and detections is not None:
                    try:
                        from PIL import Image as _PIL_Image
                        annotated_image = _PIL_Image.open(image_path)
                    except Exception:
                        pass
            # Stage: risk
            if "risk" in run_stages:
                stage = "risk"
                meta = _STAGE_META[stage]
                emit("stage_start", {
                    "stage": stage,
                    "display_name": meta["display_name"],
                    "description": meta["description"],
                })
                if _PET is not None:
                    self._prov(session.session_id, _PET.STAGE_START, stage, {
                        "stage": stage,
                        "fallback_only": fallback_only,
                    })
                t0 = time.perf_counter()
                session.current_stage = stage
                try:
                    if fallback_only:
                        from PIL import Image as _PILImage

                        img = _PILImage.open(image_path)
                        w, h = img.size
                        image_context = {
                            "width": w,
                            "height": h,
                            "total_faces": len(detections.faces),
                            "total_texts": len(detections.text_regions),
                            "total_objects": len(detections.objects),
                        }
                        wall_start = time.time()
                        assessments = orc.risk_agent._tool_based_assessment(
                            detections, image_context
                        )
                        risk_result = orc.risk_agent._build_result(
                            assessments, image_path, wall_start
                        )
                    else:
                        risk_result = orc.risk_agent.run(detections, annotated_image)

                    session.risk_result = risk_result
                    elapsed = elapsed_ms(t0)
                    session.stage_timings["risk_assessment_ms"] = elapsed
                    emit("stage_complete", {
                        "stage": stage,
                        "summary": {
                            "assessments": len(risk_result.risk_assessments),
                            "overall_risk": risk_result.overall_risk_level.value,
                        },
                        "elapsed_ms": elapsed,
                    })
                    if _PET is not None:
                        phase_type = _PET.RISK_ASSESSED_P1 if fallback_only else _PET.RISK_ASSESSED_P2
                        self._prov(session.session_id, _PET.STAGE_COMPLETE, stage, {
                            "assessments": len(risk_result.risk_assessments),
                            "overall_risk": risk_result.overall_risk_level.value,
                            "elapsed_ms": elapsed,
                        })
                        for assessment in risk_result.risk_assessments:
                            screen_state = getattr(assessment, "screen_state", None)
                            self._prov(session.session_id, phase_type, stage, {
                                "element_type": assessment.element_type,
                                "severity": assessment.severity.value,
                                "requires_protection": assessment.requires_protection,
                                "screen_state": screen_state,
                            }, detection_id=assessment.detection_id)
                            if screen_state in ("verified_on", "verified_off"):
                                self._prov(session.session_id, _PET.SCREEN_STATE_VERIFIED, stage, {
                                    "screen_state": screen_state,
                                }, detection_id=assessment.detection_id)
                except Exception as exc:
                    if _PET is not None:
                        self._prov(session.session_id, _PET.STAGE_ERROR, stage, {"error": str(exc)})
                    self._handle_stage_error(session, stage, exc, emit)
                    return

                # HITL gate: risk_review
                if self._should_pause_hitl(session, stage):
                    self._block_for_hitl(session, "risk_review", emit)
                    if session.status == "failed":
                        return
            else:
                risk_result = session.risk_result
            # Stage: consent
            if "consent" in run_stages:
                stage = "consent"
                meta = _STAGE_META[stage]
                emit("stage_start", {
                    "stage": stage,
                    "display_name": meta["display_name"],
                    "description": meta["description"],
                })
                if _PET is not None:
                    self._prov(session.session_id, _PET.STAGE_START, stage, {"stage": stage})
                t0 = time.perf_counter()
                session.current_stage = stage
                try:
                    if orc.consent_agent is not None:
                        risk_result = orc.consent_agent.run(detections, risk_result)
                        session.risk_result = risk_result
                    elapsed = elapsed_ms(t0)
                    session.stage_timings["consent_identity_ms"] = elapsed
                    emit("stage_complete", {
                        "stage": stage,
                        "summary": {"status": "complete"},
                        "elapsed_ms": elapsed,
                    })
                    if _PET is not None:
                        self._prov(session.session_id, _PET.STAGE_COMPLETE, stage, {
                            "status": "complete",
                            "elapsed_ms": elapsed,
                        })
                        for assessment in risk_result.risk_assessments:
                            if assessment.element_type != "face":
                                continue
                            consent = getattr(assessment, "consent_status", None)
                            person_id = getattr(assessment, "person_id", None)
                            if person_id:
                                self._prov(session.session_id, _PET.FACE_MATCH_HIT, stage, {
                                    "person_label": getattr(assessment, "person_label", None),
                                    "consent_status": consent.value if hasattr(consent, "value") else str(consent),
                                    "consent_confidence": getattr(assessment, "consent_confidence", 0.0),
                                }, detection_id=assessment.detection_id)
                            else:
                                self._prov(session.session_id, _PET.FACE_MATCH_MISS, stage, {
                                    "consent_status": consent.value if hasattr(consent, "value") else str(consent) if consent else None,
                                }, detection_id=assessment.detection_id)
                            if consent is not None:
                                self._prov(session.session_id, _PET.CONSENT_APPLIED, stage, {
                                    "consent_status": consent.value if hasattr(consent, "value") else str(consent),
                                    "person_id": person_id,
                                }, detection_id=assessment.detection_id)
                except Exception as exc:
                    # Non-fatal — log and continue
                    logger.warning(
                        "Consent stage error (non-fatal): %s", exc
                    )
                    elapsed = elapsed_ms(t0)
                    session.stage_timings["consent_identity_ms"] = elapsed
                    emit("stage_complete", {
                        "stage": stage,
                        "summary": {"status": "skipped", "reason": str(exc)},
                        "elapsed_ms": elapsed,
                    })
                    if _PET is not None:
                        self._prov(session.session_id, _PET.STAGE_ERROR, stage, {
                            "status": "skipped",
                            "reason": str(exc),
                        })
            # Stage: strategy
            if "strategy" in run_stages:
                stage = "strategy"
                meta = _STAGE_META[stage]
                emit("stage_start", {
                    "stage": stage,
                    "display_name": meta["display_name"],
                    "description": meta["description"],
                })
                if _PET is not None:
                    self._prov(session.session_id, _PET.STAGE_START, stage, {
                        "stage": stage,
                        "fallback_only": fallback_only,
                    })
                t0 = time.perf_counter()
                session.current_stage = stage
                try:
                    if fallback_only:
                        strategy_result = orc.strategy_agent.run(risk_result, image_path)
                    else:
                        strategy_result = orc.strategy_agent.run(
                            risk_result, image_path, annotated_image
                        )
                    session.strategy_result = strategy_result
                    elapsed = elapsed_ms(t0)
                    session.stage_timings["strategy_ms"] = elapsed
                    emit("stage_complete", {
                        "stage": stage,
                        "summary": {
                            "strategies": len(strategy_result.strategies),
                            "protections_recommended": strategy_result.total_protections_recommended,
                        },
                        "elapsed_ms": elapsed,
                    })
                    if _PET is not None:
                        phase_type = _PET.STRATEGY_ASSIGNED_P1 if fallback_only else _PET.STRATEGY_ASSIGNED_P2
                        self._prov(session.session_id, _PET.STAGE_COMPLETE, stage, {
                            "strategies": len(strategy_result.strategies),
                            "protections_recommended": strategy_result.total_protections_recommended,
                            "elapsed_ms": elapsed,
                        })
                        for strat in strategy_result.strategies:
                            method = strat.recommended_method
                            self._prov(session.session_id, phase_type, stage, {
                                "element": strat.element,
                                "severity": strat.severity.value if hasattr(strat.severity, "value") else str(strat.severity),
                                "method": method.value if hasattr(method, "value") else str(method) if method else "none",
                                "requires_user_decision": strat.requires_user_decision,
                            }, detection_id=strat.detection_id)
                except Exception as exc:
                    if _PET is not None:
                        self._prov(session.session_id, _PET.STAGE_ERROR, stage, {"error": str(exc)})
                    self._handle_stage_error(session, stage, exc, emit)
                    return

                # HITL gate: strategy_review
                if self._should_pause_hitl(session, stage):
                    self._block_for_hitl(session, "strategy_review", emit)
                    if session.status == "failed":
                        return
            else:
                strategy_result = session.strategy_result
            # Stage: sam
            if "sam" in run_stages:
                stage = "sam"
                meta = _STAGE_META[stage]
                emit("stage_start", {
                    "stage": stage,
                    "display_name": meta["display_name"],
                    "description": meta["description"],
                })
                if _PET is not None:
                    self._prov(session.session_id, _PET.STAGE_START, stage, {"stage": stage})
                t0 = time.perf_counter()
                session.current_stage = stage
                seg_results: Dict[str, Any] = {}
                run_sam = (
                    orc.segmenter is not None
                    and not fallback_only
                )
                try:
                    if run_sam:
                        output_dir = str(orc.output_dir)
                        seg_results = orc.segmenter.process_strategies(
                            image_path,
                            strategy_result.strategies,
                            risk_result.risk_assessments,
                            output_dir=output_dir,
                        )
                        for strategy in strategy_result.strategies:
                            if strategy.detection_id in seg_results:
                                mask_data = seg_results[strategy.detection_id]
                                strategy.segmentation_mask_path = mask_data.get(
                                    "mask_path"
                                )
                    elapsed = elapsed_ms(t0)
                    session.stage_timings["sam_segmentation_ms"] = elapsed
                    emit("stage_complete", {
                        "stage": stage,
                        "summary": {
                            "masks_generated": len(seg_results),
                            "skipped": not run_sam,
                        },
                        "elapsed_ms": elapsed,
                    })
                    if _PET is not None:
                        self._prov(session.session_id, _PET.STAGE_COMPLETE, stage, {
                            "masks_generated": len(seg_results),
                            "skipped": not run_sam,
                            "elapsed_ms": elapsed,
                        })
                        if run_sam:
                            for strat in strategy_result.strategies:
                                if strat.detection_id in seg_results:
                                    self._prov(session.session_id, _PET.SAM_MASK_GENERATED, stage, {
                                        "mask_path": seg_results[strat.detection_id].get("mask_path"),
                                    }, detection_id=strat.detection_id)
                                else:
                                    self._prov(session.session_id, _PET.SAM_SKIPPED, stage, {
                                        "reason": "not in seg_results",
                                    }, detection_id=strat.detection_id)
                        else:
                            for strat in strategy_result.strategies:
                                self._prov(session.session_id, _PET.SAM_SKIPPED, stage, {
                                    "reason": "fallback_only or no segmenter",
                                }, detection_id=strat.detection_id)
                except Exception as exc:
                    # SAM is non-fatal
                    logger.warning("SAM stage error (non-fatal): %s", exc)
                    elapsed = elapsed_ms(t0)
                    session.stage_timings["sam_segmentation_ms"] = elapsed
                    emit("stage_complete", {
                        "stage": stage,
                        "summary": {
                            "masks_generated": 0,
                            "skipped": True,
                            "reason": str(exc),
                        },
                        "elapsed_ms": elapsed,
                    })
                    if _PET is not None:
                        self._prov(session.session_id, _PET.STAGE_ERROR, stage, {
                            "skipped": True,
                            "reason": str(exc),
                        })
            # Stage: execution
            if "execution" in run_stages:
                stage = "execution"
                meta = _STAGE_META[stage]
                emit("stage_start", {
                    "stage": stage,
                    "display_name": meta["display_name"],
                    "description": meta["description"],
                })
                if _PET is not None:
                    self._prov(session.session_id, _PET.STAGE_START, stage, {"stage": stage})
                t0 = time.perf_counter()
                session.current_stage = stage
                try:
                    stem = Path(image_path).stem
                    protected_output = orc.output_dir / f"{stem}_protected.png"
                    execution_report = orc.execution_agent.run(
                        strategy_result=strategy_result,
                        risk_result=risk_result,
                        image_path=image_path,
                        output_path=str(protected_output),
                    )
                    session.execution_report = execution_report
                    session.protected_image_path = str(protected_output)
                    elapsed = elapsed_ms(t0)
                    session.stage_timings["execution_ms"] = elapsed
                    emit("stage_complete", {
                        "stage": stage,
                        "summary": {
                            "transformations_applied": len(
                                execution_report.transformations_applied
                            ),
                            "status": execution_report.status,
                        },
                        "elapsed_ms": elapsed,
                    })
                    if _PET is not None:
                        self._prov(session.session_id, _PET.STAGE_COMPLETE, stage, {
                            "transformations_applied": len(execution_report.transformations_applied),
                            "status": execution_report.status,
                            "elapsed_ms": elapsed,
                        })
                        for xform in execution_report.transformations_applied:
                            xstatus = getattr(xform, "status", "unknown")
                            method = xform.method
                            method_str = method.value if hasattr(method, "value") else str(method)
                            if xstatus == "success":
                                self._prov(session.session_id, _PET.OBFUSCATION_APPLIED, stage, {
                                    "element": xform.element,
                                    "method": method_str,
                                    "execution_time_ms": getattr(xform, "execution_time_ms", None),
                                }, detection_id=xform.detection_id)
                            else:
                                self._prov(session.session_id, _PET.OBFUSCATION_FAILED, stage, {
                                    "element": xform.element,
                                    "method": method_str,
                                    "error_message": getattr(xform, "error_message", None),
                                }, detection_id=xform.detection_id)
                except Exception as exc:
                    if _PET is not None:
                        self._prov(session.session_id, _PET.STAGE_ERROR, stage, {"error": str(exc)})
                    self._handle_stage_error(session, stage, exc, emit)
                    return

                # HITL gate: execution_verify
                if self._should_pause_hitl(session, stage):
                    self._block_for_hitl(session, "execution_verify", emit)
                    if session.status == "failed":
                        return
            else:
                execution_report = session.execution_report
            # Stage: export
            if "export" in run_stages:
                stage = "export"
                meta = _STAGE_META[stage]
                emit("stage_start", {
                    "stage": stage,
                    "display_name": meta["display_name"],
                    "description": meta["description"],
                })
                if _PET is not None:
                    self._prov(session.session_id, _PET.STAGE_START, stage, {"stage": stage})
                t0 = time.perf_counter()
                session.current_stage = stage
                try:
                    from utils.visualization import (
                        export_risk_results_json,
                        export_strategy_results_json,
                        generate_risk_map,
                    )

                    stem = Path(image_path).stem

                    json_path = orc.output_dir / f"{stem}_risk_results.json"
                    export_risk_results_json(
                        risk_result,
                        detections=detections,
                        output_path=str(json_path),
                    )

                    risk_map_path = orc.output_dir / f"{stem}_risk_map.png"
                    generate_risk_map(
                        risk_result,
                        image_path,
                        output_path=str(risk_map_path),
                    )
                    session.risk_map_path = str(risk_map_path)

                    strategy_json_path = orc.output_dir / f"{stem}_strategies.json"
                    export_strategy_results_json(
                        strategy_result,
                        output_path=str(strategy_json_path),
                    )

                    elapsed = elapsed_ms(t0)
                    session.stage_timings["export_ms"] = elapsed
                    emit("stage_complete", {
                        "stage": stage,
                        "summary": {"files_written": 3},
                        "elapsed_ms": elapsed,
                    })
                    if _PET is not None:
                        self._prov(session.session_id, _PET.STAGE_COMPLETE, stage, {
                            "files_written": 3,
                            "elapsed_ms": elapsed,
                        })
                except Exception as exc:
                    # Export failure is non-fatal
                    logger.warning("Export stage error (non-fatal): %s", exc)
                    elapsed = elapsed_ms(t0)
                    session.stage_timings["export_ms"] = elapsed
                    emit("stage_complete", {
                        "stage": stage,
                        "summary": {
                            "files_written": 0,
                            "reason": str(exc),
                        },
                        "elapsed_ms": elapsed,
                    })
                    if _PET is not None:
                        self._prov(session.session_id, _PET.STAGE_ERROR, stage, {
                            "files_written": 0,
                            "reason": str(exc),
                        })
            # Pipeline complete
            total_ms = elapsed_ms(pipeline_start)
            session.stage_timings["total_ms"] = total_ms
            session.status = "completed"

            protections_applied = 0
            if session.execution_report is not None:
                protections_applied = len(
                    session.execution_report.transformations_applied
                )

            results_url = f"/api/v1/pipeline/{session.session_id}/results"
            emit("pipeline_complete", {
                "stage": "done",
                "total_time_ms": total_ms,
                "protections_applied": protections_applied,
                "results_url": results_url,
            })

            # Finalize the provenance session (writes JSON + MongoDB summary)
            if self._provenance is not None:
                try:
                    _results_dir = str(getattr(orc, "output_dir", None) or "")
                    self._provenance.finalize_session(
                        session_id=session.session_id,
                        phases_completed=run_stages,
                        total_time_ms=total_ms,
                        protections_applied=protections_applied,
                        protected_image_path=session.protected_image_path,
                        results_dir=_results_dir or None,
                    )
                except Exception as _fin_exc:
                    logger.debug("Provenance finalize_session skipped: %s", _fin_exc)
            logger.info(
                "Pipeline completed: session_id=%s  total_ms=%.0f",
                session.session_id,
                total_ms,
            )

        except PipelineServiceError:
            # Already handled by _handle_stage_error
            pass
        except Exception as exc:
            logger.error(
                "Unexpected pipeline error for session %s: %s",
                session.session_id,
                exc,
            )
            traceback.print_exc()
            session.status = "failed"
            session.error_code = "PIPELINE_ERROR"
            session.error_message = str(exc)
            emit("pipeline_error", {
                "stage": session.current_stage,
                "error_code": "PIPELINE_ERROR",
                "recoverable": False,
                "suggestion": "Check server logs for details.",
            })
    # HITL helpers

    def _should_pause_hitl(
        self, session: SessionRecord, stage: str
    ) -> bool:
        """
        Determine whether the pipeline should pause at this stage's HITL gate.

        Reads the session config to check if HITL is enabled for the stage.
        """
        if stage not in _HITL_GATES:
            return False
        cfg = session.config or {}
        hitl_cfg = cfg.get("hitl", {})

        # Canonical mode values: auto (no HITL), hybrid (confidence-gated), manual (always pause)
        mode = cfg.get("mode", "auto")

        # auto mode: never pause at HITL gates
        if mode == "auto":
            return False

        # pause_on_critical: pause at risk_review if there are critical items (hybrid + manual)
        if stage == "risk" and hitl_cfg.get("pause_on_critical", False):
            if session.risk_result is not None:
                try:
                    critical = session.risk_result.get_critical_risks()
                    return bool(critical)
                except Exception:
                    pass

        # manual mode always pauses at every HITL gate
        if mode == "manual":
            return True

        # hybrid mode: pause on critical text/faces (confidence-gated)
        if mode == "hybrid" and stage in ("strategy", "execution"):
            return hitl_cfg.get("pause_on_critical", False)

        return False

    def _block_for_hitl(
        self,
        session: SessionRecord,
        checkpoint: str,
        emit,
    ) -> None:
        """
        Pause the worker thread at a HITL checkpoint.

        Emits hitl_checkpoint WS event, then blocks on threading.Event until
        the async event loop calls approve_checkpoint() (from POST /approve).

        The session status is set to hitl_paused while blocked, then restored
        to running when approval arrives.
        """
        session.status = "hitl_paused"
        session.hitl_checkpoint = checkpoint
        session.hitl_pending_approval = True
        session.hitl_event.clear()

        # Record the HITL pause in provenance
        try:
            from utils.models import ProvenanceEventType as _PET2
            self._prov(session.session_id, _PET2.HITL_PAUSED, "hitl", {
                "checkpoint": checkpoint,
            })
        except Exception:
            pass

        # Determine elements requiring review
        elements_requiring_review: List[str] = []
        if checkpoint == "risk_review" and session.risk_result:
            elements_requiring_review = [
                a.detection_id
                for a in session.risk_result.risk_assessments
                if a.severity.value in ("critical", "high")
            ]
        elif checkpoint == "strategy_review" and session.strategy_result:
            elements_requiring_review = [
                s.detection_id
                for s in session.strategy_result.strategies
                if s.requires_user_decision
            ]

        actions_available = ["approve", "override"]

        emit("hitl_checkpoint", {
            "checkpoint": checkpoint,
            "reason": f"Manual review required at {checkpoint}",
            "elements_requiring_review": elements_requiring_review,
            "actions_available": actions_available,
        })

        logger.info(
            "HITL pause: session_id=%s  checkpoint=%s",
            session.session_id,
            checkpoint,
        )

        # Block until approved (or a 30-minute timeout as a safety net)
        approved = session.hitl_event.wait(timeout=1800)
        if not approved:
            logger.error(
                "HITL timeout for session %s at checkpoint %s",
                session.session_id,
                checkpoint,
            )
            session.status = "failed"
            session.error_code = "HITL_TIMEOUT"
            session.error_message = f"No approval received within 30 minutes at {checkpoint}"
            emit("pipeline_error", {
                "stage": session.current_stage,
                "error_code": "HITL_TIMEOUT",
                "recoverable": True,
                "suggestion": "Call /approve or /rerun to continue.",
            })
            return

        # Approval received — resume
        session.status = "running"
        session.hitl_checkpoint = None
        session.hitl_pending_approval = False

        try:
            from utils.models import ProvenanceEventType as _PET3
            self._prov(session.session_id, _PET3.HITL_APPROVED, "hitl", {
                "checkpoint": checkpoint,
                "resolved_by": "user_approval",
            })
        except Exception:
            pass

        emit("pipeline_resumed", {
            "checkpoint": checkpoint,
            "resolved_by": "user_approval",
            "next_stage": session.current_stage,
        })
    # Error handling

    def _handle_stage_error(
        self,
        session: SessionRecord,
        stage: str,
        exc: Exception,
        emit,
    ) -> None:
        """Mark session as failed and emit pipeline_error WS event."""
        logger.error(
            "Pipeline stage failed: session=%s  stage=%s  error=%s",
            session.session_id,
            stage,
            exc,
        )
        traceback.print_exc()
        session.status = "failed"
        session.error_code = "PIPELINE_ERROR"
        session.error_message = str(exc)
        session.error_stage = stage
        emit("pipeline_error", {
            "stage": stage,
            "error_code": "PIPELINE_ERROR",
            "recoverable": False,
            "suggestion": "Check server logs. You may rerun from an earlier stage.",
        })
        raise PipelineServiceError(stage, exc)
    # Results accessor

    def get_results(self, session: SessionRecord) -> Dict[str, Any]:
        """
        Build a structured results dict from cached session stage outputs.

        Intended for GET /pipeline/{session_id}/results.

        Returns:
            Dict with keys: detections, risk_result, strategy_result,
            execution_report, audit_trail, stage_timings.
        """
        return {
            "session_id": session.session_id,
            "status": session.status,
            "detections": session.detections,
            "risk_result": session.risk_result,
            "strategy_result": session.strategy_result,
            "execution_report": session.execution_report,
            "protected_image_path": session.protected_image_path,
            "risk_map_path": session.risk_map_path,
            "audit_trail": session.audit_trail,
            "stage_timings": session.stage_timings,
        }
# Service-level exceptions


class PipelineServiceError(Exception):
    """Internal signal to break out of the worker after a handled error."""

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


class PipelineRunningError(Exception):
    """Raised when a pipeline start is requested but one is already active."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Pipeline already running for session {session_id}")
        self.session_id = session_id


class InvalidStageError(Exception):
    """Raised for an unknown from_stage in rerun_from."""

    def __init__(self, stage: str) -> None:
        super().__init__(f"Unknown stage: {stage!r}.  Valid: {VALID_STAGES}")
        self.stage = stage


class CheckpointMissingError(Exception):
    """Raised when rerun_from is called but required upstream cache is absent."""

    def __init__(self, stage: str, required_attr: str) -> None:
        super().__init__(
            f"Cannot rerun from '{stage}': cached '{required_attr}' is missing.  "
            "Run the pipeline from the beginning first."
        )
        self.stage = stage
        self.required_attr = required_attr


class CheckpointMismatchError(Exception):
    """Raised when the approved checkpoint name does not match the paused one."""

    def __init__(self, expected: str, got: str) -> None:
        super().__init__(
            f"Checkpoint mismatch: session is paused at '{expected}', "
            f"but approval was sent for '{got}'"
        )
        self.expected = expected
        self.got = got
