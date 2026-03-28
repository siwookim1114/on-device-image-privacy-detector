"""
HITL Interactive Demo Server
============================
Self-contained FastAPI server that serves a full interactive demo page for
testing the Coordinator Agent with real-time visual feedback.

Default mode: FULL pipeline with VLM (Qwen3-VL-30B).
Requires llama-server to be running on port 8081.

Usage:
    # Full pipeline (requires llama-server):
    conda run -n lab_env python tests/hitl_demo_server.py

    # Phase 1 only (no VLM needed):
    conda run -n lab_env python tests/hitl_demo_server.py --fallback-only

    # Custom port:
    conda run -n lab_env python tests/hitl_demo_server.py --port 9000

Then open http://localhost:8888

Startup workflow:
    Terminal 1: bash start_llama_server.sh
    Terminal 2: conda run -n lab_env python tests/hitl_demo_server.py
    Browser:    http://localhost:8888
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path bootstrap — must happen before any project imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn

from agents.pipeline import PipelineOrchestrator, PipelineConfig
from agents.coordinator.main import CoordinatorSession
from agents.coordinator.adaptive_learning import PreferenceManager
from utils.models import PrivacyProfile, PersonEntry, FaceEmbedding, ConsentHistory
from utils.storage import FaceDatabase

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hitl_demo")

# ---------------------------------------------------------------------------
# CLI args — parsed early so startup banner is accurate
# ---------------------------------------------------------------------------
import argparse as _argparse

_parser = _argparse.ArgumentParser(description="HITL Interactive Demo Server")
_parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
_parser.add_argument("--port", type=int, default=8888, help="Port (default: 8888)")
_parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")
_parser.add_argument(
    "--fallback-only",
    action="store_true",
    dest="fallback_only",
    help="Skip VLM (Phase 1 deterministic only). No llama-server required.",
)
_ARGS, _ = _parser.parse_known_args()

# ---------------------------------------------------------------------------
# VLM health check
# ---------------------------------------------------------------------------
_VLM_URL = "http://localhost:8081"
_vlm_available: bool = False


def check_vlm_health() -> bool:
    """Return True if llama-server is reachable on port 8081."""
    try:
        import httpx
        r = httpx.get(f"{_VLM_URL}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def _refresh_vlm_status() -> bool:
    """Re-check VLM and update the module-level flag."""
    global _vlm_available
    _vlm_available = check_vlm_health()
    return _vlm_available


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="HITL Interactive Demo", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global session state (single-user demo — no auth needed)
# ---------------------------------------------------------------------------
state: Dict[str, Any] = {
    "session_id": None,
    "image_path": None,           # str path to saved upload
    "original_bytes": None,       # bytes (PNG)
    "protected_bytes": None,      # bytes (PNG) — refreshed after each chat
    "orchestrator": None,         # PipelineOrchestrator
    "coordinator": None,          # CoordinatorSession
    "pipeline_output": None,      # PipelineOutput
    "logs": [],                   # List[dict] — interaction log
    "summary": {},                # dict — pipeline summary stats
    "vlm_changes": [],            # List[dict] — Phase 1 vs Phase 2 deltas
}

# Keep PipelineOrchestrator alive across requests (model weights loaded once)
_shared_orchestrator: Optional[PipelineOrchestrator] = None

_UPLOAD_DIR = _PROJECT_ROOT / "data" / "tmp_uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FaceDatabase singleton — optional, requires MongoDB
# ---------------------------------------------------------------------------
_face_db: Optional[FaceDatabase] = None


def _get_face_db() -> Optional[FaceDatabase]:
    """Return a cached FaceDatabase connection, or None if MongoDB is unavailable."""
    global _face_db
    if _face_db is None:
        try:
            _face_db = FaceDatabase(
                mongo_uri="mongodb://localhost:27017/",
                database_name="privacy_guard",
                encryption_key_path=str(_PROJECT_ROOT / "data" / "face_db" / ".encryption_key"),
                encryption_enabled=True,
            )
        except Exception as exc:
            logger.warning("FaceDatabase unavailable (MongoDB down?): %s", exc)
            _face_db = None
    return _face_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_create_orchestrator() -> PipelineOrchestrator:
    """
    Return a cached PipelineOrchestrator.  If the VLM is unavailable and we
    are NOT in --fallback-only mode, log a warning but still create the
    orchestrator with fallback_only=False — individual agents will hit
    connection errors gracefully and return Phase 1 results.
    """
    global _shared_orchestrator
    if _shared_orchestrator is None:
        use_fallback = _ARGS.fallback_only
        if not use_fallback and not _vlm_available:
            logger.warning(
                "VLM not reachable at %s — starting orchestrator in full-pipeline "
                "mode anyway; VLM phases will degrade gracefully.", _VLM_URL
            )
        mode_label = "fallback_only=True (Phase 1 only)" if use_fallback else "full pipeline (VLM enabled)"
        logger.info("Initialising PipelineOrchestrator (%s)...", mode_label)
        consent_available = _get_face_db() is not None
        if not consent_available:
            logger.warning("MongoDB unavailable — consent identity (Agent 2.5) disabled.")
        cfg = PipelineConfig(
            fallback_only=use_fallback,
            enable_consent=consent_available,
            enable_sam=not use_fallback,
        )
        _shared_orchestrator = PipelineOrchestrator(config=cfg)
    return _shared_orchestrator


def _read_image_bytes(path: str) -> Optional[bytes]:
    """Read image at *path* and return as PNG bytes, or None on failure."""
    try:
        from PIL import Image as PILImage
        img = PILImage.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        logger.warning("Could not read image %s: %s", path, exc)
        return None


def _compute_latency_breakdown(phase_timings: Dict[str, float]) -> Dict[str, Any]:
    """
    Split phase_timings into Phase 1 (deterministic) vs Phase 2 (VLM) buckets.

    Phase 1 keys:  detection_ms, consent_identity_ms, sam_segmentation_ms, export_ms
    Phase 2 keys:  risk_assessment_ms (includes both P1+P2), strategy_ms,
                   execution_ms (includes both P1+P2)

    We cannot cleanly separate risk/strategy/execution P1 from P2 inside the
    existing timing keys without modifying pipeline.py, so we report the
    composite per-stage values and annotate which stages include VLM.
    """
    t = phase_timings

    # Pure Phase 1 stages (no VLM calls)
    detection_ms     = t.get("detection_ms", 0.0)
    consent_ms       = t.get("consent_identity_ms", 0.0)
    sam_ms           = t.get("sam_segmentation_ms", 0.0)
    export_ms        = t.get("export_ms", 0.0)

    # Mixed stages (deterministic + optional VLM)
    risk_ms          = t.get("risk_assessment_ms", 0.0)
    strategy_ms      = t.get("strategy_ms", 0.0)
    execution_ms     = t.get("execution_ms", 0.0)
    total_ms         = t.get("total_ms", 0.0)

    phase1_pure_ms   = detection_ms + consent_ms + sam_ms + export_ms
    vlm_stage_ms     = risk_ms + strategy_ms + execution_ms
    other_ms         = max(0.0, total_ms - phase1_pure_ms - vlm_stage_ms)

    return {
        "total_ms": total_ms,
        "phase1_pure_ms": phase1_pure_ms,
        "vlm_stages_ms": vlm_stage_ms,
        "per_stage": {
            "detection_ms": detection_ms,
            "risk_assessment_ms": risk_ms,
            "consent_identity_ms": consent_ms,
            "strategy_ms": strategy_ms,
            "sam_segmentation_ms": sam_ms,
            "execution_ms": execution_ms,
            "export_ms": export_ms,
        },
        "other_ms": other_ms,
    }


def _extract_vlm_changes(pipeline_output) -> List[Dict[str, Any]]:
    """
    Examine risk assessments and strategies for Phase 1 vs Phase 2 deltas.
    Returns a list of human-readable change dicts.
    """
    changes: List[Dict[str, Any]] = []
    if pipeline_output is None:
        return changes

    # Risk severity changes
    risk = pipeline_output.risk_analysis
    if risk is not None:
        for a in risk.risk_assessments:
            a_dict = a.model_dump() if hasattr(a, "model_dump") else {}
            if not a_dict.get("vlm_phase2_ran", False):
                continue
            p1 = a_dict.get("original_severity") or a_dict.get("phase1_severity", "")
            p2 = a_dict.get("severity", "")
            if p1 and p2 and p1.lower() != p2.lower():
                did = a_dict.get("detection_id", "unknown")
                changes.append({
                    "stage": "risk",
                    "detection_id": did,
                    "change": f"VLM upgraded {did}: {p1.upper()} → {p2.upper()}",
                    "phase1": p1.upper(),
                    "phase2": p2.upper(),
                    "direction": "upgrade" if _severity_rank(p2) > _severity_rank(p1) else "downgrade",
                })

    # Strategy method changes — check execution_report transformations for original_method
    report = pipeline_output.execution_report
    if report is not None:
        for t_applied in report.transformations_applied:
            t_dict = t_applied.model_dump() if hasattr(t_applied, "model_dump") else {}
            original = t_dict.get("original_method", "")
            current  = t_dict.get("method", "") or (t_applied.method.value if hasattr(t_applied.method, "value") else "")
            if original and current and original != current:
                did = t_dict.get("detection_id", "unknown")
                changes.append({
                    "stage": "strategy",
                    "detection_id": did,
                    "change": f"VLM changed {did}: method {original} → {current}",
                    "phase1": original,
                    "phase2": current,
                    "direction": "modified",
                })

    return changes


def _severity_rank(sev: str) -> int:
    return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(sev.lower(), 0)


def _build_summary(pipeline_output) -> Dict[str, Any]:
    """Extract a concise summary dict from a PipelineOutput."""
    summary: Dict[str, Any] = {
        "success": False,
        "total_elements": 0,
        "protected_count": 0,
        "overall_risk": "unknown",
        "methods": [],
        "phase_timings": {},
        "latency": {},
        "vlm_enabled": not _ARGS.fallback_only,
        "vlm_available": _vlm_available,
        "error": None,
    }
    if pipeline_output is None:
        return summary

    summary["success"] = pipeline_output.success
    raw_timings = {k: round(v, 1) for k, v in (pipeline_output.phase_timings or {}).items()}
    summary["phase_timings"] = raw_timings
    summary["latency"] = {
        k: round(v, 1) for k, v in _compute_latency_breakdown(raw_timings).items()
        if not isinstance(v, dict)
    }
    summary["latency"]["per_stage"] = {
        k: round(v, 1)
        for k, v in _compute_latency_breakdown(raw_timings).get("per_stage", {}).items()
    }
    summary["error"] = pipeline_output.error_message

    risk = pipeline_output.risk_analysis
    if risk is not None:
        summary["total_elements"] = len(risk.risk_assessments)
        summary["overall_risk"] = risk.overall_risk_level.value.upper()
        summary["protected_count"] = sum(
            1 for a in risk.risk_assessments if a.requires_protection
        )

    report = pipeline_output.execution_report
    if report is not None:
        methods_used: Dict[str, int] = {}
        for t in report.transformations_applied:
            m = t.method.value if hasattr(t.method, "value") else str(t.method)
            methods_used[m] = methods_used.get(m, 0) + 1
        summary["methods"] = [
            {"method": m, "count": c} for m, c in methods_used.items()
        ]

    # Consent identity matches — collected from face risk assessments
    consent_info: List[Dict[str, Any]] = []
    if pipeline_output.risk_analysis is not None:
        for a in pipeline_output.risk_analysis.risk_assessments:
            if getattr(a, "element_type", None) == "face":
                consent = getattr(a, "consent_status", None)
                label   = getattr(a, "person_label", None)
                if consent is not None:
                    consent_val = consent.value if hasattr(consent, "value") else str(consent)
                    consent_info.append({
                        "detection_id":  a.detection_id,
                        "consent":       consent_val,
                        "person_label":  label,
                        "classification": getattr(a, "classification", None),
                    })
    summary["consent_matches"] = consent_info

    return summary


def _append_log(entry: Dict[str, Any]) -> None:
    state["logs"].append(entry)
    if len(state["logs"]) > 200:
        state["logs"] = state["logs"][-200:]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api/vlm-status")
async def vlm_status() -> JSONResponse:
    """Live VLM health probe — called by the UI on page load and periodically."""
    available = _refresh_vlm_status()
    return JSONResponse({
        "available": available,
        "url": _VLM_URL,
        "fallback_only": _ARGS.fallback_only,
    })


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accept an uploaded image, run the full pipeline (VLM enabled by default),
    and initialise a CoordinatorSession for subsequent chat commands.
    """
    t_start = time.time()

    # Probe VLM before processing so the response carries current status
    _refresh_vlm_status()

    # --- Save uploaded file ---
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    session_id = uuid.uuid4().hex
    save_path = _UPLOAD_DIR / f"{session_id}{suffix}"

    raw = await file.read()
    save_path.write_bytes(raw)
    logger.info("Saved upload: %s (%d bytes)", save_path, len(raw))

    # --- Cache original PNG bytes ---
    orig_bytes = _read_image_bytes(str(save_path))

    # --- Run pipeline ---
    orchestrator = _get_or_create_orchestrator()
    pipeline_output = None
    error_msg = None
    vlm_degraded = False
    try:
        pipeline_output = orchestrator.run(str(save_path))
    except Exception as exc:
        error_msg = str(exc)
        logger.error("Pipeline run failed: %s", exc)
        traceback.print_exc()
        # Check whether this looks like a VLM connectivity failure
        if "connect" in error_msg.lower() or "8081" in error_msg or "refused" in error_msg.lower():
            vlm_degraded = True
            error_msg = "VLM unavailable — showing Phase 1 results only. " + error_msg

    # --- Read protected image bytes ---
    protected_bytes: Optional[bytes] = None
    if pipeline_output and pipeline_output.protected_image_path:
        protected_bytes = _read_image_bytes(pipeline_output.protected_image_path)

    # Fallback: if execution produced no protected image, show original
    if protected_bytes is None:
        protected_bytes = orig_bytes

    # --- Extract Phase 1 vs Phase 2 changes ---
    vlm_changes = _extract_vlm_changes(pipeline_output)

    # --- Build CoordinatorSession with real Safety Kernel + agents ---
    from agents.coordinator.nodes import NodeContext
    from backend.services.safety_kernel import SafetyKernel

    orc = _shared_orchestrator
    ctx = NodeContext()
    ctx.fallback_only = _ARGS.fallback_only
    ctx.safety_kernel = SafetyKernel()
    ctx.output_dir = str(Path("data/full_pipeline_results"))
    if orc is not None:
        ctx.detection_agent = getattr(orc, "detection_agent", None)
        ctx.risk_agent = getattr(orc, "risk_agent", None)
        ctx.consent_agent = getattr(orc, "consent_agent", None)
        ctx.strategy_agent = getattr(orc, "strategy_agent", None)
        ctx.execution_agent = getattr(orc, "execution_agent", None)

    # Initialize small LLM for intent classification + response generation
    try:
        from agents.text_llm import TextLLM
        _intent_model = _PROJECT_ROOT / "backend-engines" / "models" / "Qwen3-1.7B-Q4_K_M.gguf"
        if _intent_model.exists():
            ctx.text_llm = TextLLM(str(_intent_model))
            logger.info("TextLLM loaded: %s", _intent_model.name)
        else:
            logger.info("Intent model not found at %s — using regex-only classification", _intent_model)
    except ImportError:
        logger.info("llama-cpp-python not installed — using regex-only classification")
    except Exception as exc:
        logger.warning("TextLLM init failed: %s — using regex-only classification", exc)

    pref_manager = PreferenceManager()
    coordinator = CoordinatorSession(
        session_id=session_id,
        ctx=ctx,
        image_path=str(save_path),
        fallback_only=_ARGS.fallback_only,
        preference_manager=pref_manager,
    )
    # Inject pipeline results into coordinator state so HITL commands work.
    # IMPORTANT: get_pipeline_state() returns a SHALLOW COPY — writes to it are
    # discarded.  Access coordinator._state["pipeline_state"] directly so that
    # the injected values survive into the next chat turn.
    if pipeline_output is not None:
        ps = coordinator._state.get("pipeline_state")
        if ps is not None:
            ps["image_path"] = str(save_path)
            if pipeline_output.risk_analysis is not None:
                ps["risk_result"] = pipeline_output.risk_analysis
            if pipeline_output.execution_report is not None:
                ps["execution_report"] = pipeline_output.execution_report
            # Inject strategy_result directly from pipeline output
            if pipeline_output.strategy_recommendations is not None:
                ps["strategy_result"] = pipeline_output.strategy_recommendations
            else:
                # Fallback: load from saved JSON
                strategy_json = _PROJECT_ROOT / "data" / "full_pipeline_results" / f"{Path(save_path).stem}_strategies.json"
                if strategy_json.exists():
                    try:
                        import json as _json
                        with open(strategy_json) as _f:
                            strat_data = _json.load(_f)
                        from utils.models import StrategyRecommendations
                        ps["strategy_result"] = StrategyRecommendations(**strat_data)
                    except Exception as _e:
                        logger.warning("Could not load strategies: %s", _e)
            # Note: pipeline_output does not carry raw detections (they stay
            # inside pipeline.py).  The coordinator can still resolve elements
            # via strategy_result and risk_result which are injected above.

    # --- Persist state ---
    state["session_id"] = session_id
    state["image_path"] = str(save_path)
    state["original_bytes"] = orig_bytes
    state["protected_bytes"] = protected_bytes
    state["orchestrator"] = orchestrator
    state["coordinator"] = coordinator
    state["pipeline_output"] = pipeline_output
    state["summary"] = _build_summary(pipeline_output)
    state["vlm_changes"] = vlm_changes
    state["logs"] = []

    elapsed_ms = (time.time() - t_start) * 1000
    summary = state["summary"]

    # Build latency string for the log entry
    latency = summary.get("latency", {})
    total_s = round(latency.get("total_ms", elapsed_ms) / 1000, 1)
    p1_s    = round(latency.get("phase1_pure_ms", 0) / 1000, 1)
    vlm_s   = round(latency.get("vlm_stages_ms", 0) / 1000, 1)
    timing_detail = f"{total_s}s total (Phase 1: {p1_s}s, VLM stages: {vlm_s}s)"

    log_entry = {
        "ts": time.time(),
        "type": "pipeline",
        "user": f"[upload] {file.filename}",
        "intent": "process",
        "response": (
            f"Pipeline complete. {summary['total_elements']} elements detected, "
            f"{summary['protected_count']} protected. "
            f"Overall risk: {summary['overall_risk']}. "
            f"Timing: {timing_detail}."
        ) if not error_msg else f"Pipeline error: {error_msg}",
        "action": "pipeline_run",
        "timing_ms": round(elapsed_ms, 1),
        "image_updated": True,
        "vlm_changes": vlm_changes,
        "vlm_degraded": vlm_degraded,
    }
    _append_log(log_entry)

    return JSONResponse({
        "success": pipeline_output.success if pipeline_output else False,
        "session_id": session_id,
        "summary": summary,
        "vlm_changes": vlm_changes,
        "vlm_available": _vlm_available,
        "vlm_degraded": vlm_degraded,
        "error": error_msg,
        "timing_ms": round(elapsed_ms, 1),
    })


@app.post("/api/chat")
async def chat(body: dict) -> JSONResponse:
    """
    Process one NL command through the CoordinatorSession.
    If the pipeline re-ran, refreshes the protected image from disk.
    """
    if state["coordinator"] is None:
        return JSONResponse(
            {"error": "No active session. Please upload an image first."},
            status_code=400,
        )

    message: str = (body.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "Empty message."}, status_code=400)

    t_start = time.time()
    coordinator: CoordinatorSession = state["coordinator"]
    result: Dict[str, Any] = {}

    try:
        result = coordinator.handle_message(message)
    except Exception as exc:
        logger.error("coordinator.handle_message failed: %s", exc)
        traceback.print_exc()
        err_str = str(exc)
        vlm_note = ""
        if "connect" in err_str.lower() or "8081" in err_str or "refused" in err_str.lower():
            vlm_note = " (VLM connection lost — results may reflect Phase 1 only)"
        result = {
            "intent": {"action": "error", "confidence": 0.0, "natural_language": message},
            "response_text": f"Error: {exc}{vlm_note}",
            "pipeline_action_taken": "error",
            "suggestions": [],
        }

    elapsed_ms = (time.time() - t_start) * 1000

    # --- Check whether the pipeline re-ran and a new protected image exists ---
    image_updated = False
    action = result.get("pipeline_action_taken")
    new_vlm_changes: List[Dict[str, Any]] = []

    if action and action not in ("none", None, "error", "query"):
        pipeline_state = coordinator.get_pipeline_state()
        new_protected_path = (pipeline_state or {}).get("protected_image_path")

        if new_protected_path and Path(new_protected_path).exists():
            new_bytes = _read_image_bytes(new_protected_path)
            if new_bytes is not None:
                state["protected_bytes"] = new_bytes
                image_updated = True
                logger.info("Protected image updated from %s", new_protected_path)
        else:
            if state["image_path"]:
                stem = Path(state["image_path"]).stem
                candidate = (
                    _PROJECT_ROOT / "data" / "full_pipeline_results"
                    / f"{stem}_protected.png"
                )
                if candidate.exists():
                    new_bytes = _read_image_bytes(str(candidate))
                    if new_bytes is not None:
                        state["protected_bytes"] = new_bytes
                        image_updated = True

        # Re-extract VLM changes from the new pipeline output if available
        new_pipeline_output = (pipeline_state or {}).get("pipeline_output")
        if new_pipeline_output is not None:
            new_vlm_changes = _extract_vlm_changes(new_pipeline_output)
            state["vlm_changes"] = new_vlm_changes

    # --- Build intent label ---
    intent_dict = result.get("intent") or {}
    intent_label = (
        f"{intent_dict.get('action', 'query')} "
        f"(conf={intent_dict.get('confidence', 0):.2f})"
    )

    # --- Log entry ---
    log_entry = {
        "ts": time.time(),
        "type": "chat",
        "user": message,
        "intent": intent_label,
        "response": result.get("response_text", ""),
        "action": action or "none",
        "timing_ms": round(elapsed_ms, 1),
        "image_updated": image_updated,
        "suggestions": result.get("suggestions") or [],
        "vlm_changes": new_vlm_changes,
    }
    _append_log(log_entry)

    return JSONResponse({
        "intent": intent_dict,
        "response_text": result.get("response_text", ""),
        "pipeline_action_taken": action,
        "suggestions": result.get("suggestions") or [],
        "timing_ms": round(elapsed_ms, 1),
        "image_updated": image_updated,
        "vlm_changes": new_vlm_changes,
    })


@app.get("/api/image/original")
async def get_original() -> Response:
    if state["original_bytes"] is None:
        return Response(status_code=404)
    return Response(content=state["original_bytes"], media_type="image/png")


@app.get("/api/image/protected")
async def get_protected() -> Response:
    data = state["protected_bytes"] or state["original_bytes"]
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type="image/png")


@app.get("/api/logs")
async def get_logs() -> JSONResponse:
    return JSONResponse(state["logs"])


@app.get("/api/summary")
async def get_summary() -> JSONResponse:
    return JSONResponse(state["summary"])


@app.get("/api/vlm-changes")
async def get_vlm_changes() -> JSONResponse:
    return JSONResponse(state.get("vlm_changes", []))


@app.get("/api/registered-faces")
async def get_registered_faces() -> JSONResponse:
    """List all registered faces in the consent database."""
    db = _get_face_db()
    if db is None:
        return JSONResponse({"persons": [], "db_available": False, "total": 0})
    try:
        persons = db.get_all_persons() or []
        return JSONResponse({
            "persons": [
                {
                    "label":         getattr(p, "label", None),
                    "relationship":  getattr(p, "relationship", None),
                    "consent_level": "explicit",
                }
                for p in persons
            ],
            "db_available": True,
            "total": len(persons),
        })
    except Exception as exc:
        logger.error("get_registered_faces failed: %s", exc)
        return JSONResponse({"persons": [], "db_available": True, "total": 0, "error": str(exc)})


@app.post("/api/register-face")
async def register_face(file: UploadFile = File(...)) -> JSONResponse:
    """
    Register a face photo in the MongoDB consent database.
    Extracts a 512-D FaceNet embedding using the already-loaded MTCNN model
    (via the shared PipelineOrchestrator's detection agent) and stores it as
    a PersonEntry with consent_level='explicit' and relationship='self'.
    """
    db = _get_face_db()
    if db is None:
        return JSONResponse(
            {"error": "MongoDB not available. Run: docker compose up -d"},
            status_code=503,
        )

    # Save uploaded bytes to a temp file
    temp_path = f"/tmp/register_face_{uuid.uuid4().hex}.jpg"
    contents = await file.read()
    with open(temp_path, "wb") as fh:
        fh.write(contents)

    try:
        # Use the shared orchestrator's face detection tool (models already loaded)
        orchestrator = _get_or_create_orchestrator()
        face_tool = orchestrator.detection_agent.face_tool
        result_json = face_tool._run(temp_path)
        result = json.loads(result_json)

        faces = result.get("faces", [])
        if not faces:
            return JSONResponse(
                {"error": "No face detected in the uploaded photo. Please use a clear, front-facing photo."},
                status_code=422,
            )

        # Pick the highest-confidence face with a valid embedding
        faces_with_emb = [f for f in faces if f.get("embedding") is not None]
        if not faces_with_emb:
            return JSONResponse(
                {"error": "Face was detected but embedding extraction failed. Please try a higher-quality photo."},
                status_code=422,
            )

        best_face = max(faces_with_emb, key=lambda f: f.get("confidence", 0.0))
        embedding_vec: List[float] = best_face["embedding"]

        # Build PersonEntry and store in MongoDB
        face_emb = FaceEmbedding(
            embedding=embedding_vec,
            source_image=temp_path,
        )
        person = PersonEntry(
            label="Me",
            relationship="self",
            embeddings=[face_emb],
            consent_history=ConsentHistory(
                times_appeared=1,
                times_approved=1,
                last_consent_decision="explicit",
                consent_confidence=1.0,
            ),
        )
        success = db.add_person(person)
        if not success:
            return JSONResponse(
                {"error": "Failed to save face to database. A face with this ID may already exist."},
                status_code=500,
            )

        # Count total registered persons
        all_persons = db.get_all_persons() or []
        total = len(all_persons)

        logger.info("Registered face: person_id=%s label=Me confidence=%.3f",
                    person.person_id, best_face.get("confidence", 0.0))
        return JSONResponse({
            "success": True,
            "person_id": person.person_id,
            "label": "Me",
            "consent_level": "explicit",
            "total_registered": total,
            "message": (
                "Face registered! Your face will now be recognized and "
                "skipped (left unprotected) in future uploads."
            ),
        })

    except Exception as exc:
        logger.error("register_face failed: %s", exc)
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


@app.post("/api/reset")
async def reset() -> JSONResponse:
    state.update({
        "session_id": None,
        "image_path": None,
        "original_bytes": None,
        "protected_bytes": None,
        "coordinator": None,
        "pipeline_output": None,
        "logs": [],
        "summary": {},
        "vlm_changes": [],
        # keep orchestrator alive — model weights should not be reloaded
    })
    return JSONResponse({"status": "reset"})


# ---------------------------------------------------------------------------
# Embedded HTML page
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>HITL Demo — On-Device Privacy Detector</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #111827;
      --surface:   #1f2937;
      --surface2:  #374151;
      --border:    #374151;
      --accent:    #3b82f6;
      --accent-h:  #2563eb;
      --success:   #10b981;
      --warn:      #f59e0b;
      --danger:    #ef4444;
      --purple:    #c084fc;
      --text:      #f9fafb;
      --text-dim:  #9ca3af;
      --radius:    8px;
      --mono:      "JetBrains Mono", "Fira Code", "Cascadia Code", monospace;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* ── VLM status banner ───────────────────────────────────────── */
    .vlm-banner {
      background: #0c1a0c;
      border-bottom: 1px solid #1a3a1a;
      padding: 5px 20px;
      font-size: 12px;
      display: flex;
      align-items: center;
      gap: 16px;
      flex-shrink: 0;
      color: var(--text-dim);
    }

    .vlm-banner.warn {
      background: #1a0f00;
      border-bottom-color: #3a2000;
    }

    .vlm-banner .vlm-indicator {
      display: flex;
      align-items: center;
      gap: 6px;
      font-weight: 500;
    }

    .vlm-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      flex-shrink: 0;
    }

    .vlm-dot.green  { background: var(--success); box-shadow: 0 0 6px var(--success); }
    .vlm-dot.red    { background: var(--danger);  box-shadow: 0 0 6px var(--danger); }
    .vlm-dot.gray   { background: var(--text-dim); }

    .vlm-banner .req {
      color: var(--text-dim);
      font-size: 11px;
    }

    .vlm-banner .req code {
      background: var(--surface2);
      padding: 1px 5px;
      border-radius: 3px;
      font-family: var(--mono);
      font-size: 10px;
      color: var(--text);
    }

    /* ── Header ─────────────────────────────────────────────────── */
    header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0 20px;
      height: 54px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
    }

    .header-left {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .logo {
      font-weight: 700;
      font-size: 15px;
      letter-spacing: -0.3px;
      color: var(--text);
    }

    .logo span { color: var(--accent); }

    .badge {
      background: var(--surface2);
      color: var(--text-dim);
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
    }

    .badge.active {
      background: #052e16;
      color: var(--success);
      border-color: var(--success);
    }

    .badge.vlm-on {
      background: #0f172a;
      color: var(--accent);
      border-color: var(--accent);
    }

    .badge.fallback {
      background: #1c1200;
      color: var(--warn);
      border-color: var(--warn);
    }

    .btn {
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 6px 14px;
      border-radius: var(--radius);
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.15s;
    }

    .btn:hover { background: var(--accent-h); }

    .btn.secondary {
      background: var(--surface2);
      color: var(--text-dim);
    }

    .btn.secondary:hover { background: var(--border); color: var(--text); }

    .btn:disabled { opacity: 0.45; cursor: not-allowed; }

    /* ── Main layout ─────────────────────────────────────────────── */
    .main {
      display: flex;
      flex: 1;
      overflow: hidden;
    }

    /* ── Left panel ──────────────────────────────────────────────── */
    .left-panel {
      display: flex;
      flex-direction: column;
      flex: 1;
      min-width: 0;
      overflow: hidden;
    }

    /* ── Images row ──────────────────────────────────────────────── */
    .images-row {
      display: flex;
      flex: 1;
      gap: 0;
      overflow: hidden;
      min-height: 0;
    }

    .image-pane {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      border-right: 1px solid var(--border);
    }

    .image-pane:last-child { border-right: none; }

    .pane-label {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 8px 14px;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.7px;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .pane-label .dot {
      width: 7px; height: 7px;
      border-radius: 50%;
      background: var(--text-dim);
    }

    .pane-label .dot.green  { background: var(--success); }
    .pane-label .dot.blue   { background: var(--accent); }

    .image-container {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      background: #0a0f1a;
      position: relative;
    }

    .image-container img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: block;
    }

    .drop-zone {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      cursor: pointer;
      border: 2px dashed var(--border);
      margin: 16px;
      border-radius: var(--radius);
      transition: border-color 0.2s, background 0.2s;
    }

    .drop-zone:hover, .drop-zone.dragover {
      border-color: var(--accent);
      background: rgba(59,130,246,0.05);
    }

    .drop-zone .icon { font-size: 36px; opacity: 0.4; }

    .drop-zone p {
      color: var(--text-dim);
      font-size: 13px;
      text-align: center;
    }

    @keyframes flash-update {
      0%   { opacity: 0.3; }
      100% { opacity: 1; }
    }

    .img-flash { animation: flash-update 0.4s ease-out; }

    /* ── Summary bar ─────────────────────────────────────────────── */
    .summary-bar {
      background: var(--surface);
      border-top: 1px solid var(--border);
      padding: 8px 16px;
      display: flex;
      align-items: center;
      gap: 20px;
      flex-shrink: 0;
      font-size: 12px;
      color: var(--text-dim);
      min-height: 38px;
      flex-wrap: wrap;
    }

    .summary-bar .stat {
      display: flex;
      align-items: center;
      gap: 5px;
    }

    .summary-bar .stat strong { color: var(--text); font-weight: 600; }

    .risk-badge {
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
    }

    .risk-LOW      { background: #052e16; color: var(--success); }
    .risk-MEDIUM   { background: #451a03; color: var(--warn); }
    .risk-HIGH     { background: #450a0a; color: var(--danger); }
    .risk-CRITICAL { background: #3b0764; color: var(--purple); }
    .risk-unknown  { background: var(--surface2); color: var(--text-dim); }

    /* ── Latency panel ───────────────────────────────────────────── */
    .latency-panel {
      background: var(--surface);
      border-top: 1px solid var(--border);
      padding: 10px 16px;
      flex-shrink: 0;
      font-size: 12px;
      display: none;
    }

    .latency-panel.visible { display: block; }

    .latency-title {
      color: var(--text-dim);
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      margin-bottom: 8px;
    }

    .latency-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px 24px;
    }

    .latency-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
    }

    .latency-label { color: var(--text-dim); }

    .latency-value {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--text);
      font-weight: 600;
    }

    .latency-value.vlm { color: var(--accent); }
    .latency-value.p1  { color: var(--success); }
    .latency-value.total { color: var(--warn); }

    .latency-divider {
      border: none;
      border-top: 1px solid var(--border);
      margin: 8px 0;
    }

    /* ── VLM changes panel ──────────────────────────────────────── */
    .vlm-changes-panel {
      background: var(--surface);
      border-top: 1px solid var(--border);
      padding: 8px 16px;
      flex-shrink: 0;
      font-size: 12px;
      display: none;
      max-height: 80px;
      overflow-y: auto;
    }

    .vlm-changes-panel.visible { display: block; }

    .vlm-change-item {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 3px 0;
      color: var(--text-dim);
      font-size: 11px;
    }

    .vlm-change-item .arrow { color: var(--accent); font-weight: 600; }

    .vlm-change-item .tag {
      padding: 1px 5px;
      border-radius: 3px;
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
    }

    .tag-upgrade  { background: #052e16; color: var(--success); }
    .tag-downgrade { background: #3b0764; color: var(--purple); }
    .tag-modified { background: #0f172a; color: var(--accent); }

    /* ── Chat area ────────────────────────────────────────────────── */
    .chat-area {
      background: var(--surface);
      border-top: 1px solid var(--border);
      padding: 10px 14px;
      display: flex;
      gap: 8px;
      align-items: flex-end;
      flex-shrink: 0;
    }

    .chat-input {
      flex: 1;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      color: var(--text);
      padding: 9px 12px;
      font-size: 13px;
      font-family: inherit;
      resize: none;
      line-height: 1.4;
      outline: none;
      transition: border-color 0.15s;
      max-height: 80px;
      overflow-y: auto;
    }

    .chat-input:focus { border-color: var(--accent); }
    .chat-input::placeholder { color: var(--text-dim); }

    .send-btn {
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: var(--radius);
      width: 36px;
      height: 36px;
      font-size: 16px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      transition: background 0.15s;
    }

    .send-btn:hover  { background: var(--accent-h); }
    .send-btn:disabled { opacity: 0.45; cursor: not-allowed; }

    /* ── Right panel (interaction log) ───────────────────────────── */
    .right-panel {
      width: 340px;
      flex-shrink: 0;
      background: var(--surface);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .panel-header {
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.7px;
      color: var(--text-dim);
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .log-count {
      background: var(--surface2);
      color: var(--text-dim);
      font-size: 10px;
      padding: 1px 6px;
      border-radius: 999px;
    }

    .log-list {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .log-list::-webkit-scrollbar { width: 4px; }
    .log-list::-webkit-scrollbar-track { background: transparent; }
    .log-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

    .log-entry {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 9px 11px;
      font-size: 12px;
      line-height: 1.5;
    }

    .log-entry.pipeline { border-left: 3px solid var(--success); }
    .log-entry.chat     { border-left: 3px solid var(--accent); }
    .log-entry.error    { border-left: 3px solid var(--danger); }

    .log-user { color: var(--text); font-weight: 500; word-break: break-word; }
    .log-user::before { content: "> "; color: var(--text-dim); }

    .log-meta {
      color: var(--text-dim);
      font-size: 11px;
      margin-top: 3px;
      font-family: var(--mono);
    }

    .log-response { color: var(--text-dim); margin-top: 4px; word-break: break-word; }

    .log-img-flag {
      display: inline-block;
      background: #052e16;
      color: var(--success);
      font-size: 10px;
      padding: 1px 5px;
      border-radius: 3px;
      margin-top: 3px;
    }

    .log-vlm-change {
      display: inline-block;
      background: #0f172a;
      color: var(--accent);
      font-size: 10px;
      padding: 1px 5px;
      border-radius: 3px;
      margin-top: 3px;
      margin-left: 4px;
    }

    .suggestions-row {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin-top: 6px;
    }

    .suggestion-chip {
      background: var(--surface2);
      color: var(--text-dim);
      font-size: 11px;
      padding: 2px 7px;
      border-radius: 999px;
      cursor: pointer;
      border: 1px solid var(--border);
      transition: background 0.15s, color 0.15s;
    }

    .suggestion-chip:hover {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }

    /* ── Overlay / loading ──────────────────────────────────────── */
    .overlay {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.65);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1000;
      backdrop-filter: blur(3px);
    }

    .overlay.hidden { display: none; }

    .spinner-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 32px 40px;
      text-align: center;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
      min-width: 280px;
    }

    .spinner {
      width: 40px; height: 40px;
      border: 3px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
    }

    @keyframes spin { to { transform: rotate(360deg); } }

    .spinner-label { color: var(--text); font-size: 14px; font-weight: 500; }
    .spinner-sub   { color: var(--text-dim); font-size: 12px; opacity: 0.8; }

    .spinner-phases {
      display: flex;
      flex-direction: column;
      gap: 4px;
      text-align: left;
      width: 100%;
    }

    .phase-row {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      color: var(--text-dim);
      font-family: var(--mono);
    }

    .phase-row .ph-label { }
    .phase-row .ph-status { color: var(--text-dim); }
    .phase-row.active .ph-label { color: var(--accent); }
    .phase-row.done   .ph-label { color: var(--success); }

    /* ── Empty state ────────────────────────────────────────────── */
    .empty-log {
      color: var(--text-dim);
      font-size: 12px;
      text-align: center;
      padding: 24px 12px;
      opacity: 0.7;
    }

    #file-input { display: none; }
    #register-input { display: none; }

    /* ── Register face button + DB badge ─────────────────────────── */
    .btn.register {
      background: #134e4a;
      color: #5eead4;
      border: 1px solid #0d9488;
    }

    .btn.register:hover { background: #0f3d39; }

    .btn.register:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }

    .db-badge {
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--surface2);
      color: var(--text-dim);
      white-space: nowrap;
    }

    .db-badge.online {
      background: #052e16;
      color: var(--success);
      border-color: var(--success);
    }

    .db-badge.offline {
      background: #1c1200;
      color: var(--warn);
      border-color: var(--warn);
    }

    /* ── Consent summary section inside summary bar ─────────────── */
    .consent-row {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
    }

    .consent-tag {
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
    }

    .consent-explicit { background: #052e16; color: var(--success); }
    .consent-none     { background: #450a0a; color: var(--danger); }
    .consent-unknown  { background: var(--surface2); color: var(--text-dim); }

    .chat-input::-webkit-scrollbar { width: 4px; }
    .chat-input::-webkit-scrollbar-track { background: transparent; }
    .chat-input::-webkit-scrollbar-thumb { background: var(--border); }

    .vlm-changes-panel::-webkit-scrollbar { width: 4px; }
    .vlm-changes-panel::-webkit-scrollbar-track { background: transparent; }
    .vlm-changes-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  </style>
</head>
<body>

<!-- Loading overlay -->
<div class="overlay hidden" id="overlay">
  <div class="spinner-box">
    <div class="spinner"></div>
    <div class="spinner-label" id="overlay-label">Running pipeline...</div>
    <div class="spinner-sub" id="overlay-sub">Phase 1 + VLM review — may take 20-60 seconds</div>
    <div class="spinner-phases" id="overlay-phases" style="display:none">
      <div class="phase-row" id="ph-detect"><span class="ph-label">Agent 1  Detection</span><span class="ph-status">waiting</span></div>
      <div class="phase-row" id="ph-risk"><span class="ph-label">Agent 2  Risk Assessment</span><span class="ph-status">waiting</span></div>
      <div class="phase-row" id="ph-consent"><span class="ph-label">Agent 2.5 Consent ID</span><span class="ph-status">waiting</span></div>
      <div class="phase-row" id="ph-strategy"><span class="ph-label">Agent 3  Strategy</span><span class="ph-status">waiting</span></div>
      <div class="phase-row" id="ph-sam"><span class="ph-label">SAM      Segmentation</span><span class="ph-status">waiting</span></div>
      <div class="phase-row" id="ph-exec"><span class="ph-label">Agent 4  Execution</span><span class="ph-status">waiting</span></div>
    </div>
  </div>
</div>

<!-- Hidden file inputs -->
<input type="file" id="file-input" accept="image/*" />
<input type="file" id="register-input" accept="image/*" />

<!-- VLM Status Banner -->
<div class="vlm-banner" id="vlm-banner">
  <div class="vlm-indicator">
    <div class="vlm-dot gray" id="vlm-dot"></div>
    <span id="vlm-status-text">Checking VLM status...</span>
  </div>
  <span class="req">
    Requires: <code>bash start_llama_server.sh</code> in a separate terminal
  </span>
  <span id="vlm-model-hint" style="display:none;color:var(--accent);font-size:11px;">
    Qwen3-VL-30B-A3B (Q4_K_M) &mdash; port 8081
  </span>
</div>

<!-- Header -->
<header>
  <div class="header-left">
    <span class="logo">On-Device Privacy Detector &mdash; <span>HITL Demo</span></span>
    <span class="badge" id="session-badge">No Session</span>
    <span class="badge" id="mode-badge">checking...</span>
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    <span class="db-badge" id="db-badge">DB checking...</span>
    <button class="btn register" id="register-btn" onclick="document.getElementById('register-input').click()" title="Register your face so it will be recognized and left unprotected in future uploads">
      Register My Face
    </button>
    <button class="btn secondary" onclick="triggerUpload()">Upload Image</button>
    <button class="btn secondary" id="reset-btn" onclick="resetSession()">Reset</button>
  </div>
</header>

<!-- Main -->
<div class="main">

  <!-- Left panel -->
  <div class="left-panel">

    <!-- Images row -->
    <div class="images-row">

      <!-- Original -->
      <div class="image-pane">
        <div class="pane-label">
          <span class="dot green"></span>
          Original Image
        </div>
        <div class="image-container" id="orig-container">
          <div class="drop-zone" id="drop-zone" onclick="triggerUpload()">
            <div class="icon">&#128247;</div>
            <p><strong>Click or drag &amp; drop</strong><br/>to upload an image</p>
          </div>
        </div>
      </div>

      <!-- Protected -->
      <div class="image-pane">
        <div class="pane-label">
          <span class="dot blue"></span>
          Protected Image
          <span id="protected-updated" style="display:none;font-size:10px;color:var(--success);margin-left:4px;">updated</span>
        </div>
        <div class="image-container" id="prot-container">
          <img id="protected-img" src="" alt="" style="display:none" />
          <div id="prot-placeholder" style="color:var(--text-dim);font-size:13px;opacity:0.5;">
            Upload an image to see results
          </div>
        </div>
      </div>

    </div>

    <!-- Summary bar -->
    <div class="summary-bar" id="summary-bar">
      <span style="color:var(--text-dim);font-size:12px;">No pipeline results yet.</span>
    </div>

    <!-- Latency breakdown panel -->
    <div class="latency-panel" id="latency-panel">
      <div class="latency-title">Pipeline Latency</div>
      <div class="latency-grid" id="latency-grid"></div>
    </div>

    <!-- VLM changes panel -->
    <div class="vlm-changes-panel" id="vlm-changes-panel">
      <div id="vlm-changes-list"></div>
    </div>

    <!-- Chat -->
    <div class="chat-area">
      <textarea
        class="chat-input"
        id="chat-input"
        rows="1"
        placeholder="Type a command... e.g. 'blur all faces', 'what risks were detected?', 'strengthen face protection'"
        onkeydown="handleChatKey(event)"
      ></textarea>
      <button class="send-btn" id="send-btn" title="Send" onclick="sendMessage()" disabled>
        &#10148;
      </button>
    </div>

  </div>

  <!-- Right panel (log) -->
  <div class="right-panel">
    <div class="panel-header">
      Interaction Log
      <span class="log-count" id="log-count">0</span>
    </div>
    <div class="log-list" id="log-list">
      <div class="empty-log">Logs will appear here after uploading an image.</div>
    </div>
  </div>

</div>

<script>
// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let hasSession    = false;
let _protectedV   = 0;   // cache-busting counter for protected image
let _vlmAvailable = false;
let _fallbackOnly = false;

// ---------------------------------------------------------------------------
// VLM status probe
// ---------------------------------------------------------------------------
async function probeVlm() {
  try {
    const r = await fetch('/api/vlm-status');
    const d = await r.json();
    _vlmAvailable = d.available;
    _fallbackOnly = d.fallback_only;
    updateVlmBanner(d);
  } catch(_) {}
}

function updateVlmBanner(d) {
  const dot    = document.getElementById('vlm-dot');
  const text   = document.getElementById('vlm-status-text');
  const banner = document.getElementById('vlm-banner');
  const hint   = document.getElementById('vlm-model-hint');
  const modeBadge = document.getElementById('mode-badge');

  if (d.fallback_only) {
    dot.className = 'vlm-dot gray';
    text.textContent = 'Phase 1 only (--fallback-only mode)';
    banner.classList.remove('warn');
    modeBadge.textContent = 'Phase 1 only';
    modeBadge.className = 'badge fallback';
    hint.style.display = 'none';
  } else if (d.available) {
    dot.className = 'vlm-dot green';
    text.textContent = 'VLM Connected — full pipeline active';
    banner.classList.remove('warn');
    modeBadge.textContent = 'Full Pipeline (VLM)';
    modeBadge.className = 'badge vlm-on';
    hint.style.display = 'inline';
  } else {
    dot.className = 'vlm-dot red';
    text.textContent = 'VLM Not Available — start llama-server to enable Phase 2';
    banner.classList.add('warn');
    modeBadge.textContent = 'VLM Offline';
    modeBadge.className = 'badge fallback';
    hint.style.display = 'none';
  }
}

// Probe on load and every 30 seconds
probeVlm();
setInterval(probeVlm, 30000);

// ---------------------------------------------------------------------------
// Face registration
// ---------------------------------------------------------------------------
async function updateRegisteredFaces() {
  try {
    const resp = await fetch('/api/registered-faces');
    const data = await resp.json();
    const badge   = document.getElementById('db-badge');
    const regBtn  = document.getElementById('register-btn');
    if (data.db_available) {
      const n = data.total || 0;
      badge.textContent = n === 0 ? '0 faces' : `${n} registered`;
      badge.className = 'db-badge online';
      regBtn.removeAttribute('disabled');
    } else {
      badge.textContent = 'DB offline';
      badge.className = 'db-badge offline';
      regBtn.setAttribute('disabled', 'true');
    }
  } catch (_) {
    document.getElementById('db-badge').textContent = 'DB offline';
    document.getElementById('db-badge').className = 'db-badge offline';
    document.getElementById('register-btn').setAttribute('disabled', 'true');
  }
}

// Probe DB status on load and every 30 seconds
updateRegisteredFaces();
setInterval(updateRegisteredFaces, 30000);

document.getElementById('register-input').addEventListener('change', async function(e) {
  const file = e.target.files[0];
  if (!file) return;
  this.value = '';

  const regBtn = document.getElementById('register-btn');
  regBtn.setAttribute('disabled', 'true');
  regBtn.textContent = 'Registering...';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const resp = await fetch('/api/register-face', { method: 'POST', body: formData });
    const data = await resp.json();

    if (data.success) {
      appendLog({
        type: 'pipeline',
        user: '[register-face] ' + file.name,
        intent: 'consent',
        response: data.message || 'Face registered successfully.',
        action: 'register_face',
        timing_ms: 0,
        image_updated: false,
        vlm_changes: [],
      });
      updateRegisteredFaces();
    } else {
      appendLog({
        type: 'error',
        user: '[register-face] ' + file.name,
        intent: 'consent',
        response: 'Registration failed: ' + (data.error || 'unknown error'),
        action: 'register_face_error',
        timing_ms: 0,
        image_updated: false,
        vlm_changes: [],
      });
    }
  } catch (err) {
    appendLog({
      type: 'error',
      user: '[register-face]',
      intent: 'consent',
      response: 'Registration error: ' + String(err),
      action: 'register_face_error',
      timing_ms: 0,
      image_updated: false,
      vlm_changes: [],
    });
  } finally {
    regBtn.textContent = 'Register My Face';
    updateRegisteredFaces();
  }
});

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------
function triggerUpload() {
  document.getElementById('file-input').click();
}

document.getElementById('file-input').addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (file) doUpload(file);
  this.value = '';
});

// Drag and drop
const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', e => { dropZone.classList.remove('dragover'); });
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) doUpload(file);
});

async function doUpload(file) {
  const isVlm = !_fallbackOnly && _vlmAvailable;
  const sub = isVlm
    ? 'Phase 1 (det + risk + strategy + exec) then VLM review — typically 20-60s'
    : 'Phase 1 deterministic only — typically 3-8s';
  showOverlay('Running pipeline...', sub, isVlm);

  const fd = new FormData();
  fd.append('file', file);

  try {
    const resp = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await resp.json();

    if (data.error && !data.success) {
      hideOverlay();
      appendLog({
        type: 'error',
        user: '[upload] ' + file.name,
        intent: 'pipeline',
        response: 'Pipeline error: ' + data.error,
        action: 'error',
        timing_ms: data.timing_ms || 0,
        image_updated: false,
        vlm_changes: [],
      });
      return;
    }

    if (data.vlm_degraded) {
      appendLog({
        type: 'error',
        user: '[system]',
        intent: 'warning',
        response: 'VLM unavailable during run — showing Phase 1 results only.',
        action: 'vlm_degraded',
        timing_ms: 0,
        image_updated: false,
        vlm_changes: [],
      });
      probeVlm();
    }

    // Show original image
    const origImg = new Image();
    origImg.onload = () => {
      const container = document.getElementById('orig-container');
      container.innerHTML = '';
      origImg.style.cssText = 'max-width:100%;max-height:100%;object-fit:contain;display:block;';
      container.appendChild(origImg);
    };
    origImg.src = '/api/image/original?v=' + Date.now();

    // Show protected image
    _protectedV++;
    refreshProtectedImage(true);

    // Update summary bar and latency
    updateSummary(data.summary);
    updateLatencyPanel(data.summary);

    // Show VLM changes
    updateVlmChanges(data.vlm_changes || []);

    // Enable chat
    hasSession = true;
    document.getElementById('send-btn').removeAttribute('disabled');
    document.getElementById('session-badge').textContent = 'Session Active';
    document.getElementById('session-badge').classList.add('active');

    refreshLogs();

  } catch(err) {
    appendLog({
      type:'error', user:'[upload]', intent:'error', response: String(err),
      action:'error', timing_ms:0, image_updated:false, vlm_changes:[],
    });
  } finally {
    hideOverlay();
  }
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------
function handleChatKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg || !hasSession) return;

  input.value = '';
  input.style.height = 'auto';

  document.getElementById('send-btn').setAttribute('disabled', 'true');
  const preview = msg.substring(0, 60) + (msg.length > 60 ? '...' : '');
  showOverlay('Processing command...', '"' + preview + '"', false);

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg }),
    });
    const data = await resp.json();

    if (data.error) {
      appendLog({
        type:'error', user: msg, intent:'error', response: data.error,
        action:'error', timing_ms:0, image_updated:false, vlm_changes:[],
      });
      return;
    }

    if (data.image_updated) {
      _protectedV++;
      refreshProtectedImage(true);
      const flag = document.getElementById('protected-updated');
      flag.style.display = 'inline';
      setTimeout(() => { flag.style.display = 'none'; }, 3000);
    }

    const intentLabel = data.intent
      ? `${data.intent.action || 'query'} (conf=${(data.intent.confidence || 0).toFixed(2)})`
      : 'unknown';

    appendLog({
      type: 'chat',
      user: msg,
      intent: intentLabel,
      response: data.response_text || '',
      action: data.pipeline_action_taken || 'none',
      timing_ms: data.timing_ms || 0,
      image_updated: data.image_updated || false,
      suggestions: data.suggestions || [],
      vlm_changes: data.vlm_changes || [],
    });

    if (data.vlm_changes && data.vlm_changes.length > 0) {
      updateVlmChanges(data.vlm_changes);
    }

    if (data.image_updated || (data.pipeline_action_taken && data.pipeline_action_taken !== 'none')) {
      await refreshSummary();
    }

  } catch(err) {
    appendLog({
      type:'error', user: msg, intent:'error', response: String(err),
      action:'error', timing_ms:0, image_updated:false, vlm_changes:[],
    });
  } finally {
    document.getElementById('send-btn').removeAttribute('disabled');
    hideOverlay();
  }
}

function useSuggestion(text) {
  const input = document.getElementById('chat-input');
  input.value = text;
  input.focus();
}

// ---------------------------------------------------------------------------
// Image refresh
// ---------------------------------------------------------------------------
function refreshProtectedImage(animated) {
  const img = document.getElementById('protected-img');
  const placeholder = document.getElementById('prot-placeholder');
  const src = '/api/image/protected?v=' + _protectedV;
  const tmp = new Image();
  img.style.display = 'none';
  tmp.onload = () => {
    img.src = src;
    img.style.display = 'block';
    placeholder.style.display = 'none';
    if (animated) {
      img.classList.remove('img-flash');
      void img.offsetWidth;
      img.classList.add('img-flash');
    }
  };
  tmp.onerror = () => {
    img.style.display = 'none';
    placeholder.style.display = 'block';
  };
  tmp.src = src;
}

// ---------------------------------------------------------------------------
// Summary bar
// ---------------------------------------------------------------------------
function updateSummary(summary) {
  if (!summary) return;
  const bar = document.getElementById('summary-bar');
  const risk = summary.overall_risk || 'unknown';

  let methodsHtml = 'none';
  if (summary.methods && summary.methods.length > 0) {
    methodsHtml = summary.methods.map(m => `${m.count}&times;${m.method}`).join(', ');
  }

  const lat = summary.latency || {};
  const totalS = lat.total_ms ? (lat.total_ms / 1000).toFixed(1) + 's' : '';

  // Build consent identity HTML
  let consentHtml = '';
  const matches = summary.consent_matches || [];
  if (matches.length > 0) {
    const parts = matches.map(m => {
      const c = (m.consent || 'unknown').toLowerCase();
      const icon = c === 'explicit' ? '&#10003;' : c === 'none' ? '&#128274;' : '?';
      const tagClass = c === 'explicit' ? 'consent-explicit' : c === 'none' ? 'consent-none' : 'consent-unknown';
      const lbl = escHtml(m.person_label || m.detection_id || 'face');
      return `<span class="consent-tag ${tagClass}">${icon} ${lbl}: ${escHtml(c)}</span>`;
    });
    consentHtml = `<span class="consent-row" title="Consent identity matches">${parts.join(' ')}</span>`;
  }

  bar.innerHTML = `
    <span class="stat">
      <span class="risk-badge risk-${risk}">${risk}</span>
    </span>
    <span class="stat">
      <span>Elements:</span>
      <strong>${summary.total_elements || 0}</strong>
    </span>
    <span class="stat">
      <span>Protected:</span>
      <strong>${summary.protected_count || 0}</strong>
    </span>
    <span class="stat" title="Obfuscation methods applied">
      <span>Methods:</span>
      <strong>${methodsHtml}</strong>
    </span>
    ${consentHtml}
    ${totalS ? `<span class="stat" style="margin-left:auto;color:var(--warn);font-family:var(--mono);font-size:11px;">${totalS} total</span>` : ''}
  `;
}

async function refreshSummary() {
  try {
    const resp = await fetch('/api/summary');
    const data = await resp.json();
    updateSummary(data);
    updateLatencyPanel(data);
  } catch (_) {}
}

// ---------------------------------------------------------------------------
// Latency panel
// ---------------------------------------------------------------------------
function updateLatencyPanel(summary) {
  if (!summary || !summary.latency) return;

  const panel = document.getElementById('latency-panel');
  const grid  = document.getElementById('latency-grid');
  const lat   = summary.latency;
  const ps    = lat.per_stage || {};

  const fmt = ms => ms > 0 ? (ms / 1000).toFixed(1) + 's' : '-';

  // Left column: totals. Right column: per-stage breakdown.
  grid.innerHTML = `
    <div>
      <div class="latency-row">
        <span class="latency-label">Total pipeline</span>
        <span class="latency-value total">${fmt(lat.total_ms)}</span>
      </div>
      <div class="latency-row">
        <span class="latency-label">Phase 1 (deterministic)</span>
        <span class="latency-value p1">${fmt(lat.phase1_pure_ms)}</span>
      </div>
      <div class="latency-row">
        <span class="latency-label">VLM stages (P2)</span>
        <span class="latency-value vlm">${fmt(lat.vlm_stages_ms)}</span>
      </div>
    </div>
    <div>
      <div class="latency-row">
        <span class="latency-label">Detection</span>
        <span class="latency-value p1">${fmt(ps.detection_ms)}</span>
      </div>
      <div class="latency-row">
        <span class="latency-label">Risk assessment</span>
        <span class="latency-value vlm">${fmt(ps.risk_assessment_ms)}</span>
      </div>
      <div class="latency-row">
        <span class="latency-label">Consent ID</span>
        <span class="latency-value p1">${fmt(ps.consent_identity_ms)}</span>
      </div>
      <div class="latency-row">
        <span class="latency-label">Strategy</span>
        <span class="latency-value vlm">${fmt(ps.strategy_ms)}</span>
      </div>
      <div class="latency-row">
        <span class="latency-label">SAM segmentation</span>
        <span class="latency-value p1">${fmt(ps.sam_segmentation_ms)}</span>
      </div>
      <div class="latency-row">
        <span class="latency-label">Execution</span>
        <span class="latency-value vlm">${fmt(ps.execution_ms)}</span>
      </div>
    </div>
  `;

  panel.classList.add('visible');
}

// ---------------------------------------------------------------------------
// VLM changes panel
// ---------------------------------------------------------------------------
function updateVlmChanges(changes) {
  const panel = document.getElementById('vlm-changes-panel');
  const list  = document.getElementById('vlm-changes-list');

  if (!changes || changes.length === 0) {
    panel.classList.remove('visible');
    return;
  }

  list.innerHTML = changes.map(c => {
    const tagClass = c.direction === 'upgrade'   ? 'tag-upgrade'
                   : c.direction === 'downgrade' ? 'tag-downgrade'
                   : 'tag-modified';
    const tagLabel = c.direction || 'changed';
    return `
      <div class="vlm-change-item">
        <span class="tag ${tagClass}">${tagLabel}</span>
        <span class="arrow">VLM</span>
        <span>${escHtml(c.change)}</span>
      </div>
    `;
  }).join('');

  panel.classList.add('visible');
}

// ---------------------------------------------------------------------------
// Log
// ---------------------------------------------------------------------------
async function refreshLogs() {
  try {
    const resp = await fetch('/api/logs');
    const logs = await resp.json();
    renderLogs(logs);
  } catch (_) {}
}

function renderLogs(logs) {
  const list = document.getElementById('log-list');
  document.getElementById('log-count').textContent = logs.length;
  if (logs.length === 0) {
    list.innerHTML = '<div class="empty-log">No interactions yet.</div>';
    return;
  }
  list.innerHTML = '';
  const reversed = [...logs].reverse();
  for (const entry of reversed) {
    list.appendChild(buildLogEntry(entry));
  }
}

function appendLog(entry) {
  const list  = document.getElementById('log-list');
  const empty = list.querySelector('.empty-log');
  if (empty) empty.remove();
  list.insertBefore(buildLogEntry(entry), list.firstChild);
  const countEl = document.getElementById('log-count');
  countEl.textContent = parseInt(countEl.textContent || '0', 10) + 1;
}

function buildLogEntry(entry) {
  const div = document.createElement('div');
  div.className = 'log-entry ' + (entry.type || 'chat');

  const user        = entry.user || '';
  const intent      = entry.intent || '';
  const response    = entry.response || '';
  const timing      = entry.timing_ms != null ? entry.timing_ms.toFixed(0) + 'ms' : '';
  const action      = entry.action || '';
  const imgUpdated  = entry.image_updated;
  const suggestions = entry.suggestions || [];
  const vlmChanges  = entry.vlm_changes || [];

  let sugHtml = '';
  if (suggestions.length > 0) {
    sugHtml = `<div class="suggestions-row">${
      suggestions.slice(0, 4).map(s =>
        `<span class="suggestion-chip" onclick="useSuggestion(${JSON.stringify(s)})">${escHtml(s)}</span>`
      ).join('')
    }</div>`;
  }

  const vlmBadge = vlmChanges.length > 0
    ? `<span class="log-vlm-change">${vlmChanges.length} VLM change${vlmChanges.length > 1 ? 's' : ''}</span>`
    : '';

  div.innerHTML = `
    <div class="log-user">${escHtml(user)}</div>
    <div class="log-meta">intent: ${escHtml(intent)}${action ? ' &middot; action: ' + escHtml(action) : ''}${timing ? ' &middot; ' + timing : ''}</div>
    ${response ? `<div class="log-response">${escHtml(response)}</div>` : ''}
    ${imgUpdated ? '<span class="log-img-flag">image updated</span>' : ''}
    ${vlmBadge}
    ${sugHtml}
  `;
  return div;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ---------------------------------------------------------------------------
// Session reset
// ---------------------------------------------------------------------------
async function resetSession() {
  if (!confirm('Reset session? This will clear all results.')) return;
  try { await fetch('/api/reset', { method: 'POST' }); } catch (_) {}

  hasSession  = false;
  _protectedV = 0;

  const origContainer = document.getElementById('orig-container');
  origContainer.innerHTML = `
    <div class="drop-zone" id="drop-zone" onclick="triggerUpload()">
      <div class="icon">&#128247;</div>
      <p><strong>Click or drag &amp; drop</strong><br/>to upload an image</p>
    </div>
  `;
  const dz = document.getElementById('drop-zone');
  dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', e => { dz.classList.remove('dragover'); });
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('dragover');
    const file = e.dataTransfer.files[0]; if (file) doUpload(file);
  });

  document.getElementById('protected-img').style.display        = 'none';
  document.getElementById('prot-placeholder').style.display     = 'block';
  document.getElementById('prot-placeholder').textContent       = 'Upload an image to see results';
  document.getElementById('send-btn').setAttribute('disabled', 'true');
  document.getElementById('session-badge').textContent          = 'No Session';
  document.getElementById('session-badge').classList.remove('active');
  document.getElementById('summary-bar').innerHTML              = '<span style="color:var(--text-dim);font-size:12px;">No pipeline results yet.</span>';
  document.getElementById('log-list').innerHTML                 = '<div class="empty-log">Logs will appear here after uploading an image.</div>';
  document.getElementById('log-count').textContent              = '0';
  document.getElementById('protected-updated').style.display   = 'none';
  document.getElementById('latency-panel').classList.remove('visible');
  document.getElementById('vlm-changes-panel').classList.remove('visible');
}

// ---------------------------------------------------------------------------
// Overlay
// ---------------------------------------------------------------------------
function showOverlay(label, sub, showPhases) {
  document.getElementById('overlay-label').textContent = label || 'Processing...';
  document.getElementById('overlay-sub').textContent   = sub   || '';
  const phasesEl = document.getElementById('overlay-phases');
  phasesEl.style.display = showPhases ? 'flex' : 'none';
  document.getElementById('overlay').classList.remove('hidden');
}

function hideOverlay() {
  document.getElementById('overlay').classList.add('hidden');
}

// ---------------------------------------------------------------------------
// Auto-resize textarea
// ---------------------------------------------------------------------------
document.getElementById('chat-input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 80) + 'px';
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # VLM probe before printing banner
    _refresh_vlm_status()

    mode_label = "fallback_only=True (Phase 1 deterministic only)" if _ARGS.fallback_only else "FULL PIPELINE (VLM enabled)"
    vlm_status = "N/A (fallback mode)" if _ARGS.fallback_only else ("CONNECTED" if _vlm_available else "NOT AVAILABLE — start with: bash start_llama_server.sh")

    print("=" * 68)
    print("  HITL Interactive Demo Server  v2.0")
    print(f"  URL   : http://localhost:{_ARGS.port}")
    print(f"  Mode  : {mode_label}")
    print(f"  VLM   : {vlm_status}")
    print()
    if not _ARGS.fallback_only:
        print("  Startup workflow:")
        print("    Terminal 1 : bash start_llama_server.sh")
        print(f"    Terminal 2 : conda run -n lab_env python tests/hitl_demo_server.py")
        print(f"    Browser    : http://localhost:{_ARGS.port}")
        print()
        print("  Phase 1 only (no VLM):")
        print(f"    conda run -n lab_env python tests/hitl_demo_server.py --fallback-only")
    print("=" * 68)

    if not _ARGS.fallback_only and not _vlm_available:
        print()
        print("  WARNING: llama-server not detected on port 8081.")
        print("  The server will start, but VLM phases will fail gracefully.")
        print("  Run 'bash start_llama_server.sh' in another terminal to enable full pipeline.")
        print()

    uvicorn.run(
        "tests.hitl_demo_server:app" if _ARGS.reload else app,
        host=_ARGS.host,
        port=_ARGS.port,
        reload=_ARGS.reload,
        log_level="info",
    )
