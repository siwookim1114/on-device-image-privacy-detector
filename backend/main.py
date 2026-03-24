from __future__ import annotations

import os
import traceback
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.routers import auth, consent as consent_router_module, history, hitl, images, pipeline, profile as profile_router_module
from backend.schemas.responses import ErrorDetail, ErrorResponse


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager.

    Startup:
    - Create upload and results directories.
    - Initialise the SessionManager and attach it to app.state.
    - Attach settings to app.state for use by routers.

    Other services (pipeline_service, safety_kernel, coordinator_agent) are
    populated by their respective agent-authored modules at a later stage;
    routers degrade gracefully when those services are absent.
    """
    # -- Directory bootstrap ---------------------------------------------------
    for dir_path in (settings.upload_dir, settings.results_dir):
        os.makedirs(dir_path, exist_ok=True)

    # -- Settings on app.state -------------------------------------------------
    app.state.settings = settings

    # -- SessionManager --------------------------------------------------------
    # Import here to avoid circular imports at module load time; the service
    # module will be provided by the session_manager agent.
    try:
        from backend.services.session_manager import SessionManager  # type: ignore[import]

        session_manager = SessionManager(ttl_hours=settings.session_ttl_hours)
        app.state.session_manager = session_manager
    except ImportError:
        # Session manager not yet written by its owning agent — install a
        # lightweight stub so that auth routes return a clear 503 rather than
        # an unhandled AttributeError.
        app.state.session_manager = _StubSessionManager()

    # -- WebSocketManager (always created — needed by WS router) ---------------
    try:
        from backend.services.websocket_manager import WebSocketManager  # type: ignore[import]

        ws_manager = WebSocketManager()
        app.state.ws_manager = ws_manager
    except ImportError:
        app.state.ws_manager = None

    # -- Optional services (not blocking if absent) ----------------------------
    app.state.pipeline_service = None
    app.state.safety_kernel = None
    app.state.coordinator_agent = None
    app.state.profile_service = None
    app.state.consent_service = None

    try:
        from backend.services.pipeline_service import PipelineService  # type: ignore[import]

        app.state.pipeline_service = PipelineService(
            ws_manager=getattr(app.state, "ws_manager", None),
            session_manager=getattr(app.state, "session_manager", None),
            config_path=settings.pipeline_config_path,
            upload_dir=settings.upload_dir,
            results_dir=settings.results_dir,
            max_concurrent=settings.max_concurrent_pipelines,
        )
    except ImportError:
        pass

    try:
        from backend.services.safety_kernel import SafetyKernel  # type: ignore[import]

        app.state.safety_kernel = SafetyKernel()
    except ImportError:
        pass

    # -- ProfileService --------------------------------------------------------
    try:
        import pymongo  # type: ignore[import]
        from backend.services.profile_service import ProfileService  # type: ignore[import]

        _mongo_client = pymongo.MongoClient(settings.mongo_url, serverSelectionTimeoutMS=3000)
        _mongo_db = _mongo_client[settings.mongo_db]
        # Ensure unique index on user_id for O(1) lookups
        _mongo_db["privacy_profiles"].create_index("user_id", unique=True, background=True)
        app.state.profile_service = ProfileService(db=_mongo_db)
    except Exception as _profile_svc_exc:  # noqa: BLE001
        # Non-fatal: profile endpoints degrade to 503 until MongoDB is reachable
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "ProfileService not started: %s", _profile_svc_exc
        )

    # -- ConsentService --------------------------------------------------------
    try:
        from backend.services.consent_service import ConsentService  # type: ignore[import]

        _consent_svc = ConsentService()
        _consent_svc.initialize(
            mongo_url=settings.mongo_url,
            database_name=settings.mongo_db,
        )
        app.state.consent_service = _consent_svc
    except Exception as _consent_svc_exc:  # noqa: BLE001
        # Non-fatal: consent endpoints degrade to 503 until MongoDB is reachable
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "ConsentService not started: %s", _consent_svc_exc
        )

    yield

    # -- Shutdown cleanup ------------------------------------------------------
    pipeline_service = getattr(app.state, "pipeline_service", None)
    if pipeline_service is not None and hasattr(pipeline_service, "shutdown"):
        # PipelineService.shutdown() is synchronous (calls executor.shutdown).
        pipeline_service.shutdown()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    application = FastAPI(
        title="On-Device Image Privacy Detector API",
        description=(
            "Multi-agent privacy protection pipeline — detection, risk "
            "assessment, consent identity, strategy, segmentation, and "
            "execution — all running locally on Apple Silicon."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # -- CORS -------------------------------------------------------------------
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Global exception handler -----------------------------------------------
    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="PIPELINE_ERROR",
                    message="An unexpected error occurred.",
                    details={"exception": str(exc), "traceback": tb},
                )
            ).model_dump(),
        )

    # -- Health check (outside /api/v1) -----------------------------------------
    @application.get("/health", tags=["ops"], summary="Liveness check.")
    async def health(request: Request) -> dict:
        """Returns service health including reachability of downstream deps."""
        import httpx

        vlm_ok = False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(settings.vlm_health_url)
                vlm_ok = resp.status_code == 200
        except Exception:
            vlm_ok = False

        session_manager_ok = getattr(request.app.state, "session_manager", None) is not None
        pipeline_service_ok = getattr(request.app.state, "pipeline_service", None) is not None

        return {
            "status": "ok",
            "version": "1.0.0",
            "dependencies": {
                "vlm_server": "ok" if vlm_ok else "unavailable",
                "session_manager": "ok" if session_manager_ok else "stub",
                "pipeline_service": "ok" if pipeline_service_ok else "not_loaded",
            },
        }

    # -- Routers (all under /api/v1) -------------------------------------------
    api_prefix = "/api/v1"
    application.include_router(auth.router, prefix=api_prefix)
    application.include_router(pipeline.router, prefix=api_prefix)
    application.include_router(hitl.router, prefix=api_prefix)
    application.include_router(images.router, prefix=api_prefix)
    application.include_router(history.router, prefix=api_prefix)
    application.include_router(profile_router_module.router, prefix=api_prefix)
    application.include_router(consent_router_module.router, prefix=api_prefix)

    return application


# ---------------------------------------------------------------------------
# Stub session manager — used when the real one is not yet available
# ---------------------------------------------------------------------------

class _StubSessionManager:
    """Minimal no-op SessionManager so routers don't crash on import."""

    def create_session(self):  # type: ignore[return]
        from fastapi import HTTPException

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "code": "PIPELINE_ERROR",
                    "message": "Session manager not yet available.",
                    "details": {},
                }
            },
        )

    def get_by_token(self, token: str):  # type: ignore[return]
        return None

    def update_image_meta(self, **kwargs) -> None:  # noqa: ANN003
        pass

    def list_sessions(self) -> list:
        return []

    def list_all(self) -> list:
        return []


# ---------------------------------------------------------------------------
# Module-level app instance (used by uvicorn and import checks)
# ---------------------------------------------------------------------------

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
