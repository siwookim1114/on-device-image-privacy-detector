"""
PipelineOrchestrator — Single entry point for the full privacy protection pipeline.

Replaces the manual phase stitching in tests/test_full_pipeline.py with a reusable,
configurable class that tracks per-stage timing and peak memory usage.

Phase flow:
  A  — Detection (Agent 1)
  B  — Risk Assessment (Agent 2, two-phase)
  C  — Consent Identity (Agent 2.5)
  D  — Strategy (Agent 3, two-phase)
  D.5— SAM Segmentation (PrecisionSegmenter, optional)
  E  — Execution (Agent 4, two-phase)
  F  — Export (JSON + risk map + strategy JSON)
"""

import time
import tracemalloc
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

from PIL import Image

from utils.config import load_config
from utils.models import (
    PipelineInput,
    PipelineOutput,
    PrivacyProfile,
    RiskLevel,
    RiskAnalysisResult,
    DetectionResults,
)
from utils.visualization import (
    export_risk_results_json,
    generate_risk_map,
    export_strategy_results_json,
    generate_protection_preview,
)
from utils.storage import FaceDatabase
from agents.detection_agent import DetectionAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.consent_identity_agent import ConsentIdentityAgent
from agents.strategy_agent import StrategyAgent
from agents.execution_agent import ExecutionAgent


# Configuration dataclass (ablation toggles)
@dataclass
class PipelineConfig:
    """
    Runtime toggles for the PipelineOrchestrator.

    All flags default to enabled — set False to ablate a specific phase
    during evaluation or testing.
    """
    # Enable Agent 2 Phase 2 VLM review and Agent 3/4 Phase 2 VLM review
    enable_vlm_review: bool = True
    # Enable Agent 2.5 (Consent Identity) face-matching phase
    enable_consent: bool = True
    # Enable SAM segmentation (Phase D.5)
    enable_sam: bool = True
    # Enable Agent 4 Phase 2 VLM verification
    enable_execution_verify: bool = True
    # Master switch: disables ALL VLM calls (implies enable_vlm_review=False,
    # enable_execution_verify=False).  Equivalent to --fallback-only.
    fallback_only: bool = False

    # MongoDB settings for Consent Identity Agent
    mongo_uri: str = "mongodb://localhost:27017/"
    mongo_db_name: str = "privacy_guard"
    encryption_key_path: str = "data/face_db/.encryption_key"

    # Output directory (relative paths resolved against project root)
    output_dir: str = "data/full_pipeline_results"

    # Reasoning mode for Risk Assessment Agent
    reasoning_mode: str = "balanced"

    # VLM backend identifier
    vlm_backend: str = "llama-cpp"

    def use_vlm(self) -> bool:
        """True when any agent is allowed to call the VLM."""
        if self.fallback_only:
            return False
        return self.enable_vlm_review or self.enable_execution_verify


# PipelineOrchestrator

class PipelineOrchestrator:
    """
    Orchestrates the full six-phase privacy protection pipeline.

    Agents are initialised once in __init__() and reused across multiple
    run() calls, so model weights are only loaded once per process.

    Usage:
        orchestrator = PipelineOrchestrator(config=PipelineConfig())
        output = orchestrator.run("path/to/image.jpg")
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        privacy_profile: Optional[PrivacyProfile] = None,
    ):
        """
        Load system config, initialise all agents, and (optionally) connect to
        MongoDB for consent identity matching.

        Args:
            config:          Ablation / feature toggles.  Defaults to fully
                             enabled (all VLM phases on).
            privacy_profile: User privacy preferences.  Defaults to system
                             defaults if not provided.
        """
        self.pipeline_cfg = config or PipelineConfig()
        self.privacy_profile = privacy_profile or PrivacyProfile()

        # Load system-level configuration (YAML / device settings)
        self.sys_config = load_config()

        # Resolve output directory (absolute)
        project_root = Path(__file__).parent.parent
        self.output_dir = Path(self.pipeline_cfg.output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = project_root / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------ #
        # Agent initialisation                                                  #
        # ------------------------------------------------------------------ #
        print("[PipelineOrchestrator] Initialising agents...")

        # Agent 1 — Detection
        self.detection_agent = DetectionAgent(self.sys_config)
        print("  Agent 1 (Detection) ready")

        # Agent 2 — Risk Assessment (shares EasyOCR reader with Agent 1)
        _vlm_backend_risk = (
            self.pipeline_cfg.vlm_backend
            if not self.pipeline_cfg.fallback_only
            else "llama-cpp"
        )
        self.risk_agent = RiskAssessmentAgent(
            config=self.sys_config,
            privacy_profile=self.privacy_profile,
            reasoning_mode=self.pipeline_cfg.reasoning_mode,
            ocr_reader=self.detection_agent.text_tool.detector,
        )
        print("  Agent 2 (Risk Assessment) ready")

        # Agent 2.5 — Consent Identity (requires MongoDB)
        self.face_db: Optional[FaceDatabase] = None
        self.consent_agent: Optional[ConsentIdentityAgent] = None
        if self.pipeline_cfg.enable_consent:
            try:
                self.face_db = FaceDatabase(
                    mongo_uri=self.pipeline_cfg.mongo_uri,
                    database_name=self.pipeline_cfg.mongo_db_name,
                    encryption_key_path=self.sys_config.get(
                        "storage.encryption_key_path",
                        self.pipeline_cfg.encryption_key_path,
                    ),
                    encryption_enabled=True,
                )
                self.consent_agent = ConsentIdentityAgent(
                    config=self.sys_config,
                    privacy_profile=self.privacy_profile,
                    face_db=self.face_db,
                )
                print("  Agent 2.5 (Consent Identity) ready")
            except Exception as exc:
                print(
                    f"  Agent 2.5 (Consent Identity) FAILED to initialise — "
                    f"consent phase will be skipped.  Error: {exc}"
                )
                self.consent_agent = None

        # Agent 3 — Strategy
        _vlm_backend_strategy = (
            self.pipeline_cfg.vlm_backend
            if not self.pipeline_cfg.fallback_only
            else "llama-cpp"
        )
        self.strategy_agent = StrategyAgent(
            config=self.sys_config,
            privacy_profile=self.privacy_profile,
            vlm_backend=_vlm_backend_strategy,
        )
        print("  Agent 3 (Strategy) ready")

        # Agent 4 — Execution
        _exec_vlm = (
            self.pipeline_cfg.vlm_backend
            if (
                not self.pipeline_cfg.fallback_only
                and self.pipeline_cfg.enable_execution_verify
            )
            else None
        )
        self.execution_agent = ExecutionAgent(vlm_backend=_exec_vlm)
        print("  Agent 4 (Execution) ready")

        # SAM PrecisionSegmenter (optional — import may fail if mobile-sam not installed)
        self.segmenter = None
        if self.pipeline_cfg.enable_sam and not self.pipeline_cfg.fallback_only:
            try:
                from utils.segmentation import PrecisionSegmenter
                self.segmenter = PrecisionSegmenter(device="cpu")
                print("  SAM (PrecisionSegmenter) ready")
            except ImportError:
                print("  SAM skipped — mobile-sam not installed")
            except Exception as exc:
                print(f"  SAM skipped — initialisation error: {exc}")

        print("[PipelineOrchestrator] All agents ready.\n")

    # Public API                                                               

    def run(
        self,
        image_path: str,
        annotated_image: Optional[Image.Image] = None,
    ) -> PipelineOutput:
        """
        Execute the full A→F pipeline for a single image.

        Args:
            image_path:       Absolute or relative path to the source image.
            annotated_image:  Pre-computed annotated image (Detection Agent
                              output).  If None, the orchestrator runs Phase A
                              and generates it internally.

        Returns:
            PipelineOutput with success flag, protected image path, risk
            analysis, execution report, phase timings, and peak memory stats.
        """
        image_path = str(Path(image_path).resolve())
        stem = Path(image_path).stem

        phase_timings: Dict[str, float] = {}
        pipeline_error: Optional[str] = None

        # Saved artefact paths (populated as phases succeed)
        protected_path: Optional[str] = None
        risk_result: Optional[RiskAnalysisResult] = None
        execution_report = None
        strategy_result = None
        detections: Optional[DetectionResults] = None

        # Start memory profiling
        tracemalloc.start()
        pipeline_start = time.perf_counter()

        try:
            # ----------------------------------------------------------------
            # Phase A — Detection
            # ----------------------------------------------------------------
            phase_start = time.perf_counter()
            print(f"\n  Phase A: Detection Agent...")
            try:
                detections = self.detection_agent.run(image_path)
                if annotated_image is None:
                    annotated_image = self.detection_agent.get_annotated_image()
                print(
                    f"  Detection: {len(detections.faces)} faces, "
                    f"{len(detections.text_regions)} text regions, "
                    f"{len(detections.objects)} objects"
                )
            except Exception as exc:
                raise _PhaseError("detection", exc)
            finally:
                phase_timings["detection_ms"] = _elapsed_ms(phase_start)

            _, peak_after_a = tracemalloc.get_traced_memory()

            # ----------------------------------------------------------------
            # Phase B — Risk Assessment
            # ----------------------------------------------------------------
            phase_start = time.perf_counter()
            print(f"\n  Phase B: Risk Assessment Agent...")
            try:
                if self.pipeline_cfg.fallback_only or not self.pipeline_cfg.enable_vlm_review:
                    # Phase 1 only — no VLM
                    img = Image.open(image_path)
                    w, h = img.size
                    image_context = {
                        "width": w,
                        "height": h,
                        "total_faces": len(detections.faces),
                        "total_texts": len(detections.text_regions),
                        "total_objects": len(detections.objects),
                    }
                    _t0 = time.time()
                    assessments = self.risk_agent._tool_based_assessment(
                        detections, image_context
                    )
                    risk_result = self.risk_agent._build_result(
                        assessments, image_path, _t0
                    )
                else:
                    risk_result = self.risk_agent.run(detections, annotated_image)

                print(
                    f"  Risk Assessment: {len(risk_result.risk_assessments)} assessments, "
                    f"overall={risk_result.overall_risk_level.value.upper()}"
                )
            except Exception as exc:
                raise _PhaseError("risk_assessment", exc)
            finally:
                phase_timings["risk_assessment_ms"] = _elapsed_ms(phase_start)

            _, peak_after_b = tracemalloc.get_traced_memory()

            # ----------------------------------------------------------------
            # Phase C — Consent Identity
            # ----------------------------------------------------------------
            phase_start = time.perf_counter()
            if self.pipeline_cfg.enable_consent and self.consent_agent is not None:
                print(f"\n  Phase C: Consent Identity Agent...")
                try:
                    risk_result = self.consent_agent.run(detections, risk_result)
                    print("  Consent Identity: complete")
                except Exception as exc:
                    # Non-fatal: log and continue with unmodified risk_result
                    print(f"  Phase C error (non-fatal): {exc}")
                    traceback.print_exc()
            else:
                print(f"\n  Phase C: Consent Identity skipped (disabled or unavailable)")
            phase_timings["consent_identity_ms"] = _elapsed_ms(phase_start)

            _, peak_after_c = tracemalloc.get_traced_memory()

            # ----------------------------------------------------------------
            # Phase D — Strategy
            # ----------------------------------------------------------------
            phase_start = time.perf_counter()
            print(f"\n  Phase D: Strategy Agent...")
            try:
                if self.pipeline_cfg.fallback_only or not self.pipeline_cfg.enable_vlm_review:
                    strategy_result = self.strategy_agent.run(risk_result, image_path)
                else:
                    strategy_result = self.strategy_agent.run(
                        risk_result, image_path, annotated_image
                    )
                print(
                    f"  Strategy: {len(strategy_result.strategies)} strategies, "
                    f"{strategy_result.total_protections_recommended} protections recommended"
                )
            except Exception as exc:
                raise _PhaseError("strategy", exc)
            finally:
                phase_timings["strategy_ms"] = _elapsed_ms(phase_start)

            _, peak_after_d = tracemalloc.get_traced_memory()

            # ----------------------------------------------------------------
            # Phase D.5 — SAM Segmentation
            # ----------------------------------------------------------------
            phase_start = time.perf_counter()
            seg_results: Dict[str, Any] = {}

            run_sam = (
                self.segmenter is not None
                and self.pipeline_cfg.enable_sam
                and not self.pipeline_cfg.fallback_only
            )
            if run_sam:
                print(f"\n  Phase D.5: SAM Segmentation...")
                try:
                    seg_results = self.segmenter.process_strategies(
                        image_path,
                        strategy_result.strategies,
                        risk_result.risk_assessments,
                        output_dir=str(self.output_dir),
                    )
                    # Attach mask paths to strategy objects for Agent 4
                    for strategy in strategy_result.strategies:
                        if strategy.detection_id in seg_results:
                            mask_data = seg_results[strategy.detection_id]
                            strategy.segmentation_mask_path = mask_data.get("mask_path")

                    print(f"  SAM: {len(seg_results)} masks generated")

                    if seg_results:
                        preview_path = self.output_dir / f"{stem}_protection_preview.png"
                        try:
                            generate_protection_preview(
                                image_path,
                                strategy_result,
                                risk_result,
                                seg_results,
                                output_path=str(preview_path),
                            )
                        except Exception as exc:
                            print(f"  Protection preview error (non-fatal): {exc}")

                except FileNotFoundError as exc:
                    print(f"  SAM skipped ({exc})")
                except Exception as exc:
                    print(f"  SAM error (non-fatal): {exc}")
                    traceback.print_exc()
            else:
                print(f"\n  Phase D.5: SAM Segmentation skipped")

            phase_timings["sam_segmentation_ms"] = _elapsed_ms(phase_start)
            _, peak_after_d5 = tracemalloc.get_traced_memory()

            # ----------------------------------------------------------------
            # Phase E — Execution
            # ----------------------------------------------------------------
            phase_start = time.perf_counter()
            print(f"\n  Phase E: Execution Agent...")
            try:
                protected_output = self.output_dir / f"{stem}_protected.png"
                execution_report = self.execution_agent.run(
                    strategy_result=strategy_result,
                    risk_result=risk_result,
                    image_path=image_path,
                    output_path=str(protected_output),
                )
                protected_path = str(protected_output)
                print(f"  Protected image saved: {protected_path}")
            except Exception as exc:
                raise _PhaseError("execution", exc)
            finally:
                phase_timings["execution_ms"] = _elapsed_ms(phase_start)

            _, peak_after_e = tracemalloc.get_traced_memory()

            # ----------------------------------------------------------------
            # Phase F — Export
            # ----------------------------------------------------------------
            phase_start = time.perf_counter()
            print(f"\n  Phase F: Exporting results...")
            try:
                json_path = self.output_dir / f"{stem}_risk_results.json"
                export_risk_results_json(
                    risk_result,
                    detections=detections,
                    output_path=str(json_path),
                )

                risk_map_path = self.output_dir / f"{stem}_risk_map.png"
                generate_risk_map(
                    risk_result,
                    image_path,
                    output_path=str(risk_map_path),
                )

                strategy_json_path = self.output_dir / f"{stem}_strategies.json"
                export_strategy_results_json(
                    strategy_result,
                    output_path=str(strategy_json_path),
                )
            except Exception as exc:
                # Export failure is non-fatal; protected image is already written
                print(f"  Phase F export error (non-fatal): {exc}")
                traceback.print_exc()
            finally:
                phase_timings["export_ms"] = _elapsed_ms(phase_start)

        except _PhaseError as pe:
            pipeline_error = f"Phase {pe.phase} failed: {pe.cause}"
            print(f"\n  FATAL: {pipeline_error}")
            traceback.print_exc()

        except Exception as exc:
            pipeline_error = f"Unexpected pipeline error: {exc}"
            print(f"\n  FATAL: {pipeline_error}")
            traceback.print_exc()

        # Stop memory profiling
        _current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_ms = _elapsed_ms(pipeline_start)
        phase_timings["total_ms"] = total_ms

        # Peak memory in MB for readability
        peak_mem_mb = peak_mem / (1024 * 1024)
        print(
            f"\n  Pipeline complete — total {total_ms:.0f}ms, "
            f"peak memory {peak_mem_mb:.1f} MB"
        )
        print(f"  Per-phase timings (ms): {_format_timings(phase_timings)}")

        success = pipeline_error is None
        return PipelineOutput(
            success=success,
            protected_image_path=protected_path,
            risk_analysis=risk_result,
            strategy_recommendations=strategy_result,
            execution_report=execution_report,
            total_time_ms=total_ms,
            error_message=pipeline_error,
            # Attach extra metadata as a provenance placeholder; a full
            # ProvenanceLog is not produced here (would require hashing etc.)
            provenance_log=None,
            phase_timings=phase_timings,
        )

    # Convenience helpers                                                      

    def close(self):
        """
        Release external resources (MongoDB connection).

        Call this when the orchestrator is no longer needed, especially in
        test harnesses that create/destroy orchestrators per test case.
        """
        if self.consent_agent is not None:
            try:
                self.consent_agent.close()
            except Exception:
                pass
        if self.face_db is not None:
            try:
                self.face_db.client.close()
            except Exception:
                pass


# Internal helpers

def _elapsed_ms(start: float) -> float:
    """Return milliseconds elapsed since *start* (perf_counter epoch)."""
    return (time.perf_counter() - start) * 1000.0


def _format_timings(timings: Dict[str, float]) -> str:
    """Format phase timings dict as a compact string for console output."""
    parts = [f"{k}={v:.0f}" for k, v in timings.items() if k != "total_ms"]
    return ", ".join(parts)


class _PhaseError(Exception):
    """
    Wraps an exception from a specific pipeline phase so the orchestrator
    can distinguish fatal phase failures from unexpected runtime errors.
    """

    def __init__(self, phase: str, cause: Exception):
        super().__init__(str(cause))
        self.phase = phase
        self.cause = cause
