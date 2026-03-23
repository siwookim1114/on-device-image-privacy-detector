"""
Agent 4: Execution Agent (Transformation Applier)

Two-phase architecture:
  Phase 1: Deterministic execution — apply obfuscation strategies (fast, no LLM)
  Phase 2: VLM visual verification — check for leaked PII, patch if needed (optional)

Pipeline:
  Phase 1:
    1. Sort strategies by execution_priority (critical first)
    2. For each strategy with method != none:
       a. Load SAM mask if available (precise segmentation)
       b. Fall back to bbox (rectangular region)
       c. Apply obfuscation method (blur, pixelate, solid_overlay)
  Phase 2:
    3. Send protected image to VLM
    4. VLM checks for residual leaks (readable text, identifiable faces, visible screens)
    5. VLM calls tools to patch any leaks found
    6. Save final protected image
    7. Generate execution report

Input:  StrategyRecommendations + RiskAnalysisResult + original image path
Output: ExecutionReport + protected image file
"""

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langgraph.errors import GraphRecursionError
from langchain_core.messages import HumanMessage

from utils.models import (
    RiskAnalysisResult,
    StrategyRecommendations,
    ProtectionStrategy,
    ExecutionReport,
    TransformationResult,
    ObfuscationMethod,
    BoundingBox,
)
from utils.visualization import _apply_obfuscation_bbox, _apply_obfuscation_mask
from agents.local_wrapper import VisionLLM
from agents.tools import (
    PatchRegionTool,
    AddProtectionTool,
    GetProtectionStatusTool,
    FinalizeVerificationTool,
)


class ExecutionAgent:
    """
    Agent 4: Two-Phase Execution Agent

    Phase 1: Deterministic — apply all obfuscation strategies (SAM mask or bbox).
    Phase 2: VLM verification — visual safety net to catch leaked PII.
    """

    def __init__(self, vlm_backend: Optional[str] = None):
        """
        Args:
            vlm_backend: "llama-cpp", "ollama", or "mlx". None = Phase 1 only.
        """
        self.vlm = None
        self.vlm_model = None

        if vlm_backend:
            backend_config = {
                "llama-cpp": {"base_url": "http://localhost:8081"},
                "ollama": {"base_url": "http://localhost:11434"},
                "mlx": {"base_url": "http://localhost:8000"},
            }
            base_url = backend_config.get(vlm_backend, backend_config["llama-cpp"])["base_url"]

            if vlm_backend == "llama-cpp":
                self.vlm_model = "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf"
            elif vlm_backend == "mlx":
                self.vlm_model = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
            else:
                self.vlm_model = "qwen3-vl:30b-a3b"

            self.vlm = VisionLLM(
                model=self.vlm_model,
                base_url=base_url,
                backend=vlm_backend,
            )

        print(f"\n[ExecutionAgent] Initialized")
        print(f"  Phase 1: Deterministic execution (blur, pixelate, solid_overlay)")
        print(f"  Phase 2: VLM verification ({'enabled — ' + self.vlm_model if self.vlm else 'disabled'})")
        print(f"  SAM mask support: enabled (falls back to bbox)")

    # ==================== Public API ====================

    def run(
        self,
        strategy_result: StrategyRecommendations,
        risk_result: RiskAnalysisResult,
        image_path: str,
        output_path: Optional[str] = None,
    ) -> ExecutionReport:
        """
        Apply all approved strategies to the original image.

        Args:
            strategy_result: From Agent 3 (strategy recommendations)
            risk_result: From Agent 2 + 2.5 (risk assessments with bboxes)
            image_path: Path to original image
            output_path: Where to save the protected image (auto-generated if None)

        Returns:
            ExecutionReport with transformation details and protected image path
        """
        start = time.time()

        print(f"\n{'='*60}")
        print(f"Execution Agent - Applying Protections")
        print(f"{'='*60}")

        # Load original image
        image = Image.open(image_path).convert("RGB")
        print(f"  Original image: {image.width}x{image.height}")

        # Build detection_id → BoundingBox lookup
        bbox_lookup = self._build_bbox_lookup(risk_result)

        # ---- Phase 1: Deterministic Execution ----
        print(f"\n  Phase 1: Deterministic execution...")
        transformations, unchanged, stats = self._phase1_execute(
            image, strategy_result, bbox_lookup
        )
        print(f"  Phase 1 complete: {stats['applied']} applied "
              f"({stats['sam']} SAM, {stats['bbox']} bbox), "
              f"{len(unchanged)} unchanged")

        # ---- Phase 2: VLM Verification ----
        verification_patches = 0
        if self.vlm and stats["applied"] > 0:
            print(f"\n  Phase 2: VLM visual verification...")
            verification_patches = self._phase2_verify(
                image, transformations, bbox_lookup
            )
            print(f"  Phase 2 complete: {verification_patches} patches applied")
        elif not self.vlm:
            print(f"\n  Phase 2: Skipped (VLM not enabled)")

        # ---- Save and Report ----
        if output_path is None:
            stem = Path(image_path).stem
            output_dir = Path(image_path).parent
            output_path = str(output_dir / f"{stem}_protected.png")

        image.save(output_path, quality=95)

        total_ms = (time.time() - start) * 1000

        failed = sum(1 for t in transformations if t.status == "failed")
        if failed == 0:
            status = "completed"
        elif failed < len(transformations):
            status = "partial"
        else:
            status = "failed"

        report = ExecutionReport(
            image_path=image_path,
            status=status,
            tranformations_applied=transformations,
            elements_unchanged=unchanged,
            total_execution_time_ms=total_ms,
            protected_image_path=output_path,
        )

        self._print_summary(
            report, stats["applied"], stats["sam"], stats["bbox"],
            len(unchanged), verification_patches,
        )
        return report

    # ==================== Phase 1: Deterministic Execution ====================

    def _phase1_execute(
        self,
        image: Image.Image,
        strategy_result: StrategyRecommendations,
        bbox_lookup: Dict[str, BoundingBox],
    ) -> tuple:
        """Apply all strategies deterministically. Returns (transformations, unchanged, stats)."""
        strategies = sorted(
            strategy_result.strategies,
            key=lambda s: s.execution_priority,
        )

        transformations: List[TransformationResult] = []
        unchanged = []
        applied = 0
        sam_count = 0
        bbox_count = 0

        for strategy in strategies:
            if strategy.recommended_method == ObfuscationMethod.NONE:
                unchanged.append({
                    "detection_id": strategy.detection_id,
                    "element": strategy.element,
                    "reason": "method=none (no protection needed)",
                })
                continue

            t_start = time.time()
            try:
                used_sam = self._apply_strategy(image, strategy, bbox_lookup)
                t_ms = (time.time() - t_start) * 1000

                if used_sam:
                    sam_count += 1
                else:
                    bbox_count += 1
                applied += 1

                transformations.append(TransformationResult(
                    detection_id=strategy.detection_id,
                    element=strategy.element,
                    method=strategy.recommended_method,
                    parameters=strategy.parameters,
                    status="success",
                    execution_time_ms=t_ms,
                ))
            except Exception as e:
                t_ms = (time.time() - t_start) * 1000
                print(f"    FAILED: {strategy.element} — {e}")
                transformations.append(TransformationResult(
                    detection_id=strategy.detection_id,
                    element=strategy.element,
                    method=strategy.recommended_method,
                    parameters=strategy.parameters,
                    status="failed",
                    execution_time_ms=t_ms,
                    error_message=str(e),
                ))

        stats = {"applied": applied, "sam": sam_count, "bbox": bbox_count}
        return transformations, unchanged, stats

    def _apply_strategy(
        self,
        image: Image.Image,
        strategy: ProtectionStrategy,
        bbox_lookup: Dict[str, BoundingBox],
    ) -> bool:
        """Apply a single strategy. Returns True if SAM mask was used."""
        method = strategy.recommended_method
        params = strategy.parameters

        # Try SAM mask first
        if strategy.segmentation_mask_path:
            try:
                mask = np.load(strategy.segmentation_mask_path)
                _apply_obfuscation_mask(image, mask, method, params)
                print(f"    SAM  {method.value:15} -> {strategy.element[:40]}")
                return True
            except Exception as e:
                print(f"    SAM mask failed for {strategy.element}: {e}, falling back to bbox")

        # Bbox fallback
        bbox = bbox_lookup.get(strategy.detection_id)
        if bbox is None:
            raise ValueError(f"No bbox found for detection_id={strategy.detection_id}")

        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        _apply_obfuscation_bbox(image, x, y, w, h, method, params)
        print(f"    BBOX {method.value:15} -> {strategy.element[:40]}")
        return False

    # ==================== Phase 2: VLM Visual Verification ====================

    def _phase2_verify(
        self,
        image: Image.Image,
        transformations: List[TransformationResult],
        bbox_lookup: Dict[str, BoundingBox],
    ) -> int:
        """
        VLM-based visual verification of the protected image.

        Sends the protected image to the VLM and checks for residual leaks.
        Returns number of patches applied.
        """
        print(f"\n{'-'*60}")
        print(f"Execution Phase 2: VLM Visual Verification")
        print(f"{'-'*60}")

        try:
            # Resize for VLM (max 1024px)
            vlm_image = image.copy()
            max_dim = 1024
            w, h = vlm_image.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                vlm_image = vlm_image.resize(
                    (int(w * scale), int(h * scale)), Image.LANCZOS
                )

            image_b64 = self.vlm._image_to_base64(vlm_image)

            # Build shared state
            transformation_dicts = [
                {
                    "detection_id": t.detection_id,
                    "element": t.element,
                    "method": t.method.value,
                    "parameters": t.parameters,
                    "status": t.status,
                    "bbox": bbox_lookup[t.detection_id].to_list()
                    if t.detection_id in bbox_lookup else [0, 0, 0, 0],
                }
                for t in transformations
                if t.status == "success"
            ]
            patches_applied = []

            # Build agent
            agent, max_iters = self._build_verification_agent(
                image, transformation_dicts, bbox_lookup, patches_applied
            )

            # Build status summary
            status_lines = []
            for t in transformation_dicts:
                status_lines.append(
                    f"  {t['detection_id'][:12]} | {t['element'][:35]} | "
                    f"method={t['method']} | bbox={t['bbox']}"
                )
            status_text = "\n".join(status_lines)

            input_message = HumanMessage(content=[
                {"type": "text", "text": (
                    f"This is the PROTECTED image after applying {len(transformation_dicts)} "
                    f"obfuscation transformations. Check for any residual privacy leaks.\n\n"
                    f"Applied protections:\n{status_text}"
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ])

            print(f"  Starting VLM verification ({len(transformation_dicts)} protected elements)...")

            result = agent.invoke(
                {"messages": [input_message]},
                config={"recursion_limit": 2 * max_iters + 5},
            )

            # Count tool calls
            tool_calls = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(tc["name"])

            print(f"  VLM made {len(tool_calls)} tool calls: {tool_calls}")
            print(f"  Patches applied: {len(patches_applied)}")

            return len(patches_applied)

        except GraphRecursionError:
            print(f"  VLM hit max iterations — returning with patches applied so far")
            return len(patches_applied) if 'patches_applied' in dir() else 0

        except Exception as e:
            print(f"  VLM verification failed: {e}")
            traceback.print_exc()
            print(f"  Continuing with Phase 1 protections only")
            return 0

    def _build_verification_agent(
        self,
        image: Image.Image,
        transformation_dicts: List[Dict],
        bbox_lookup: Dict[str, BoundingBox],
        patches_applied: List[Dict],
    ):
        """Build Phase 2 VLM agent with verification tools."""
        tools = [
            PatchRegionTool(
                image=image,
                bbox_lookup=bbox_lookup,
                transformations=transformation_dicts,
                patches_applied=patches_applied,
            ),
            AddProtectionTool(
                image=image,
                patches_applied=patches_applied,
            ),
            GetProtectionStatusTool(
                transformations=transformation_dicts,
                patches_applied=patches_applied,
            ),
            FinalizeVerificationTool(
                patches_applied=patches_applied,
            ),
        ]

        n = len(transformation_dicts)
        max_iters = max(5, min(n + 3, 15))

        system_prompt = (
            "You are a privacy protection verification agent performing a FINAL SAFETY CHECK. "
            "The image has ALREADY been processed with obfuscation (blur, pixelate, solid_overlay). "
            "You are looking at the PROTECTED image.\n\n"
            "YOUR TASK: Check if any sensitive content is STILL VISIBLE despite the applied protections.\n\n"
            "CHECK FOR THESE SPECIFIC LEAKS:\n"
            "1. FACES — Can you recognize or identify any face that should be protected? "
            "Look for faces only partially blurred, with blur too light, or missed entirely.\n"
            "2. TEXT — Can you STILL READ any sensitive text (numbers, names, passwords, PII) "
            "through the obfuscation? If you can make out specific characters or words, "
            "the protection is insufficient.\n"
            "3. SCREENS — Can you see readable content on any screen device? If text or "
            "interface elements are still distinguishable, protection is inadequate.\n\n"
            "IMPORTANT CONSTRAINTS:\n"
            "- Areas that appear blurred/pixelated/black ARE the protections working correctly. "
            "Only flag them if protection is INSUFFICIENT (you can still read text, identify a face).\n"
            "- Do NOT add protection to areas already adequately obscured.\n"
            "- Do NOT add protection to non-sensitive areas (backgrounds, furniture, walls).\n"
            "- When patching, ALWAYS strengthen — use higher kernel_size for blur (35-45), "
            "smaller block_size for pixelate (6-8), or switch to solid_overlay for critical leaks.\n"
            "- Use expand_px (5-20) when protection doesn't fully cover the sensitive region.\n\n"
            "TOOL USAGE:\n"
            "- Use get_protection_status to see what was applied and where.\n"
            "- Use patch_region to strengthen existing protections by detection_id.\n"
            "- Use add_protection ONLY for leaked content not covered by any existing protection.\n"
            "- Call finalize_verification when done. If all protections look adequate, "
            "call it immediately without patching.\n\n"
            "ALWAYS call a tool in EVERY response. No explanations without tool calls.\n"
            "Go directly to tool calls. No preamble or reasoning text.\n"
            "Do NOT use <think> tags or internal reasoning. Act immediately."
        )

        class MessageTrimMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                messages = request.messages
                if len(messages) > 10:
                    trimmed = [messages[0]] + messages[-8:]
                    return handler(request.override(messages=trimmed))
                return handler(request)

        agent = create_agent(
            model=self.vlm.llm,
            tools=tools,
            system_prompt=system_prompt,
            middleware=[
                MessageTrimMiddleware(),
                ModelCallLimitMiddleware(run_limit=max_iters),
            ],
        )

        print(f"  Phase 2 agent built:")
        print(f"    LLM: {self.vlm_model}")
        print(f"    Max iterations: {max_iters}")
        print(f"    Tools: {[t.name for t in tools]}")

        return agent, max_iters

    # ==================== Helpers ====================

    def _build_bbox_lookup(self, risk_result: RiskAnalysisResult) -> Dict[str, BoundingBox]:
        """Build detection_id -> BoundingBox lookup from risk assessments."""
        lookup = {}
        for assessment in risk_result.risk_assessments:
            lookup[assessment.detection_id] = assessment.bbox
        return lookup

    def _print_summary(
        self,
        report: ExecutionReport,
        applied: int,
        sam: int,
        bbox: int,
        unchanged: int,
        verification_patches: int,
    ):
        """Print execution summary."""
        print(f"\n{'='*60}")
        print(f"Execution Complete")
        print(f"{'='*60}")
        print(f"  Status: {report.status}")
        print(f"  Phase 1 — Transformations applied: {applied}")
        if applied > 0:
            print(f"    - SAM (precise): {sam}")
            print(f"    - BBOX (rectangular): {bbox}")
        print(f"  Phase 2 — Verification patches: {verification_patches}")
        print(f"  Elements unchanged: {unchanged}")
        failed = sum(1 for t in report.tranformations_applied if t.status == "failed")
        if failed:
            print(f"  Failed: {failed}")
        print(f"  Processing time: {report.total_execution_time_ms:.1f}ms")
        print(f"  Protected image: {report.protected_image_path}")
        print(f"{'='*60}")
