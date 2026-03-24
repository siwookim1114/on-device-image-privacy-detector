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
from PIL import Image, ImageDraw

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
from agents.agent_factory import create_vlm, resize_for_vlm, build_vlm_agent
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
            self.vlm, self.vlm_model = create_vlm(vlm_backend)

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
        original_snapshot = image.copy()  # Snapshot before any overlays for label restore
        transformations, unchanged, stats = self._phase1_execute(
            image, strategy_result, bbox_lookup, risk_result
        )

        # ---- Phase 1.5: Z-order label restoration with value-aware exclusion ----
        self._restore_labels(image, original_snapshot, strategy_result, bbox_lookup, risk_result)
        print(f"  Phase 1 complete: {stats['applied']} applied "
              f"({stats['sam']} SAM, {stats['bbox']} bbox), "
              f"{len(unchanged)} unchanged")

        # ---- Phase 2: VLM Verification ----
        verification_patches = 0
        if self.vlm and stats["applied"] > 0:
            print(f"\n  Phase 2: VLM visual verification...")
            verification_patches = self._phase2_verify(
                image, transformations, bbox_lookup, strategy_result
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
            transformations_applied=transformations,
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
        risk_result: Optional[RiskAnalysisResult] = None,
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
                used_sam = self._apply_strategy(image, strategy, bbox_lookup, risk_result)
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
        risk_result: Optional[RiskAnalysisResult] = None,
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

        # Try polygon overlay for text solid_overlay (tighter than axis-aligned bbox)
        if method == ObfuscationMethod.SOLID_OVERLAY and "text" in strategy.element.lower():
            polygon = self._get_polygon(strategy.detection_id, risk_result)
            if polygon and len(polygon) >= 3:
                try:
                    self._apply_polygon_overlay(image, polygon, params)
                    print(f"    POLY {method.value:15} -> {strategy.element[:40]}")
                    return False
                except Exception as e:
                    print(f"    POLY failed for {strategy.element}: {e}, falling back to bbox")

        # Bbox fallback
        bbox = bbox_lookup.get(strategy.detection_id)
        if bbox is None:
            raise ValueError(f"No bbox found for detection_id={strategy.detection_id}")

        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        # Add 3px padding for text solid_overlay to close gaps between adjacent fragments.
        # Skip padding for split parts — they share an edge with their sibling label,
        # and padding would bleed into the label's bbox.
        is_split = "_split_" in strategy.detection_id
        pad = 0 if is_split else (
            3 if (method == ObfuscationMethod.SOLID_OVERLAY
                  and "text" in strategy.element.lower()) else 0
        )
        _apply_obfuscation_bbox(image, x, y, w, h, method, params, padding=pad)
        print(f"    BBOX {method.value:15} -> {strategy.element[:40]}")
        return False

    # ==================== Polygon + Label Restoration Helpers ====================

    def _get_polygon(
        self, detection_id: str, risk_result: Optional[RiskAnalysisResult]
    ) -> Optional[List[List[int]]]:
        """Get the EasyOCR 4-point polygon for a text element, if available."""
        if risk_result is None:
            return None
        # Never use polygon for split elements — parent polygon covers full line
        if "_split_" in detection_id:
            return None
        for a in risk_result.risk_assessments:
            if a.detection_id == detection_id and a.text_polygon:
                return a.text_polygon
        return None

    def _apply_polygon_overlay(
        self, image: Image.Image, polygon: List[List[int]], params: dict
    ):
        """Apply solid_overlay using a polygon (tighter than axis-aligned bbox)."""
        color = params.get("color", "black")
        if color == "black" or isinstance(color, str):
            color = (0, 0, 0)
        draw = ImageDraw.Draw(image)
        pts = [(p[0], p[1]) for p in polygon]
        draw.polygon(pts, fill=color)

    def _restore_labels(
        self,
        image: Image.Image,
        original: Image.Image,
        strategy_result: StrategyRecommendations,
        bbox_lookup: Dict[str, BoundingBox],
        risk_result: RiskAnalysisResult,
    ):
        """Restore text label regions with value-aware exclusion.

        After all overlays are drawn, paste label pixels from the original image
        ONLY where they don't overlap any protected element's bbox. This ensures:
        - Labels are visible (restored from original)
        - Value text stays covered (excluded from restoration)
        - No gap between label and value (overlays drawn at full bbox)
        """
        # Collect ALL protected element bboxes (padded) as exclusion rects
        protected_rects = []  # list of (x1, y1, x2, y2)
        for s in strategy_result.strategies:
            if s.recommended_method == ObfuscationMethod.NONE:
                continue
            bbox = bbox_lookup.get(s.detection_id)
            if bbox is None:
                continue
            # Include 3px padding for text solid_overlay (matches _apply_strategy padding).
            # Skip padding for split parts — they don't get padded during execution.
            is_split = "_split_" in s.detection_id
            pad = 0 if is_split else (
                3 if (s.recommended_method == ObfuscationMethod.SOLID_OVERLAY
                      and "text" in s.element.lower()) else 0
            )
            protected_rects.append((
                max(0, bbox.x - pad),
                max(0, bbox.y - pad),
                min(image.width, bbox.x + bbox.width + pad),
                min(image.height, bbox.y + bbox.height + pad),
            ))

        if not protected_rects:
            return

        # Collect text labels to restore
        label_items = []
        for s in strategy_result.strategies:
            if s.recommended_method != ObfuscationMethod.NONE:
                continue
            if "text" not in s.element.lower():
                continue
            bbox = bbox_lookup.get(s.detection_id)
            if bbox:
                label_items.append((s.detection_id, bbox))

        if not label_items:
            return

        restored = 0
        for det_id, bbox in label_items:
            lx1 = max(0, bbox.x)
            ly1 = max(0, bbox.y)
            lx2 = min(image.width, bbox.x + bbox.width)
            ly2 = min(image.height, bbox.y + bbox.height)
            if lx2 <= lx1 or ly2 <= ly1:
                continue

            lw, lh = lx2 - lx1, ly2 - ly1

            # Build safe mask excluding protected value bboxes
            safe_mask = np.ones((lh, lw), dtype=bool)
            for vx1, vy1, vx2, vy2 in protected_rects:
                # Convert to label-local coordinates
                local_x1 = max(0, vx1 - lx1)
                local_y1 = max(0, vy1 - ly1)
                local_x2 = min(lw, vx2 - lx1)
                local_y2 = min(lh, vy2 - ly1)
                if local_x2 > local_x1 and local_y2 > local_y1:
                    safe_mask[local_y1:local_y2, local_x1:local_x2] = False

            # Skip if entire label is inside protected regions
            if not safe_mask.any():
                continue

            # Composite: restore only safe pixels from original
            label_crop = np.array(original.crop((lx1, ly1, lx2, ly2)))
            protected_region = np.array(image.crop((lx1, ly1, lx2, ly2)))
            protected_region[safe_mask] = label_crop[safe_mask]
            image.paste(Image.fromarray(protected_region), (lx1, ly1))
            restored += 1

        if restored > 0:
            print(f"  Label restore: {restored} text labels restored (value-aware exclusion)")

    # ==================== Phase 2: VLM Visual Verification ====================

    def _phase2_verify(
        self,
        image: Image.Image,
        transformations: List[TransformationResult],
        bbox_lookup: Dict[str, BoundingBox],
        strategy_result: Optional[StrategyRecommendations] = None,
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
            vlm_image = resize_for_vlm(image.copy(), max_dim=1024)

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

            # Build screen exclusion zones for AddProtectionTool guard
            screen_exclusion_zones = []
            if strategy_result:
                for s in strategy_result.strategies:
                    if s.recommended_method != ObfuscationMethod.NONE:
                        continue
                    element_desc = s.element.lower()
                    is_screen = any(kw in element_desc for kw in
                                    ["laptop", "tv", "monitor", "screen", "tablet", "phone", "computer"])
                    if not is_screen:
                        continue
                    bbox = bbox_lookup.get(s.detection_id)
                    if bbox:
                        screen_exclusion_zones.append([bbox.x, bbox.y, bbox.width, bbox.height])

            # Build agent
            agent, max_iters = self._build_verification_agent(
                image, transformation_dicts, bbox_lookup, patches_applied,
                screen_exclusion_zones=screen_exclusion_zones,
            )

            # Build status summary (protected elements)
            status_lines = []
            for t in transformation_dicts:
                status_lines.append(
                    f"  {t['detection_id'][:12]} | {t['element'][:35]} | "
                    f"method={t['method']} | bbox={t['bbox']}"
                )
            status_text = "\n".join(status_lines)

            # Build screen exclusion info from strategy_result
            # Include ALL screen/object elements with method=none and screen_state
            screen_exclusion_lines = []
            if strategy_result:
                for s in strategy_result.strategies:
                    if s.recommended_method != ObfuscationMethod.NONE:
                        continue
                    # Find the original assessment's screen_state
                    # Strategy element descriptions contain the element type info
                    det_id = s.detection_id
                    element_desc = s.element
                    # Check if this is a screen device by looking for screen keywords
                    is_screen = any(kw in element_desc.lower() for kw in
                                    ["laptop", "tv", "monitor", "screen", "tablet", "phone", "computer"])
                    if not is_screen:
                        continue
                    bbox = bbox_lookup.get(det_id)
                    bbox_str = bbox.to_list() if bbox else "[unknown]"
                    screen_exclusion_lines.append(
                        f"  {det_id[:12]} | {element_desc[:35]} | "
                        f"method=none (VERIFIED OFF — do NOT protect) | bbox={bbox_str}"
                    )

            if screen_exclusion_lines:
                status_text += (
                    "\n\nSCREEN DEVICES VERIFIED AS OFF (no protection needed):\n"
                    + "\n".join(screen_exclusion_lines)
                )

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
            return len(patches_applied)

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
        screen_exclusion_zones: Optional[List[List[int]]] = None,
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
                screen_exclusion_zones=screen_exclusion_zones or [],
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
            "interface elements are still distinguishable, protection is inadequate.\n"
            "   HOWEVER: Screen devices listed as 'VERIFIED OFF' in the status below have already been "
            "analyzed by a dedicated screen verification agent and confirmed as OFF, CLOSED, or showing "
            "their back/lid. Do NOT add protection to these screens. They are intentionally unprotected.\n\n"
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

        agent = build_vlm_agent(
            vlm=self.vlm,
            tools=tools,
            system_prompt=system_prompt,
            max_iters=max_iters,
            trim_threshold=10,
            trim_keep=8,
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
        failed = sum(1 for t in report.transformations_applied if t.status == "failed")
        if failed:
            print(f"  Failed: {failed}")
        print(f"  Processing time: {report.total_execution_time_ms:.1f}ms")
        print(f"  Protected image: {report.protected_image_path}")
        print(f"{'='*60}")
