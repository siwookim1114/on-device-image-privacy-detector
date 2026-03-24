## Execution Agent Tools (Phase 2 — VLM Verification)

import json
from typing import Any, List, Dict, Optional, Type

from langchain.tools import BaseTool
from utils.models import (
    ObfuscationMethod,
    PatchRegionInput,
    AddProtectionInput,
)
from utils.visualization import _apply_obfuscation_bbox


class PatchRegionTool(BaseTool):
    """Strengthen or re-apply protection for a leaking element."""
    name: str = "patch_region"
    description: str = (
        "Strengthen protection for an element that is still partially visible. "
        "Provide the detection_id (from protection status), a stronger method, "
        "optional parameters (kernel_size, block_size), expand_px to widen coverage, "
        "and reasoning for what you still see."
    )
    args_schema: Type = PatchRegionInput
    handle_tool_error: bool = True
    image: Any = None          # shared PIL Image (mutated in-place)
    bbox_lookup: Any = None    # detection_id -> BoundingBox
    transformations: Any = None  # list of transformation dicts from Phase 1
    patches_applied: Any = None  # list to track Phase 2 patches

    def _run(
        self,
        detection_id: str,
        method: str,
        parameters: Dict[str, Any] = None,
        expand_px: int = 0,
        reasoning: str = "VLM verification",
    ) -> str:
        if parameters is None:
            parameters = {}

        method_lower = method.lower().strip()
        if method_lower not in {"blur", "pixelate", "solid_overlay"}:
            return json.dumps({"status": "error", "message": f"Invalid method '{method}'. Use: blur, pixelate, solid_overlay"})

        if method_lower == "none":
            return json.dumps({"status": "error", "message": "Cannot set method to 'none' — patch_region only strengthens protection"})

        bbox = self.bbox_lookup.get(detection_id)
        if bbox is None:
            return json.dumps({"status": "error", "message": f"Unknown detection_id '{detection_id}'. Use get_protection_status to see valid IDs."})

        # Expand bbox
        expand_px = max(0, min(expand_px, 30))
        x = max(0, bbox.x - expand_px)
        y = max(0, bbox.y - expand_px)
        w = min(self.image.width - x, bbox.width + 2 * expand_px)
        h = min(self.image.height - y, bbox.height + 2 * expand_px)

        method_enum = ObfuscationMethod(method_lower)

        # Set default params if not provided
        if not parameters:
            if method_lower == "blur":
                parameters = {"kernel_size": 35}
            elif method_lower == "pixelate":
                parameters = {"block_size": 8}
            elif method_lower == "solid_overlay":
                parameters = {"color": "black"}

        _apply_obfuscation_bbox(self.image, x, y, w, h, method_enum, parameters)

        patch_record = {
            "detection_id": detection_id,
            "method": method_lower,
            "parameters": parameters,
            "expand_px": expand_px,
            "bbox": [x, y, w, h],
            "reasoning": reasoning,
        }
        self.patches_applied.append(patch_record)

        # Find original method for logging
        orig_method = "unknown"
        for t in self.transformations:
            if t.get("detection_id") == detection_id:
                orig_method = t.get("method", "unknown")
                break

        print(f"    PATCH: {detection_id[:8]} {orig_method} → {method_lower} "
              f"(+{expand_px}px) — {reasoning[:60]}")

        return json.dumps({
            "status": "patched",
            "detection_id": detection_id,
            "old_method": orig_method,
            "new_method": method_lower,
            "parameters": parameters,
            "expanded_by": expand_px,
        })


class AddProtectionTool(BaseTool):
    """Add protection to a region missed by the detection pipeline."""
    name: str = "add_protection"
    description: str = (
        "Add protection to a NEW region not covered by any existing transformation. "
        "Use this ONLY for leaked content that was completely missed by the detection pipeline. "
        "Provide x, y, width, height coordinates, method, and reasoning."
    )
    args_schema: Type = AddProtectionInput
    handle_tool_error: bool = True
    image: Any = None
    patches_applied: Any = None
    screen_exclusion_zones: Any = None  # list of [x, y, w, h] for verified-off screens

    def _run(
        self,
        x: int, y: int, width: int, height: int,
        method: str,
        parameters: Dict[str, Any] = None,
        reasoning: str = "VLM found missed content",
    ) -> str:
        if parameters is None:
            parameters = {}

        method_lower = method.lower().strip()
        if method_lower not in {"blur", "pixelate", "solid_overlay"}:
            return json.dumps({"status": "error", "message": f"Invalid method '{method}'. Use: blur, pixelate, solid_overlay"})

        # Guard: reject patches that overlap significantly with verified-off screen zones
        if self.screen_exclusion_zones:
            for zone in self.screen_exclusion_zones:
                zx, zy, zw, zh = zone
                # Compute intersection
                ix1 = max(x, zx)
                iy1 = max(y, zy)
                ix2 = min(x + width, zx + zw)
                iy2 = min(y + height, zy + zh)
                if ix2 > ix1 and iy2 > iy1:
                    overlap_area = (ix2 - ix1) * (iy2 - iy1)
                    patch_area = width * height
                    # Block if >50% of the patch overlaps a verified-off screen
                    # OR if >40% of the screen zone is covered by the patch
                    zone_area = zw * zh
                    if patch_area > 0 and (
                        overlap_area / patch_area > 0.5
                        or (zone_area > 0 and overlap_area / zone_area > 0.4)
                    ):
                        return json.dumps({
                            "status": "blocked",
                            "message": (
                                f"Region [{x},{y},{width},{height}] overlaps a screen device that was "
                                f"verified as OFF by the screen verification agent. Do NOT protect "
                                f"verified-off screens."
                            ),
                        })

        # Validate region
        if width < 10 or height < 10:
            return json.dumps({"status": "error", "message": f"Region too small ({width}x{height}). Minimum 10x10."})

        # Reject absurdly large patches (>25% of image area) to prevent VLM overreaction
        max_area = self.image.width * self.image.height * 0.25
        if width * height > max_area:
            return json.dumps({"status": "error", "message": f"Region too large ({width}x{height}={width*height}px). Max 25% of image ({int(max_area)}px)."})

        x = max(0, min(x, self.image.width - 10))
        y = max(0, min(y, self.image.height - 10))
        width = min(width, self.image.width - x)
        height = min(height, self.image.height - y)

        if not parameters:
            if method_lower == "blur":
                parameters = {"kernel_size": 35}
            elif method_lower == "pixelate":
                parameters = {"block_size": 8}
            elif method_lower == "solid_overlay":
                parameters = {"color": "black"}

        method_enum = ObfuscationMethod(method_lower)
        _apply_obfuscation_bbox(self.image, x, y, width, height, method_enum, parameters)

        patch_record = {
            "detection_id": "vlm_added",
            "method": method_lower,
            "parameters": parameters,
            "bbox": [x, y, width, height],
            "reasoning": reasoning,
        }
        self.patches_applied.append(patch_record)

        print(f"    ADD:   [{x},{y},{width},{height}] {method_lower} — {reasoning[:60]}")

        return json.dumps({
            "status": "added",
            "region": [x, y, width, height],
            "method": method_lower,
            "parameters": parameters,
        })


class GetProtectionStatusTool(BaseTool):
    """View current protection status of all elements."""
    name: str = "get_protection_status"
    description: str = (
        "View what protections have been applied to each element. "
        "Shows detection_id, element description, method, bbox, and whether "
        "it was already patched. No arguments needed."
    )
    handle_tool_error: bool = True
    transformations: Any = None
    patches_applied: Any = None

    def _run(self, tool_input: str = "") -> str:
        patched_ids = {p["detection_id"] for p in self.patches_applied}

        lines = []
        for t in self.transformations:
            det_id = t.get("detection_id", "?")
            patched = " [PATCHED]" if det_id in patched_ids else ""
            lines.append(
                f"  {det_id[:12]:12} | {t.get('element', '?')[:35]:35} | "
                f"method={t.get('method', '?'):15} | "
                f"bbox={t.get('bbox', '?')}{patched}"
            )

        return f"Protected elements ({len(self.transformations)}):\n" + "\n".join(lines)


class FinalizeVerificationTool(BaseTool):
    """Finalize the verification check."""
    name: str = "finalize_verification"
    description: str = (
        "Finalize verification after checking all protections. "
        "Call this when all leaks have been patched or no leaks were found. "
        "No arguments needed."
    )
    handle_tool_error: bool = True
    patches_applied: Any = None
    already_finalized: bool = False

    def _run(self, tool_input: str = "") -> str:
        if self.already_finalized:
            return json.dumps({"status": "already_finalized", "message": "Verification already complete."})

        self.already_finalized = True
        return json.dumps({
            "status": "verification_complete",
            "patches_applied": len(self.patches_applied),
            "message": "Verification finalized. Do NOT call any more tools.",
        })
