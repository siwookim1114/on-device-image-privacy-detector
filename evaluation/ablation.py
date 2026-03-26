"""Ablation study configurations."""
from agents.pipeline import PipelineConfig

ABLATION_CONFIGS = {
    "full_system": PipelineConfig(),
    "phase1_only": PipelineConfig(fallback_only=True),
    "no_vlm": PipelineConfig(
        enable_vlm_review=False,
    ),
    "no_sam": PipelineConfig(
        enable_sam=False,
    ),
}


