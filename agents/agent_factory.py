"""
Shared VLM agent factory — eliminates boilerplate duplication across agents.

Provides:
  - get_vlm_config(backend) -> (base_url, model_name)
  - create_vlm(backend) -> (VisionLLM, model_name)
  - resize_for_vlm(image, max_dim) -> resized PIL Image
  - MessageTrimMiddleware (parameterized)
  - build_vlm_agent(vlm, tools, system_prompt, max_iters, ...) -> agent
"""

from typing import List, Tuple

from PIL import Image
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware

from agents.local_wrapper import VisionLLM


# ── Backend configuration ──────────────────────────────────────────

BACKEND_CONFIG = {
    "llama-cpp": {"base_url": "http://localhost:8081"},
    "ollama": {"base_url": "http://localhost:11434"},
    "mlx": {"base_url": "http://localhost:8000"},
}

MODEL_NAMES = {
    "llama-cpp": "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf",
    "ollama": "qwen3-vl:30b-a3b",
    "mlx": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
}


def get_vlm_config(backend: str, default_backend: str = "llama-cpp") -> Tuple[str, str]:
    """Return (base_url, model_name) for the given VLM backend."""
    cfg = BACKEND_CONFIG.get(backend, BACKEND_CONFIG[default_backend])
    base_url = cfg["base_url"]
    model_name = MODEL_NAMES.get(backend, MODEL_NAMES[default_backend])
    return base_url, model_name


def create_vlm(backend: str, default_backend: str = "llama-cpp") -> Tuple[VisionLLM, str]:
    """Create a VisionLLM instance. Returns (vlm, model_name)."""
    base_url, model_name = get_vlm_config(backend, default_backend)
    vlm = VisionLLM(model=model_name, base_url=base_url, backend=backend)
    return vlm, model_name


# ── Image resize utility ───────────────────────────────────────────

def resize_for_vlm(image: Image.Image, max_dim: int = 1024) -> Image.Image:
    """Resize so longest side is at most max_dim. Returns unchanged if already fits."""
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    scale = max_dim / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ── MessageTrimMiddleware ──────────────────────────────────────────

class MessageTrimMiddleware(AgentMiddleware):
    """Trim old messages to keep context within VLM limits.
    Preserves first message (image+prompt) and last `keep` messages."""

    def __init__(self, threshold: int = 12, keep: int = 10):
        self.threshold = threshold
        self.keep = keep

    def wrap_model_call(self, request, handler):
        messages = request.messages
        if len(messages) > self.threshold:
            trimmed = [messages[0]] + messages[-self.keep:]
            return handler(request.override(messages=trimmed))
        return handler(request)


# ── Agent builder ──────────────────────────────────────────────────

def build_vlm_agent(vlm, tools, system_prompt, max_iters,
                     trim_threshold=12, trim_keep=10):
    """Build a LangChain agent with MessageTrim + ModelCallLimit middleware."""
    return create_agent(
        model=vlm.llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            MessageTrimMiddleware(threshold=trim_threshold, keep=trim_keep),
            ModelCallLimitMiddleware(run_limit=max_iters),
        ],
    )
