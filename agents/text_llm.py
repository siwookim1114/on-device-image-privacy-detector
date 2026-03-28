"""
In-process small LLM for text tasks (intent classification + response generation).

Uses llama-cpp-python with Metal (Apple Silicon) acceleration.
No external server needed -- the model is loaded directly in the Python process.
This runs alongside the 30B VLM server (which handles vision tasks on port 8081).

Design: Singleton-style -- instantiated once at startup, shared across all coordinator
calls via NodeContext.text_llm.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional, Type

logger = logging.getLogger(__name__)

# Guard import so module loads even without llama-cpp-python installed
try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None
    _LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. TextLLM will not be available.")


class TextLLM:
    """
    Small in-process LLM for text-only tasks.

    Loads a GGUF model via llama-cpp-python with Metal GPU acceleration.
    Used for:
      - Intent classification (structured JSON output)
      - Response generation (natural language chatbot replies)

    Args:
        model_path: Path to GGUF model file.
        n_ctx: Context window size (2048 sufficient for classification/responses).
        n_gpu_layers: GPU layers to offload (-1 = all, for Metal).
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
    ) -> None:
        if not _LLAMA_CPP_AVAILABLE:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python"
            )

        logger.info("Loading TextLLM model: %s", model_path)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.model_path = model_path
        logger.info("TextLLM ready (%d ctx, %d GPU layers)", n_ctx, n_gpu_layers)

    def call(
        self,
        system_prompt: str,
        user_message: str,
        json_schema: Optional[dict] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """
        Chat completion with optional JSON schema constraint.

        For Qwen3 thinking models, /no_think is appended to suppress
        chain-of-thought reasoning for faster inference.

        Args:
            system_prompt: System message defining the task.
            user_message: User input to classify/respond to.
            json_schema: Optional JSON schema for constrained generation.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = deterministic).

        Returns:
            Generated text content as string.
        """
        # Append /no_think for Qwen3 models to skip reasoning
        effective_user_msg = user_message
        if "qwen3" in self.model_path.lower() or "qwen_3" in self.model_path.lower():
            effective_user_msg = user_message + " /no_think"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": effective_user_msg},
        ]

        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Use JSON schema constrained generation if provided
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": json_schema,
            }

        response = self.llm.create_chat_completion(**kwargs)
        content = response["choices"][0]["message"]["content"] or ""
        # Strip Qwen3 <think>...</think> reasoning tokens from output
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
        return content.strip()

    def call_structured(
        self,
        system_prompt: str,
        user_message: str,
        response_model: Type,
        max_tokens: int = 512,
    ) -> Any:
        """
        Chat completion returning a validated Pydantic model.

        Uses JSON schema from the Pydantic model to constrain generation,
        then validates the output against the model.

        Args:
            system_prompt: System message.
            user_message: User input.
            response_model: Pydantic BaseModel class.
            max_tokens: Maximum tokens to generate.

        Returns:
            Validated Pydantic model instance.

        Raises:
            ValidationError: If LLM output doesn't match schema.
        """
        schema = response_model.model_json_schema()
        raw = self.call(
            system_prompt=system_prompt,
            user_message=user_message,
            json_schema=schema,
            max_tokens=max_tokens,
        )
        return response_model.model_validate_json(raw)
