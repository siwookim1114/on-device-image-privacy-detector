import base64
from io import BytesIO
from PIL import Image
from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


class VisionLLM:
    """
    Thin wrapper around local VLM backends for vision tasks.
    Supports llama-cpp (llama-server), MLX (vllm-mlx), and Ollama backends.
    Maintains current image state and injects it into LLM calls.
    """
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        backend: str = None,
    ):
        self.backend = backend
        is_thinking = "thinking" in model.lower()

        if backend == "llama-cpp":
            # llama-server (llama.cpp) serves OpenAI-compatible API
            # Supports VLM + native tool calling via --jinja
            api_base = base_url.rstrip("/")
            if not api_base.endswith("/v1"):
                api_base += "/v1"
            self.llm = ChatOpenAI(
                model=model,
                base_url=api_base,
                api_key="not-needed",
                temperature=0,
                streaming=True,
            )
            print(f"Initialized llama-server VLM: {model}")
            print(f"Make sure llama-server is running at {base_url}")
        elif backend == "mlx":
            # vllm-mlx serves OpenAI-compatible API
            api_base = base_url.rstrip("/")
            if not api_base.endswith("/v1"):
                api_base += "/v1"
            self.llm = ChatOpenAI(
                model=model,
                base_url=api_base,
                api_key="not-needed",
                temperature=0,
                streaming=True,
            )
            print(f"Initialized MLX VLM: {model}")
            print(f"Make sure vllm-mlx is serving at {base_url}")
        else:  # ollama
            self.llm = ChatOllama(
                model=model,
                temperature=0,
                base_url=base_url,
                reasoning=True if is_thinking else None,
            )
            print(f"Initialized Ollama VLM: {model}")
            print(f"Make sure the model is pulled: ollama pull {model}")

        self.current_image: Optional[Image.Image] = None

    def set_image(self, image: Image.Image):
        """Set current image for analysis"""
        self.current_image = image

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64.

        Encodes as JPEG for smaller base64 payload (vs PNG).
        No resizing — preserves full resolution for text readability.
        """
        buffered = BytesIO()
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def invoke(self, prompt: str) -> str:
        """Call LLM with current image and prompt"""
        if self.current_image is None:
            # No image - call as text-only
            return self.llm.invoke(prompt).content

        # Prepare message with image
        image_b64 = self._image_to_base64(self.current_image)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                }
            ]
        )

        response = self.llm.invoke([message])
        return response.content
