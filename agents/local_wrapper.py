import base64
import requests
from io import BytesIO
from PIL import Image
from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


class VisionLLM:
    """
    Thin wrapper around ChatOllama for vision tasks
    Maintains current image state and injects it into LLM calls
    """
    def __init__(self, model: str = "llava-phi3", base_url: str = "http://localhost:11434"):
        # Enable reasoning capture for thinking models
        is_thinking = "thinking" in model.lower()
        self.llm = ChatOllama(
            model=model,
            temperature=0,
            base_url=base_url,
            reasoning=True if is_thinking else None,
        )
        self.current_image: Optional[Image.Image] = None
        print(f"Initialized Ollama VLM: {model}")
        print(f"Make sure the model is pulled: ollama pull {model}")

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
                    "image_url": f"data:image/jpeg;base64,{image_b64}"
                }
            ]
        )
        
        response = self.llm.invoke([message])
        return response.content
