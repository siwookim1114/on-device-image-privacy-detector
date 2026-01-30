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
        self.llm = ChatOllama(
            model=model,
            temperature=0,
            base_url=base_url
        )
        self.current_image: Optional[Image.Image] = None
        print(f"Initialized Ollama VLM: {model}")
        print(f"Make sure the model is pulled: ollama pull {model}")

    def set_image(self, image: Image.Image):
        """Set current image for analysis"""
        self.current_image = image

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
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
                    "image_url": f"data:image/png;base64,{image_b64}"
                }
            ]
        )

        response = self.llm.invoke([message])
        return response.content
