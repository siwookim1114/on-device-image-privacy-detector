"""
Agent 1: Detection Agent
Uses local VLM to reason about images and selectively call detection tools
Only runs necessary detectors based on image content analysis
"""
import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import json
import re

from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from pydantic import BaseModel, Field

from utils.models import (
    DetectionResults, 
    FaceDetection, 
    TextDetection,
    ObjectDetection,
    BoundingBox
)

# Local VLM Wrapper for LangChain
class LocalVLM(LLM):
    """
    Local Vision-Language Model for On-Device Reasoning
    Wraps Phi-3.5-Vision or similar models for Langchain
    """
    model_name: str = "microsoft/Phi-3.5-vision-instruct"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: Any = None
    processor: Any = None
    current_image: Optional[Image.Image] = None

    def __init__(self, config, **kwargs):
        """Initialize Local VLM"""
        super().__init__(**kwargs)
        self.config = config
        self.device = torch.device(config.system.device)
        self.load_model()
    
    def _load_model(self):
        """Load the VLM model"""
        try:
            print(f"Loading VLM: {self.model_name} on {self.device}")

            """Load with 4-bit quantization for memory efficiency"""
            if self.device.type == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit = True,
                    bnb_4bit_compute_dtype = torch.float16,
                    bnb_4bit_use_double_quant = True,
                    bnb_4bit_quant_type = "nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config = quantization_config,
                    trust_remote_code = True,
                    device_map = "auto",
                    torch_dtype = torch.float16
                )
            else:
                # CPU Mode
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code = True,
                    torch_dtype = torch.float32
                )
                self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remotecode = True
            )
            print(f"VLM loaded successfully!")
        except Exception as e:
            print(f"Failed to load VLM: {e}")
            raise
    
    def set_image(self, image: Image.Image):
        """Set the current image for reasoning"""
        self.current_image = image
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Call the VLM with text prompt and current image

        Args:
            prompt: Text prompt
            stop: Stop sequences
        Returns:
            Model response as string
        """
        if self.current_image is None:
            return "Error: No image set for analysis"
        try:
            # Prepare messages for Phi-3.5-Vision
            messages = [
                {
                    "role": "user",
                    "content": f"<|image_1|>\n{prompt}"
                }
            ]
            prompt_text = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = True
            )
            inputs = self.processor(
                prompt_text,
                [self.current_image],
                return_tensor = "pt"
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens = 500,
                    do_sample = False,
                    temperature = 0.0, 
                    eos_token_id = self.processor.tokenizer.eos_token_id
                )
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens = True
            )

            # Apply stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in response:
                        response = response.split(stop_seq)[0]
            return response.strip()

        except Exception as e:
            return f"Error in VLM inference: {e}"
    @property
    def _llm_type(self) -> str:
        return "local_vlm"

