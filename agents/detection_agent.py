import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import json
import re
import base64
from io import BytesIO

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from utils.models import (
    DetectionResults,
    FaceDetection,
    TextDetection,
    ObjectDetection,
    BoundingBox
)
from agents.tools import FaceDetectionTool, TextDetectionTool, ObjectDetectionTool
from agents.local_wrapper import VisionLLM


class DetectionAgent:
    """
    Agent 1: Detection Agent
    Uses local VLM to reason about images and selectively call detection tools
    Only runs necessary detectors based on image content analysis
    """
    def __init__(self, config):
        """
        Initialize detection agent

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.system.device)

        print(f"Initializing Detection Agent on {self.device}")

        # Initialize VLM using ChatOllama
        self.vlm = VisionLLM(
            model="llava-phi3",
            base_url="http://localhost:11434"
        )

        # Initialize detection tools
        self.tools = self._init_tools()

        # Create agent
        self.agent = self._create_agent()

    def _init_tools(self):
        """
        Initialize all detection tools
        Face Detection, Text Detection, Object Detection
        """
        tools = [
            FaceDetectionTool(self.config),
            TextDetectionTool(self.config),
            ObjectDetectionTool(self.config)
        ]
        return tools

    def _analyze_image(self, image: Image.Image) -> str:
        """
        Stage 1: Force detailed image description before tool selection
        Returns a structured description of what's in the image
        """
        analysis_prompt = """Analyze this image carefully and describe what you see in detail.

You MUST answer these specific questions:

1. PEOPLE: Are there any people, faces, or human figures visible?
   - How many? Where are they positioned?

2. TEXT: Is there ANY text, numbers, documents, signs, labels, sticky notes,
   papers, screens with text, handwriting, or printed text visible?
   - What kind of text? Where is it located?

3. OBJECTS: Are there any of these objects visible?
   - Electronics: laptops, phones, tablets, monitors, TVs, keyboards
   - Vehicles: cars, trucks, buses
   - Documents: books, papers, cards
   - Where are they located?

Be thorough - it's better to mention something uncertain than to miss it.
Format your response as:
PEOPLE: [description or "None visible"]
TEXT: [description or "None visible"]
OBJECTS: [description or "None visible"]"""

        try:
            response = self.vlm.invoke(analysis_prompt)
            print(f"\n{'='*60}")
            print("Stage 1 - Image Analysis:")
            print(f"{'='*60}")
            print(response)
            print(f"{'='*60}\n")
            return response
        except Exception as e:
            print(f"Image analysis failed: {e}")
            return "Analysis failed - call all tools to be safe"

    def _create_agent(self):
        """Create the React Agent"""
        template = """
        You are an intelligent image analysis agent for privacy protection.

        Your task: Based on the pre-analysis, decide which detection tools to use.

        Available tools:
        {tools}

        Tool names: {tool_names}

        PRE-ANALYSIS OF IMAGE:
        {image_analysis}

        DECISION RULES (based on pre-analysis above):
        - If PEOPLE section mentions any people, faces, or humans -> MUST call detect_faces
        - If TEXT section mentions any text, numbers, documents, papers, signs, sticky notes, or writing -> MUST call detect_text
        - If OBJECTS section mentions laptops, phones, monitors, vehicles, electronics, or books -> MUST call detect_objects
        - When UNCERTAIN about any category -> call the tool (better safe than sorry)
        - Only skip a tool if the pre-analysis clearly states "None visible" for that category

        RESPONSE FORMAT:
        Thought: [Based on pre-analysis, which tools are needed]
        Action: [Tool name]
        Action Input: [Image path]
        Observation: [Tool result]
        ... (repeat for each needed tool)
        Thought: I have completed all necessary detections
        Final Answer: [Summary of all detections]

        Current task:
        Question: {input}

        {agent_scratchpad}
        """
        prompt = PromptTemplate(
            template = template,
            input_variables = ["input", "agent_scratchpad", "image_analysis"],
            partial_variables = {
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )

        # Create React Agent
        agent = create_react_agent(
            llm = self.vlm.llm,  # Use the underlying ChatOllama
            tools = self.tools,
            prompt = prompt
        )

        # Create executor
        agent_executor = AgentExecutor(
            agent = agent,
            tools = self.tools,
            verbose = True,
            max_iterations = 10,
            handle_parsing_errors = True,
            return_intermediate_steps = True
        )
        return agent_executor

    def run(self, image_path: str):
        """
        Main detection method with tool selection

        Args:
            image_path: Path to input image
        Returns:
            DetectionResults with all detections
        """
        start_time = time.time()
        try:
            print(f"=" * 60)
            print(f"Processing: {Path(image_path).name}")
            print(f"=" * 60 + "\n")

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Resize if needed
            max_size = self.config.pipeline.max_image_size
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.LANCZOS)
                print(f"Resized to {new_size}\n")

            # Set image in VLM
            self.vlm.set_image(image)

            # Stage 1: Get detailed image analysis
            image_analysis = self._analyze_image(image)

            # Stage 2: Run agent with pre-analysis to guide tool selection
            result = self.agent.invoke({
                "input": f"Detect privacy-relevant elements in the image at {image_path}. "
                         f"Use the pre-analysis to decide which tools to call.",
                "image_analysis": image_analysis
            })

            # Parse agent output
            detections = self._parse_agent_output(result, image_path)
            processing_time = (time.time() - start_time) * 1000
            detections.processing_time_ms = processing_time

            print(f"\n{'='*60}")
            print(f"  Detection complete in {processing_time:.2f}ms")
            print(f"  Total: {detections.total_detections} elements")
            print(f"  Faces: {len(detections.faces)}")
            print(f"  Text: {len(detections.text_regions)}")
            print(f"  Objects: {len(detections.objects)}")
            print(f"{'='*60}\n")

            return detections

        except Exception as e:
            print(f"Detection failed: {e}")
            raise

    def _parse_agent_output(self, result: Dict, image_path: str):
        """Parse agent output from intermediate tool steps into DetectionResults"""
        detections = DetectionResults(image_path = image_path)

        # Parse structured data directly from intermediate tool steps
        for action, observation in result.get("intermediate_steps", []):
            try:
                data = json.loads(observation)
            except (json.JSONDecodeError, TypeError):
                continue

            if action.tool == "detect_faces" and "faces" in data:
                for face_data in data["faces"]:
                    bbox_list = face_data.get("bbox", [0, 0, 0, 0])
                    bbox = BoundingBox(
                        x = bbox_list[0],
                        y = bbox_list[1],
                        width = bbox_list[2],
                        height = bbox_list[3]
                    )
                    face = FaceDetection(
                        bbox = bbox,
                        confidence = face_data.get("confidence", 0.0),
                        size = self._classify_size(bbox.width, bbox.height),
                        clarity = "high" if face_data.get("confidence", 0) > 0.9 else "medium"
                    )
                    detections.faces.append(face)

            elif action.tool == "detect_text" and "texts" in data:
                for text_data in data["texts"]:
                    bbox_list = text_data.get("bbox", [0, 0, 0, 0])
                    bbox = BoundingBox(
                        x = bbox_list[0],
                        y = bbox_list[1],
                        width = bbox_list[2],
                        height = bbox_list[3]
                    )
                    text = TextDetection(
                        bbox = bbox,
                        confidence = text_data.get("confidence", 0.0),
                        text_content = text_data.get("text", ""),
                        text_type = self._classify_text(text_data.get("text", ""))
                    )
                    detections.text_regions.append(text)

            elif action.tool == "detect_objects" and "objects" in data:
                for obj_data in data["objects"]:
                    bbox_list = obj_data.get("bbox", [0, 0, 0, 0])
                    bbox = BoundingBox(
                        x = bbox_list[0],
                        y = bbox_list[1],
                        width = bbox_list[2],
                        height = bbox_list[3]
                    )
                    obj = ObjectDetection(
                        bbox = bbox,
                        confidence = obj_data.get("confidence", 0.0),
                        object_class = obj_data.get("class", "unknown"),
                        contains_screen = "monitor" in obj_data.get("class", "").lower() or "laptop" in obj_data.get("class", "").lower()
                    )
                    detections.objects.append(obj)

        return detections

    def _classify_size(self, width: int, height: int):
        """Classify face size"""
        avg = (width + height) / 2
        if avg > 150:
            return "large"
        elif avg > 80:
            return "medium"
        else:
            return "small"

    def _classify_text(self, text: str):
        """Classify text type"""
        if re.search(r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}", text):
            return "phone_number"
        elif re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text):
            return "email"
        elif re.search(r"\d{3}-\d{2}-\d{4}", text):
            return "ssn"
        else:
            return "general_text"
