import torch
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
from pathlib import Path
import time
import json

from utils.models import (
    DetectionResults,
    FaceDetection,
    TextDetection,
    ObjectDetection,
    BoundingBox
)
from agents.tools import FaceDetectionTool, TextDetectionTool, ObjectDetectionTool


class DetectionAgent:
    """
    Agent 1: Detection Agent (Vision Specialist)

    Purpose: Detect ALL visual elements in images

    Architecture (Phase 1):
    - ALWAYS runs all detection tools (more reliable)

    Tools:
    - detect_faces: MTCNN for face detection + FaceNet for face embedding extraction
    - detect_text: EasyOCR for text recognition
    - detect_objects: YOLO for object detection
    """
    def __init__(self, config):
        """
        Initialize detection agent with all detection tools

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.system.device)
        self._annotated_image: Optional[Image.Image] = None  # Stores annotated image in memory

        print(f"Initializing Detection Agent on {self.device}")

        # Initialize detection tools
        self.face_tool = FaceDetectionTool(config)   # MTCNN
        self.text_tool = TextDetectionTool(config)   # EasyOCR
        self.object_tool = ObjectDetectionTool(config) # YOLO

    def run(self, image_path: str, create_annotated: bool = True) -> DetectionResults:
        """
        Main detection pipeline = runs ALL detection tools

        Args:
            image_path: Path to input image
            create_annotated: Whether to create annotated image in memory (default: True)
                              Access via get_annotated_image() after run

        Returns:
            DetectionResults with all detections
        """
        start_time = time.time()
        self._annotated_image = None  # Reset from previous run

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

            # Initialize results container
            detections = DetectionResults(image_path=image_path)

            # Stage 1: Run ALL detection tools
            print("\nRunning Detection Tools:")
            print("-" * 40)

            # Run all detections
            self._run_face_detection(image_path, detections)
            self._run_text_detection(image_path, detections)
            self._run_object_detection(image_path, detections)

            # Stage 2: Create annotated image in memory (for downstream agents)
            if create_annotated:
                self._annotated_image = self._create_annotated_image(image_path, detections)
                print(f"  Annotated image created (in memory)")

            # Processing time
            processing_time = (time.time() - start_time) * 1000
            detections.processing_time_ms = processing_time

            # Print Summary
            print(f"\n{'='*60}")
            print(f"Detection Complete")
            print(f"{'='*60}")
            print(f"  Processing time: {processing_time:.2f}ms")
            print(f"  Total detections: {detections.total_detections}")
            print(f"    - Faces: {len(detections.faces)}")
            print(f"    - Text regions: {len(detections.text_regions)}")
            print(f"    - Objects: {len(detections.objects)}")
            print(f"  Annotated image: {'Ready (in memory)' if self._annotated_image else 'Not created'}")
            print(f"{'='*60}\n")

            return detections

        except Exception as e:
            print(f"Detection failed: {e}")
            raise

    def _run_face_detection(self, image_path: str, detections: DetectionResults):
        """Run face detection tool and add results to detections"""
        try:
            result = self.face_tool._run(image_path)
            data = json.loads(result)

            if "error" in data:
                print(f"  Face Detection: {data['error']}")
                return

            if "faces" in data:
                for face_data in data["faces"]:
                    bbox_list = face_data.get("bbox", [0, 0, 0, 0])
                    bbox = BoundingBox(
                        x=bbox_list[0],
                        y=bbox_list[1],
                        width=bbox_list[2],
                        height=bbox_list[3]
                    )
                    face = FaceDetection(
                        bbox=bbox,
                        confidence=face_data.get("confidence", 0.0),
                        size=face_data.get("size", self._classify_size(bbox.width, bbox.height)),
                        clarity="high" if face_data.get("confidence", 0) > 0.95 else "medium",
                        attributes = {
                            "has_landmarks": face_data.get("has_landmarks", False),
                            "has_embedding": face_data.get("has_embedding", False),
                            "embedding": face_data.get("embedding")
                        }
                    )
                    detections.faces.append(face)
                print(f"  Face Detection: Found {len(data['faces'])} faces")
        except Exception as e:
            print(f"  Face Detection failed: {e}")

    def _run_text_detection(self, image_path: str, detections: DetectionResults):
        """Run text detection tool and add results to detections"""
        try:
            result = self.text_tool._run(image_path)
            data = json.loads(result)

            if "error" in data:
                print(f"  Text Detection: {data['error']}")
                return

            if "texts" in data:
                for text_data in data["texts"]:
                    bbox_list = text_data.get("bbox", [0, 0, 0, 0])
                    bbox = BoundingBox(
                        x=bbox_list[0],
                        y=bbox_list[1],
                        width=bbox_list[2],
                        height=bbox_list[3]
                    )
                    text = TextDetection(
                        bbox=bbox,
                        confidence=text_data.get("confidence", 0.0),
                        text_content=text_data.get("text_content", ""),  # Matches tool output
                        text_type=text_data.get("text_type", "general_text"),
                        attributes = {
                            "is_sensitive": text_data.get("is_sensitive", False),
                            "is_pii": text_data.get("is_pii", False),
                            "is_critical": text_data.get("is_critical", False)
                        }
                    )
                    detections.text_regions.append(text)
                pii_count = data.get("pii_count", 0)
                critical_count = data.get("critical_count", 0)
                print(f"  Text Detection: Found {len(data['texts'])} text regions ({pii_count} PII, {critical_count} critical)")
        except Exception as e:
            print(f"  Text Detection failed: {e}")

    def _run_object_detection(self, image_path: str, detections: DetectionResults):
        """Run object detection tool and add results to detections"""
        try:
            result = self.object_tool._run(image_path)
            data = json.loads(result)

            if "error" in data:
                print(f"  Object Detection: {data['error']}")
                return

            if "objects" in data:
                for obj_data in data["objects"]:
                    bbox_list = obj_data.get("bbox", [0, 0, 0, 0])
                    bbox = BoundingBox(
                        x=bbox_list[0],
                        y=bbox_list[1],
                        width=bbox_list[2],
                        height=bbox_list[3]
                    )
                    obj = ObjectDetection(
                        bbox=bbox,
                        confidence = obj_data.get("confidence", 0.0),
                        object_class = obj_data.get("label", "unknown"),  # Matches tool output
                        contains_screen = obj_data.get("contains_screen", False),
                        attributes = {
                            "is_privacy_relevant": obj_data.get("is_privacy_relevant", False),
                            "risk_category": obj_data.get("risk_category", "other")
                        }
                    )
                    detections.objects.append(obj)
                privacy_count = data.get("privacy_relevant_count", 0)
                print(f"  Object Detection: Found {len(data['objects'])} objects ({privacy_count} privacy-relevant)")
        except Exception as e:
            print(f"  Object Detection failed: {e}")

    def _check_overlap(self, bbox: BoundingBox, existing_objects: list, threshold: float = 0.5) -> bool:
        """
        Check if bbox significantly overlaps with existing detections

        Args:
            bbox: BoundingBox to check
            existing_objects: List of ObjectDetection objects
            threshold: threshold to consider as duplicate

        Returns:
            True if overlaps significantly with existing detection
        """
        for obj in existing_objects:
            # Calculating Intersection over Union
            x1 = max(bbox.x, obj.bbox.x)
            y1 = max(bbox.y, obj.bbox.y)
            x2 = min(bbox.x + bbox.width, obj.bbox.x + obj.bbox.width)
            y2 = min(bbox.y + bbox.height, obj.bbox.y + obj.bbox.height)

            # No overlap
            if x2 <= x1 or y2 <= y1:
                continue

            intersection = (x2 - x1) * (y2 - y1)
            area1 = bbox.width * bbox.height
            area2 = obj.bbox.width * obj.bbox.height
            union = area1 + area2 - intersection

            iou = intersection / union if union > 0 else 0
            if iou > threshold:
                return True

        return False

    def _classify_size(self, width: int, height: int):
        """Classify face size"""
        avg = (width + height) / 2
        if avg > 150:
            return "large"
        elif avg > 80:
            return "medium"
        else:
            return "small"
    
    def _create_annotated_image(self, image_path: str, detections: DetectionResults) -> Image.Image:
        """
        Create annotated image with bounding boxes (in memory, no disk write).

        Color coding:
        - Red: Faces (identity exposure risk)
        - Green: Text regions (information disclosure risk)
        - Blue: Objects (context exposure risk)

        Args:
            image_path: Path to original image
            detections: DetectionResults object

        Returns:
            PIL Image with bounding boxes drawn (in memory)
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Draw faces (red boxes)
        for face in detections.faces:
            bbox = face.bbox
            x1, y1 = bbox.x, bbox.y
            x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            label = f"Face {face.confidence:.2f}"
            label_y = max(0, y1 - 20)
            draw.text((x1, label_y), label, fill="red", font=font)

        # Draw text regions (green boxes)
        for text_region in detections.text_regions:
            bbox = text_region.bbox
            x1, y1 = bbox.x, bbox.y
            x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height

            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            # Show text type if classified, otherwise content preview
            if text_region.text_type and text_region.text_type != "general_text":
                label = f"Text: {text_region.text_type}"
            else:
                label = f"Text: {text_region.text_content[:20]}"
            label_y = max(0, y1 - 18)
            draw.text((x1, label_y), label, fill="green", font=small_font)

        # Draw objects (blue boxes)
        for obj in detections.objects:
            bbox = obj.bbox
            x1, y1 = bbox.x, bbox.y
            x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height

            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            label = f"{obj.object_class} {obj.confidence:.2f}"
            label_y = max(0, y1 - 20)
            draw.text((x1, label_y), label, fill="blue", font=font)

        return image

    def get_annotated_image(self) -> Optional[Image.Image]:
        """
        Get the annotated image from the last run.

        Returns:
            PIL Image with bounding boxes, or None if no image processed yet
        """
        return getattr(self, '_annotated_image', None)

    def save_annotated_image(self, output_path: str) -> str:
        """
        Save the annotated image to disk.

        Args:
            output_path: Path to save the image

        Returns:
            Path where image was saved
        """
        if self._annotated_image is None:
            raise ValueError("No annotated image available. Run detection first.")

        self._annotated_image.save(output_path)
        print(f"Annotated image saved: {output_path}")
        return output_path
            