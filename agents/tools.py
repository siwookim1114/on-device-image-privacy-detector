# Import required utils
import torch
from PIL import Image
from typing import Any
import json
import numpy as np

# Detector Libraries
from facenet_pytorch import MTCNN
import easyocr
from ultralytics import YOLO

from langchain.tools import BaseTool

class FaceDetectionTool(BaseTool):
    """Tool for detecting faces in images"""
    name: str = "detect_faces"
    description: str = (
        "Detects human faces in the image. "
        "Returns list of face locations with confidence scores. "
        "Use this tool when you see people or human faces in the image."
    )
    detector: Any = None
    device: torch.device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    config: Any = None

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.device = torch.device(config.system.device)
        self._init_detector()
    
    def _init_detector(self):
        """Initialize face detector"""
        try:
            self.detector = MTCNN(
                image_size = 160,    # Output face size
                margin = 0,          # Extra pixels around face
                min_face_size = 20,  # Ignore tiny faces
                thresholds = [0.6, 0.7, 0.7], # Detection confidence thresholds
                factor = 0.709,               # Image pyramid scale factor
                device = self.device,         # CPU or GPU
                keep_all = True               # Return ALL faces, not just the best one
            )
            print("Face detection tool ready")
        
        except Exception as e:
            print(f"Face detector failed: {e}")
            self.detector = None
    
    def _run(self, image_path: str) -> str:   # Runs when the agent calls the tool -> Need to be _run for BaseTool
        """Run face detection"""
        if self.detector is None:
            return "Face detector not available"
        
        try:
            image = Image.open(image_path).convert("RGB")
            boxes, probs, landmarks = self.detector.detect(image, landmarks = True)

            if boxes is None:
                return json.dumps({"faces": [], "count": 0})
            
            faces = []
            for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                if prob < 0.7:
                    continue
                x1, y1, x2, y2 = map(int, box)
                faces.append({
                    "id": i,
                    "bbox": [x1, y1, x2-x1, y2-y1],
                    "confidence": float(prob),
                    "has_landmarks": landmark is not None
                })
            return json.dumps({"faces": faces, "count": len(faces)})
        
        except Exception as e:
            return json.dumps({"error": str(e), "faces": [], "count": 0})

class TextDetectionTool(BaseTool):
    """Tool for detecting and recognizing text in images"""
    name: str = "detect_text"
    description: str = (
        "Detects and recognizes text in the image using OCR. "
        "Returns detected text content and locations. "
        "Use this tool when you see text, numbers, signs, or documents in the image."
    )
    detector: Any = None
    config: Any = None

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self._init_detector()
    
    def _init_detector(self):
        """Initialize text detector"""
        try:
            self.detector = easyocr.Reader(
                ["en"],
                gpu = self.config.system.device == "cuda",
                verbose = False
            )
            print("Text detection tool ready")

        except Exception as e:
            print(f"Text detector failed: {e}")
            self.detector = None
    
    def _run(self, image_path: str) -> str:
        """Run text detection"""
        if self.detector is None:
            return "Text detector not available"
        
        try:
            image = Image.open(image_path)
            img_array = np.array(image)
            results = self.detector.readtext(img_array)
            texts = []
            for detection in results:
                bbox_points, text, confidence = detection

                if confidence < 0.5:
                    continue

                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]

                texts.append({
                    "text": text,
                    "bbox": [
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords) - min(x_coords)),
                        int(max(y_coords) - min(y_coords))
                    ],
                    "confidence": float(confidence)
                })
            return json.dumps({"texts": texts, "count": len(texts)})
    
        except Exception as e:
            return json.dumps({"error": str(e), "texts": [], "count": 0})
    
class ObjectDetectionTool(BaseTool):
    """Tool for detecting objects in images"""
    name: str = "detect_objects"
    description: str = (
        "Detects objects in the image like cars, laptops, phones, screens, etc . "
        "Returns list of detected objects with locations. "
        "Use this tool when you see vehicles, electronic devices, or any other items or objects."
    )

    detector: Any = None
    device: torch.device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    config: Any = None

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.device = torch.device(config.system.device)
        self._init_detector()
    
    def _init_detector(self):
        """Initialize object detector"""
        try:
            self.detector = YOLO("yolov8n.pt")
            self.detector.to(self.device)
            print("Object detection tool ready")
        except Exception as e:
            print(f"Object detector failed: {e}")
            self.detector = None
    
    def _run(self, image_path: str) -> str:
        """Run object detection"""
        if self.detector is None:
            return "Object detector not available"
        
        try:
            results = self.detector.predict(
                image_path,
                conf = 0.5,
                verbose = False
            )
            objects = []
            for result in results:
                boxes = result.boxes

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]

                    # Filter privacy-relevant objects
                    privacy_objects = {
                        "laptop", "tv", "cell phone", "monitor", "car",
                        "truck", "bus", "keyboard", "book"
                    }
                    if class_name.lower() in privacy_objects:
                        objects.append({
                            "class": class_name,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "confidence": float(box.conf[0])
                        })
            
            return json.dumps({"objects": objects, "count": len(objects)})

        except Exception as e:
            return json.dumps({"error": str(e), "objects": [], "count": 0})
    

        