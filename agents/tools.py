# Detection tools
import torch
from PIL import Image
from typing import List, Dict, Any, Optional
import json
from facenet_pytorch import MTCNN
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
    device: torch.device = torch.device("cpu")
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
        
