"""Dataset loader for benchmark evaluation."""
import json
from pathlib import Path
from typing import List, Optional
from .models import ImageAnnotation, DatasetManifest

class BenchmarkDataset:
    def __init__(self, dataset_root: str):
        self.root = Path(dataset_root)
        self.manifest: Optional[DatasetManifest] = None
        self.annotations: List[ImageAnnotation] = []
        self._load()

    def _load(self):
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            raw = json.loads(manifest_path.read_text())
            self.manifest = DatasetManifest(**raw)
            self.annotations = self.manifest.images
        else:
            ann_dir = self.root / "annotations"
            if ann_dir.exists():
                for f in sorted(ann_dir.glob("*.json")):
                    raw = json.loads(f.read_text())
                    self.annotations.append(ImageAnnotation(**raw))

    def __len__(self) -> int:
        return len(self.annotations)

    def __iter__(self):
        return iter(self.annotations)

    def filter_by_category(self, category: str) -> List[ImageAnnotation]:
        return [a for a in self.annotations if a.metadata.get("category") == category]

    def filter_by_element_type(self, element_type: str) -> List[ImageAnnotation]:
        return [a for a in self.annotations
                if any(e.element_type == element_type for e in a.elements)]

    def resolve_image_path(self, annotation: ImageAnnotation) -> str:
        p = Path(annotation.image_path)
        if p.is_absolute():
            return str(p)
        return str(self.root / p)

    def validate(self) -> List[str]:
        errors = []
        for ann in self.annotations:
            img_path = self.resolve_image_path(ann)
            if not Path(img_path).exists():
                errors.append(f"{ann.image_id}: image not found at {img_path}")
            for elem in ann.elements:
                if len(elem.bbox) != 4:
                    errors.append(f"{ann.image_id}/{elem.id}: bbox must have 4 values")
                if elem.severity_gt not in ("critical", "high", "medium", "low"):
                    errors.append(f"{ann.image_id}/{elem.id}: invalid severity '{elem.severity_gt}'")
        return errors
