"""Evaluation models: ground truth annotations and benchmark result dataclasses."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class GroundTruthElement(BaseModel):
    id: str
    element_type: str
    bbox: List[int] = Field(min_length=4, max_length=4)
    severity_gt: str
    should_protect: bool
    consent_status: Optional[str] = None
    person_label: Optional[str] = None
    text_content: Optional[str] = None
    text_type: Optional[str] = None
    screen_state: Optional[str] = None
    object_class: Optional[str] = None
    notes: Optional[str] = None


class ImageAnnotation(BaseModel):
    image_id: str
    image_path: str
    scene_description: str = ""
    elements: List[GroundTruthElement] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def faces(self) -> List[GroundTruthElement]:
        return [e for e in self.elements if e.element_type == "face"]

    @property
    def texts(self) -> List[GroundTruthElement]:
        return [e for e in self.elements if e.element_type == "text"]

    @property
    def objects(self) -> List[GroundTruthElement]:
        return [e for e in self.elements if e.element_type == "object"]

    @property
    def protectable(self) -> List[GroundTruthElement]:
        return [e for e in self.elements if e.should_protect]


class DatasetManifest(BaseModel):
    name: str
    version: str
    description: str = ""
    images: List[ImageAnnotation] = Field(default_factory=list)


@dataclass
class DetectionMetrics:
    element_type: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        return self.true_positives / max(self.true_positives + self.false_positives, 1)

    @property
    def recall(self) -> float:
        return self.true_positives / max(self.true_positives + self.false_negatives, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r) / max(p + r, 1e-9)


@dataclass
class SeverityAccuracy:
    total: int = 0
    correct: int = 0
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)


@dataclass
class ProtectionEfficacy:
    face_reid_rate_before: float = 0.0
    face_reid_rate_after: float = 0.0
    ocr_cer_before: float = 0.0
    ocr_cer_after: float = 0.0
    protection_decision_accuracy: float = 0.0
    false_protection_rate: float = 0.0
    missed_protection_rate: float = 0.0

    @property
    def face_reid_reduction(self) -> float:
        if self.face_reid_rate_before == 0:
            return 0.0
        return 1.0 - (self.face_reid_rate_after / self.face_reid_rate_before)

    @property
    def ocr_recovery_reduction(self) -> float:
        if self.ocr_cer_before == 0:
            return 0.0
        return 1.0 - (self.ocr_cer_after / self.ocr_cer_before)


@dataclass
class LatencyStats:
    stage: str
    samples: List[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return sum(self.samples) / max(len(self.samples), 1)

    @property
    def median_ms(self) -> float:
        s = sorted(self.samples)
        n = len(s)
        if n == 0:
            return 0.0
        mid = n // 2
        return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


@dataclass
class ImageResult:
    image_id: str
    config_name: str
    detection_metrics: Dict[str, DetectionMetrics] = field(default_factory=dict)
    severity_accuracy: SeverityAccuracy = field(default_factory=SeverityAccuracy)
    protection_efficacy: ProtectionEfficacy = field(default_factory=ProtectionEfficacy)
    latency: Dict[str, float] = field(default_factory=dict)
    peak_memory_mb: float = 0.0
    pipeline_output: Any = None


@dataclass
class BenchmarkResults:
    config_name: str
    image_results: List[ImageResult] = field(default_factory=list)
    aggregate_detection: Dict[str, DetectionMetrics] = field(default_factory=dict)
    aggregate_severity: SeverityAccuracy = field(default_factory=SeverityAccuracy)
    aggregate_protection: ProtectionEfficacy = field(default_factory=ProtectionEfficacy)
    latency_stats: Dict[str, LatencyStats] = field(default_factory=dict)
    total_images: int = 0
    successful_images: int = 0


@dataclass
class AblationResults:
    configs: List[str] = field(default_factory=list)
    results: Dict[str, BenchmarkResults] = field(default_factory=dict)


@dataclass
class BaselineResults:
    baselines: List[str] = field(default_factory=list)
    results: Dict[str, BenchmarkResults] = field(default_factory=dict)
