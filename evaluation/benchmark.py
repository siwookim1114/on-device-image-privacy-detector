"""BenchmarkRunner: drives pipeline over dataset and computes metrics."""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .models import (
    BenchmarkResults, ImageResult, AblationResults, BaselineResults,
    DetectionMetrics, SeverityAccuracy, ProtectionEfficacy,
)
from .dataset import BenchmarkDataset

class BenchmarkRunner:
    def __init__(self, output_root: str = "evaluation/data/results"):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        dataset: BenchmarkDataset,
        config_name: str = "full_system",
        pipeline_config=None,
    ) -> BenchmarkResults:
        from agents.pipeline import PipelineOrchestrator, PipelineConfig
        from .metrics.detection import compute_detection_metrics, compute_progressive_narrowing
        from .metrics.risk_accuracy import compute_severity_accuracy
        from .metrics.protection import compute_protection_decision_accuracy
        from .metrics.latency import aggregate_latency_stats

        if pipeline_config is None:
            from .ablation import ABLATION_CONFIGS
            pipeline_config = ABLATION_CONFIGS.get(config_name, PipelineConfig())

        orc = PipelineOrchestrator(config=pipeline_config)
        results = BenchmarkResults(config_name=config_name)

        try:
            for ann in dataset:
                img_path = dataset.resolve_image_path(ann)
                print(f"  [{config_name}] {ann.image_id}...")

                try:
                    output = orc.run(img_path)
                    latency = getattr(output, "phase_timings", {})
                    if not latency:
                        latency = {"total_ms": output.total_time_ms}

                    result = ImageResult(
                        image_id=ann.image_id,
                        config_name=config_name,
                        pipeline_output=output,
                        latency=latency,
                    )

                    if output.success and output.risk_analysis:
                        result.detection_metrics = compute_detection_metrics(
                            output.risk_analysis.risk_assessments,
                            ann.elements,
                            iou_threshold=0.5,
                        )
                        sev_result = compute_severity_accuracy(
                            output.risk_analysis.risk_assessments,
                            ann.elements,
                        )
                        result.severity_accuracy = SeverityAccuracy(
                            total=sev_result["total"],
                            correct=sev_result["correct"],
                            confusion=sev_result.get("confusion_matrix", {}),
                        )
                        pda = compute_protection_decision_accuracy(output, ann.elements)
                        result.protection_efficacy = ProtectionEfficacy(
                            protection_decision_accuracy=pda["accuracy"],
                            false_protection_rate=pda["false_protection_rate"],
                            missed_protection_rate=pda["missed_protection_rate"],
                        )

                    results.image_results.append(result)
                    results.successful_images += 1 if output.success else 0
                except Exception as e:
                    print(f"    ERROR: {e}")
                    results.image_results.append(
                        ImageResult(image_id=ann.image_id, config_name=config_name)
                    )
                results.total_images += 1
        finally:
            orc.close()

        results.aggregate_detection = self._aggregate_detection(results.image_results)
        results.aggregate_severity = self._aggregate_severity(results.image_results)
        results.aggregate_protection = self._aggregate_protection(results.image_results)
        results.latency_stats = {k: v for k, v in aggregate_latency_stats(results.image_results).items()}

        return results

    def run_ablation(
        self,
        dataset: BenchmarkDataset,
        config_names: Optional[List[str]] = None,
    ) -> AblationResults:
        from .ablation import ABLATION_CONFIGS

        if config_names is None:
            config_names = list(ABLATION_CONFIGS.keys())

        ablation = AblationResults(configs=config_names)
        for name in config_names:
            print(f"\n{'=' * 50}")
            print(f"Ablation: {name}")
            print(f"{'=' * 50}")
            ablation.results[name] = self.run(dataset, config_name=name)

        return ablation

    def run_baselines(
        self,
        dataset: BenchmarkDataset,
        baseline_names: Optional[List[str]] = None,
    ) -> BaselineResults:
        from .baselines import get_baseline_runner

        if baseline_names is None:
            baseline_names = ["blur_all", "face_only", "phase1_only"]

        br = BaselineResults(baselines=baseline_names)
        for name in baseline_names:
            print(f"\n{'=' * 50}")
            print(f"Baseline: {name}")
            print(f"{'=' * 50}")
            runner = get_baseline_runner(name)
            br.results[name] = runner(dataset, self.output_root / name)

        return br

    def export(self, results: BenchmarkResults, output_dir: Optional[str] = None):
        out = Path(output_dir) if output_dir else self.output_root / results.config_name
        out.mkdir(parents=True, exist_ok=True)

        data = {
            "config_name": results.config_name,
            "total_images": results.total_images,
            "successful_images": results.successful_images,
            "aggregate_detection": {
                k: {"precision": v.precision, "recall": v.recall, "f1": v.f1}
                for k, v in results.aggregate_detection.items()
            },
        }

        with open(out / "results.json", "w") as f:
            json.dump(data, f, indent=2, default=str)

        try:
            from .visualization.tables import generate_all_tables
            generate_all_tables(results, str(out))
        except ImportError:
            pass

        try:
            from .visualization.plots import generate_all_plots
            generate_all_plots(results, str(out))
        except ImportError:
            pass

    def _aggregate_detection(self, results: List[ImageResult]) -> Dict[str, DetectionMetrics]:
        agg: Dict[str, DetectionMetrics] = {}
        for r in results:
            for etype, dm in r.detection_metrics.items():
                if etype not in agg:
                    agg[etype] = DetectionMetrics(element_type=etype)
                agg[etype].true_positives += dm.true_positives
                agg[etype].false_positives += dm.false_positives
                agg[etype].false_negatives += dm.false_negatives
        return agg

    def _aggregate_severity(self, results: List[ImageResult]) -> SeverityAccuracy:
        agg = SeverityAccuracy()
        for r in results:
            agg.total += r.severity_accuracy.total
            agg.correct += r.severity_accuracy.correct
            for gt_sev, pred_dict in r.severity_accuracy.confusion.items():
                if gt_sev not in agg.confusion:
                    agg.confusion[gt_sev] = {}
                for pred_sev, count in pred_dict.items():
                    agg.confusion[gt_sev][pred_sev] = agg.confusion[gt_sev].get(pred_sev, 0) + count
        return agg

    def _aggregate_protection(self, results: List[ImageResult]) -> ProtectionEfficacy:
        n = len([r for r in results if r.pipeline_output and getattr(r.pipeline_output, 'success', False)])
        if n == 0:
            return ProtectionEfficacy()
        return ProtectionEfficacy(
            protection_decision_accuracy=sum(r.protection_efficacy.protection_decision_accuracy for r in results) / n,
            false_protection_rate=sum(r.protection_efficacy.false_protection_rate for r in results) / n,
            missed_protection_rate=sum(r.protection_efficacy.missed_protection_rate for r in results) / n,
        )
