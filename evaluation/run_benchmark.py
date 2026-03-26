"""CLI entry point for the evaluation suite.

Usage:
    conda run -n lab_env python -m evaluation.run_benchmark --mode full
    conda run -n lab_env python -m evaluation.run_benchmark --mode ablation
    conda run -n lab_env python -m evaluation.run_benchmark --mode baselines
    conda run -n lab_env python -m evaluation.run_benchmark --mode all
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.dataset import BenchmarkDataset
from evaluation.benchmark import BenchmarkRunner

def main():
    parser = argparse.ArgumentParser(description="Evaluation Suite for Privacy Detector")
    parser.add_argument("--dataset", default="evaluation/data/benchmark_v1",
                        help="Path to benchmark dataset root")
    parser.add_argument("--mode", choices=["full", "ablation", "baselines", "all"],
                        default="full", help="Evaluation mode")
    parser.add_argument("--output", default="evaluation/data/results",
                        help="Output directory")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Specific ablation config names")
    args = parser.parse_args()

    dataset = BenchmarkDataset(args.dataset)
    errors = dataset.validate()
    if errors:
        print("Dataset validation errors:")
        for e in errors:
            print(f"  - {e}")
        if not dataset.annotations:
            print("No valid annotations found. Exiting.")
            return

    print(f"Dataset: {len(dataset)} images loaded")
    runner = BenchmarkRunner(output_root=args.output)

    if args.mode in ("full", "all"):
        print("\n=== Full System Evaluation ===")
        results = runner.run(dataset, "full_system")
        runner.export(results)
        print(f"Done: {results.successful_images}/{results.total_images} successful")

    if args.mode in ("ablation", "all"):
        print("\n=== Ablation Study ===")
        ablation = runner.run_ablation(dataset, config_names=args.configs)
        for name, res in ablation.results.items():
            runner.export(res, f"{args.output}/{name}")

    if args.mode in ("baselines", "all"):
        print("\n=== Baseline Comparisons ===")
        baselines = runner.run_baselines(dataset)
        for name, res in baselines.results.items():
            runner.export(res, f"{args.output}/baseline_{name}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
