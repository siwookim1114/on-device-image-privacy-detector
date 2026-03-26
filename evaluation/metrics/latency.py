"""Latency and memory profiling metrics."""
from typing import Dict, List
import numpy as np

def compute_latency_stats(samples: List[float]) -> dict:
    a = np.array(samples, dtype=float)
    n = len(a)
    if n == 0:
        return {"mean": 0, "median": 0, "p95": 0, "p99": 0, "std": 0, "ci95": 0, "n": 0}

    mean = float(np.mean(a))
    median = float(np.median(a))
    p95 = float(np.percentile(a, 95))
    p99 = float(np.percentile(a, 99))
    std = float(np.std(a, ddof=1)) if n > 1 else 0.0

    try:
        from scipy import stats as sp_stats
        ci95 = float(sp_stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n)) if n > 1 else 0.0
    except ImportError:
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0

    return {
        "mean": round(mean, 1),
        "median": round(median, 1),
        "p95": round(p95, 1),
        "p99": round(p99, 1),
        "std": round(std, 1),
        "ci95": round(ci95, 1),
        "n": n,
    }


def aggregate_latency_stats(image_results: list) -> Dict[str, dict]:
    phase_samples: Dict[str, List[float]] = {}
    for result in image_results:
        latency = result.latency if hasattr(result, "latency") else {}
        for stage, ms in latency.items():
            phase_samples.setdefault(stage, []).append(ms)

    return {stage: compute_latency_stats(samples) for stage, samples in phase_samples.items()}


def compute_memory_stats(peak_rss_samples: List[float]) -> dict:
    if not peak_rss_samples:
        return {"mean_mb": 0, "peak_mb": 0}
    return {
        "mean_mb": round(float(np.mean(peak_rss_samples)), 1),
        "peak_mb": round(float(np.max(peak_rss_samples)), 1),
    }
