"""Matplotlib figure generation for paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

COLORS = {
    "critical": "#EF4444",
    "high": "#F97316",
    "medium": "#EAB308",
    "low": "#22C55E",
}

ACM_STYLE = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (3.5, 2.5),
    "figure.dpi": 150,
}


def plot_detection_f1_by_type(results_dict: dict, output_path: str):
    plt.rcParams.update(ACM_STYLE)
    fig, ax = plt.subplots()

    configs = list(results_dict.keys())
    element_types = ["face", "text", "screen"]
    x = range(len(element_types))
    width = 0.8 / max(len(configs), 1)

    for i, config in enumerate(configs):
        res = results_dict[config]
        f1s = [res.aggregate_detection.get(et, type("", (), {"f1": 0})).f1 for et in element_types]
        ax.bar([xi + i * width for xi in x], f1s, width, label=config, alpha=0.85)

    ax.set_xticks([xi + width * (len(configs) - 1) / 2 for xi in x])
    ax.set_xticklabels([t.capitalize() for t in element_types])
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_latency_breakdown(results, output_path: str):
    plt.rcParams.update(ACM_STYLE)
    fig, ax = plt.subplots(figsize=(5, 3))

    stages = []
    medians = []
    for stage, stats in results.latency_stats.items():
        if stage == "total_ms":
            continue
        stages.append(stage.replace("_ms", "").replace("_", "\n"))
        med = stats.get("median", 0) if isinstance(stats, dict) else stats.median_ms
        medians.append(med)

    bars = ax.barh(range(len(stages)), medians, color="#3B82F6", alpha=0.85)
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    ax.set_xlabel("Latency (ms)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_severity_confusion(confusion_matrix: dict, output_path: str):
    plt.rcParams.update(ACM_STYLE)
    levels = ["critical", "high", "medium", "low"]
    matrix = [[confusion_matrix.get(gt, {}).get(pred, 0) for pred in levels] for gt in levels]

    fig, ax = plt.subplots(figsize=(3.5, 3))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([l.capitalize() for l in levels], rotation=45)
    ax.set_yticklabels([l.capitalize() for l in levels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")

    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_narrowing_ratio(narrowing_data: dict, output_path: str):
    plt.rcParams.update(ACM_STYLE)
    fig, ax = plt.subplots()

    stages = ["Detected", "Requires\nProtection", "Executed"]
    counts = [
        narrowing_data.get("detected", 0),
        narrowing_data.get("requires_protection", 0),
        narrowing_data.get("executed", 0),
    ]

    colors = ["#3B82F6", "#F97316", "#22C55E"]
    ax.bar(stages, counts, color=colors, alpha=0.85)
    ax.set_ylabel("Element Count")

    ratio = narrowing_data.get("narrowing_ratio", 0)
    ax.set_title(f"Progressive Narrowing (ratio: {ratio:.2f})")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_reid_reduction(reid_data: dict, output_path: str):
    plt.rcParams.update(ACM_STYLE)
    fig, ax = plt.subplots()

    methods = list(reid_data.keys())
    before = [reid_data[m].get("before_rate", 1.0) for m in methods]
    after = [reid_data[m].get("after_rate", 0.0) for m in methods]

    x = range(len(methods))
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], before, width, label="Before", color="#EF4444", alpha=0.8)
    ax.bar([xi + width / 2 for xi in x], after, width, label="After", color="#22C55E", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in methods])
    ax.set_ylabel("Re-ID Rate")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def generate_all_plots(results, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if results.latency_stats:
        plot_latency_breakdown(results, str(out / "fig_latency.pdf"))
