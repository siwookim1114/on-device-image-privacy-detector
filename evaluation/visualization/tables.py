"""LaTeX table generation for paper-ready output."""
from pathlib import Path

def generate_detection_table(results, output_path: str):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Detection accuracy per element type}",
        r"\label{tab:detection}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Element Type & Precision & Recall & F1 \\",
        r"\midrule",
    ]
    for etype, dm in results.aggregate_detection.items():
        if etype == "macro_avg":
            continue
        lines.append(f"{etype.capitalize()} & {dm.precision:.3f} & {dm.recall:.3f} & {dm.f1:.3f} \\\\")

    macro = results.aggregate_detection.get("macro_avg")
    if macro:
        lines.append(r"\midrule")
        lines.append(f"\\textbf{{Macro Avg}} & \\textbf{{{macro.precision:.3f}}} & \\textbf{{{macro.recall:.3f}}} & \\textbf{{{macro.f1:.3f}}} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines))


def generate_ablation_table(ablation_results, output_path: str):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study results}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Configuration & Det F1 & Sev Acc & Prot Acc & Latency (ms) \\",
        r"\midrule",
    ]
    for config_name, res in ablation_results.results.items():
        macro = res.aggregate_detection.get("macro_avg")
        f1 = f"{macro.f1:.3f}" if macro else "-"
        sev = f"{res.aggregate_severity.accuracy:.3f}" if res.aggregate_severity.total > 0 else "-"
        prot = f"{res.aggregate_protection.protection_decision_accuracy:.3f}" if res.aggregate_protection else "-"
        lat = "-"
        if "total_ms" in res.latency_stats:
            lat = f"{res.latency_stats['total_ms'].get('median', 0):.0f}"
        lines.append(f"{config_name} & {f1} & {sev} & {prot} & {lat} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(output_path).write_text("\n".join(lines))


def generate_latency_table(results, output_path: str):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-stage latency breakdown (N=30 runs)}",
        r"\label{tab:latency}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Stage & Median (ms) & P95 (ms) & $\sigma$ (ms) \\",
        r"\midrule",
    ]
    for stage, stats in results.latency_stats.items():
        if isinstance(stats, dict):
            med = stats.get("median", 0)
            p95 = stats.get("p95", 0)
            std = stats.get("std", 0)
        else:
            med = stats.median_ms
            p95 = 0
            std = 0
        lines.append(f"{stage} & {med:.0f} & {p95:.0f} & {std:.0f} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(output_path).write_text("\n".join(lines))


def generate_baseline_table(baseline_results, output_path: str):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Baseline comparison}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Baseline & Det F1 & Latency (ms) \\",
        r"\midrule",
    ]
    for name, res in baseline_results.results.items():
        macro = res.aggregate_detection.get("macro_avg")
        f1 = f"{macro.f1:.3f}" if macro else "-"
        lat = "-"
        lines.append(f"{name} & {f1} & {lat} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(output_path).write_text("\n".join(lines))


def generate_all_tables(results, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generate_detection_table(results, str(out / "table_detection.tex"))
    generate_latency_table(results, str(out / "table_latency.tex"))
