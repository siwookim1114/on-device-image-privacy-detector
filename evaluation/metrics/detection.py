"""Detection metrics: IoU-based P/R/F1 per element type."""
from typing import Dict, List, Optional

def compute_iou(bbox1: list, bbox2: list) -> float:
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    w1, h1 = max(w1, 0), max(h1, 0)
    w2, h2 = max(w2, 0), max(h2, 0)
    if w1 * h1 + w2 * h2 == 0:
        return 1.0 if (x1 == x2 and y1 == y2) else 0.0
    xi = max(x1, x2)
    yi = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi) * max(0, yi2 - yi)
    union = w1 * h1 + w2 * h2 - inter
    return inter / max(union, 1e-9)


def compute_detection_metrics(
    predictions: list,
    ground_truth: list,
    iou_threshold: float = 0.5,
) -> Dict[str, dict]:
    from evaluation.models import DetectionMetrics

    element_types = set()
    for gt in ground_truth:
        etype = gt.element_type if hasattr(gt, "element_type") else gt.get("element_type", "")
        element_types.add(etype)
    for pred in predictions:
        etype = pred.element_type if hasattr(pred, "element_type") else pred.get("element_type", "")
        element_types.add(etype)

    results = {}
    for etype in element_types:
        gt_items = [g for g in ground_truth
                    if (g.element_type if hasattr(g, "element_type") else g.get("element_type")) == etype]
        pred_items = [p for p in predictions
                      if (p.element_type if hasattr(p, "element_type") else p.get("element_type")) == etype]

        gt_bboxes = [g.bbox if hasattr(g, "bbox") else g.get("bbox", [0, 0, 0, 0]) for g in gt_items]
        pred_bboxes = [_get_bbox(p) for p in pred_items]

        matched_gt = set()
        matched_pred = set()

        pairs = []
        for pi, pb in enumerate(pred_bboxes):
            for gi, gb in enumerate(gt_bboxes):
                iou = compute_iou(list(pb), list(gb))
                if iou >= iou_threshold:
                    pairs.append((iou, pi, gi))
        pairs.sort(key=lambda x: -x[0])

        for _, pi, gi in pairs:
            if pi not in matched_pred and gi not in matched_gt:
                matched_pred.add(pi)
                matched_gt.add(gi)

        tp = len(matched_pred)
        fp = len(pred_items) - tp
        fn = len(gt_items) - len(matched_gt)
        results[etype] = DetectionMetrics(element_type=etype, true_positives=tp, false_positives=fp, false_negatives=fn)

    # Micro-average (sum raw counts across types)
    all_tp = sum(m.true_positives for m in results.values())
    all_fp = sum(m.false_positives for m in results.values())
    all_fn = sum(m.false_negatives for m in results.values())
    results["micro_avg"] = DetectionMetrics(element_type="micro_avg", true_positives=all_tp, false_positives=all_fp, false_negatives=all_fn)

    # Macro-average (average per-type F1 scores)
    type_f1s = [m.f1 for m in results.values() if m.element_type not in ("micro_avg", "macro_avg")]
    macro_f1 = sum(type_f1s) / max(len(type_f1s), 1)
    macro_p = sum(m.precision for m in results.values() if m.element_type not in ("micro_avg", "macro_avg")) / max(len(type_f1s), 1)
    macro_r = sum(m.recall for m in results.values() if m.element_type not in ("micro_avg", "macro_avg")) / max(len(type_f1s), 1)
    results["macro_avg"] = DetectionMetrics(element_type="macro_avg", true_positives=all_tp, false_positives=all_fp, false_negatives=all_fn)

    return results


def compute_progressive_narrowing(pipeline_output, ground_truth) -> dict:
    risk = pipeline_output.risk_analysis if pipeline_output else None
    exec_report = pipeline_output.execution_report if pipeline_output else None

    n_detected = 0
    n_requires_protection = 0
    n_strategy_active = 0
    n_executed = 0

    if risk:
        assessments = risk.risk_assessments if hasattr(risk, "risk_assessments") else []
        n_detected = len(assessments)
        n_requires_protection = sum(1 for a in assessments
                                     if (a.requires_protection if hasattr(a, "requires_protection") else a.get("requires_protection", False)))

    if exec_report:
        transforms = exec_report.transformations_applied if hasattr(exec_report, "transformations_applied") else []
        n_executed = sum(1 for t in transforms
                        if (t.status if hasattr(t, "status") else t.get("status")) == "success")

    narrowing_ratio = n_executed / max(n_detected, 1)

    return {
        "detected": n_detected,
        "requires_protection": n_requires_protection,
        "executed": n_executed,
        "narrowing_ratio": round(narrowing_ratio, 3),
    }


def _get_bbox(pred) -> list:
    if hasattr(pred, "bbox"):
        bbox = pred.bbox
        if hasattr(bbox, "x"):
            return [bbox.x, bbox.y, bbox.width, bbox.height]
        return list(bbox)
    return pred.get("bbox", [0, 0, 0, 0])
