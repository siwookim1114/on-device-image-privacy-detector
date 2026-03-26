"""Risk severity accuracy metrics."""
from typing import Dict
from evaluation.metrics.detection import compute_iou, _normalize_pred_type

SEVERITY_LEVELS = ["critical", "high", "medium", "low"]

def compute_severity_accuracy(
    predictions: list,
    ground_truth: list,
    iou_threshold: float = 0.5,
) -> dict:
    total = 0
    correct = 0
    confusion: Dict[str, Dict[str, int]] = {s: {t: 0 for t in SEVERITY_LEVELS} for s in SEVERITY_LEVELS}

    for gt in ground_truth:
        gt_bbox = gt.bbox if hasattr(gt, "bbox") else gt.get("bbox", [0, 0, 0, 0])
        gt_type = gt.element_type if hasattr(gt, "element_type") else gt.get("element_type", "")
        gt_sev = gt.severity_gt if hasattr(gt, "severity_gt") else gt.get("severity_gt", "low")

        best_iou = 0.0
        best_pred_sev = None
        for pred in predictions:
            pred_type = _normalize_pred_type(pred)
            if pred_type != gt_type:
                continue
            pred_bbox = _get_pred_bbox(pred)
            iou = compute_iou(list(gt_bbox), list(pred_bbox))
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                sev = pred.severity if hasattr(pred, "severity") else pred.get("severity", "low")
                best_pred_sev = sev.value if hasattr(sev, "value") else str(sev)

        if best_pred_sev is not None:
            total += 1
            gt_sev_str = str(gt_sev).lower()
            pred_sev_str = str(best_pred_sev).lower()
            if gt_sev_str == pred_sev_str:
                correct += 1
            if gt_sev_str in confusion and pred_sev_str in confusion[gt_sev_str]:
                confusion[gt_sev_str][pred_sev_str] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "confusion_matrix": confusion,
    }


def _get_pred_bbox(pred) -> list:
    if hasattr(pred, "bbox"):
        bbox = pred.bbox
        if hasattr(bbox, "x"):
            return [bbox.x, bbox.y, bbox.width, bbox.height]
        return list(bbox)
    return pred.get("bbox", [0, 0, 0, 0])
