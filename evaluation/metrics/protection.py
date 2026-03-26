"""Protection efficacy metrics: face re-ID + OCR recovery."""
from typing import Dict, List, Optional

def compute_face_reid_rate(
    original_image_path: str,
    protected_image_path: str,
    face_bboxes: List[list],
    similarity_threshold: float = 0.6,
) -> dict:
    try:
        from PIL import Image
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1
    except ImportError:
        return {"error": "facenet_pytorch not available", "before_rate": 0, "after_rate": 0, "reduction": 0}

    device = "cpu"
    mtcnn = MTCNN(keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    original = Image.open(original_image_path).convert("RGB")
    protected = Image.open(protected_image_path).convert("RGB")

    results = []
    for bbox in face_bboxes:
        x, y, w, h = bbox
        orig_crop = original.crop((x, y, x + w, y + h)).resize((160, 160))
        prot_crop = protected.crop((x, y, x + w, y + h)).resize((160, 160))

        try:
            import torchvision.transforms as T
            transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
            orig_tensor = transform(orig_crop).unsqueeze(0).to(device)
            prot_tensor = transform(prot_crop).unsqueeze(0).to(device)

            with torch.no_grad():
                orig_emb = facenet(orig_tensor)
                prot_emb = facenet(prot_tensor)

            # Measure if original face is actually detectable (embedding has meaningful signal)
            orig_norm = torch.norm(orig_emb).item()
            original_detectable = orig_norm > 0.1  # FaceNet produces near-zero norm for non-face inputs

            similarity = torch.nn.functional.cosine_similarity(orig_emb, prot_emb).item()
            results.append({
                "bbox": bbox,
                "similarity": similarity,
                "reidentified": similarity > similarity_threshold,
                "original_detectable": original_detectable,
                "original_embedding_norm": round(orig_norm, 4),
            })
        except Exception:
            results.append({"bbox": bbox, "similarity": 0.0, "reidentified": False, "original_detectable": False})

    n = len(results)
    before_identifiable = sum(1 for r in results if r.get("original_detectable", False))
    before_rate = before_identifiable / max(n, 1)
    after_rate = sum(1 for r in results if r["reidentified"]) / max(n, 1)
    reduction = 1.0 - (after_rate / max(before_rate, 1e-9))

    return {
        "before_rate": before_rate,
        "after_rate": round(after_rate, 3),
        "reduction": round(reduction, 3),
        "per_face": results,
        "n_faces": n,
    }


def compute_ocr_recovery_rate(
    original_image_path: str,
    protected_image_path: str,
    text_bboxes: List[dict],
) -> dict:
    try:
        import easyocr
        from PIL import Image
        import numpy as np
    except ImportError:
        return {"error": "easyocr not available", "before_cer": 0, "after_cer": 0}

    reader = easyocr.Reader(["en"], gpu=False)
    original = np.array(Image.open(original_image_path).convert("RGB"))
    protected = np.array(Image.open(protected_image_path).convert("RGB"))

    before_cers = []
    results = []
    for item in text_bboxes:
        bbox = item.get("bbox", [0, 0, 0, 0])
        gt_text = item.get("text_content", "")
        pii_type = item.get("text_type", "unknown")

        x, y, w, h = bbox

        # Measure BEFORE CER on original image
        orig_crop = original[y:y + h, x:x + w]
        if orig_crop.size > 0 and gt_text:
            try:
                orig_ocr = " ".join(reader.readtext(orig_crop, detail=0)).strip()
                before_cer = _character_error_rate(gt_text, orig_ocr)
            except Exception:
                before_cer = 0.0
        else:
            before_cer = 0.0
        before_cers.append(before_cer)

        crop = protected[y:y + h, x:x + w]
        if crop.size == 0:
            results.append({"pii_type": pii_type, "recovered": "", "cer": 1.0})
            continue

        try:
            ocr_results = reader.readtext(crop, detail=0)
            recovered = " ".join(ocr_results).strip()
        except Exception:
            recovered = ""

        if gt_text:
            cer = _character_error_rate(gt_text, recovered)
        else:
            cer = 0.0 if not recovered else 1.0

        results.append({"pii_type": pii_type, "recovered": recovered, "cer": round(cer, 3)})

    avg_cer = sum(r["cer"] for r in results) / max(len(results), 1)

    by_type: Dict[str, List[float]] = {}
    for r in results:
        by_type.setdefault(r["pii_type"], []).append(r["cer"])

    avg_before_cer = sum(before_cers) / max(len(before_cers), 1)

    return {
        "before_cer": round(avg_before_cer, 3),
        "after_cer": round(avg_cer, 3),
        "per_region": results,
        "by_pii_type": {k: round(sum(v) / len(v), 3) for k, v in by_type.items()},
        "n_regions": len(results),
    }


def compute_protection_decision_accuracy(pipeline_output, ground_truth_elements) -> dict:
    from evaluation.metrics.detection import compute_iou

    tp = fp = fn = tn = 0

    for gt in ground_truth_elements:
        gt_bbox = gt.bbox if hasattr(gt, "bbox") else gt.get("bbox", [0, 0, 0, 0])
        should = gt.should_protect if hasattr(gt, "should_protect") else gt.get("should_protect", False)

        matched = False
        best_iou = 0.0
        best_assess = None
        if pipeline_output and pipeline_output.risk_analysis:
            for assess in pipeline_output.risk_analysis.risk_assessments:
                pred_bbox = assess.bbox
                if hasattr(pred_bbox, "x"):
                    pred_bbox = [pred_bbox.x, pred_bbox.y, pred_bbox.width, pred_bbox.height]
                iou = compute_iou(list(gt_bbox), list(pred_bbox))
                if iou >= 0.5 and iou > best_iou:
                    best_iou = iou
                    best_assess = assess

            if best_assess is not None:
                protected = best_assess.requires_protection if hasattr(best_assess, "requires_protection") else True
                if should and protected:
                    tp += 1
                elif should and not protected:
                    fn += 1
                elif not should and protected:
                    fp += 1
                else:
                    tn += 1
                matched = True

        if not matched and should:
            fn += 1
        elif not matched and not should:
            tn += 1

    total = tp + fp + fn + tn
    return {
        "accuracy": (tp + tn) / max(total, 1),
        "false_protection_rate": fp / max(fp + tn, 1),
        "missed_protection_rate": fn / max(fn + tp, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    ref = reference.lower().replace(" ", "")
    hyp = hypothesis.lower().replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0

    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return min(d[len(ref)][len(hyp)] / len(ref), 1.0)
