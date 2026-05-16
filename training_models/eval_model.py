#!/usr/bin/env python3
"""Evaluate trained DGM4 LoRA model — full Table 2 metrics + batched inference."""
from __future__ import annotations
import argparse, json, re, os, time
from collections import defaultdict
from typing import Any

import torch
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score)


# ═══════════════════════════════════════════════════════
#  Parsing
# ═══════════════════════════════════════════════════════
def parse_5line(text: str) -> dict | None:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) != 5:
        return None
    out = {}
    for line in lines:
        for prefix in ["Verdict:", "Category:", "Fake Image Box:",
                        "Fake Text Pos:", "Evidence:"]:
            if line.startswith(prefix):
                val = line[len(prefix):].strip()
                if prefix == "Fake Image Box:":
                    out["fake_image_box_pred"] = _parse_box(val)
                elif prefix == "Fake Text Pos:":
                    out["fake_text_pos_pred"] = _parse_pos(val)
                elif prefix == "Verdict:":
                    out["verdict"] = val.upper()
                elif prefix == "Category:":
                    out["category"] = val.lower()
                elif prefix == "Evidence:":
                    out["evidence"] = val
    return out if len(out) == 5 else None


def _parse_box(s: str) -> list | None:
    s = s.strip().strip("[]")
    if not s:
        return []
    nums = re.findall(r"[\d.]+", s)
    return [float(x) for x in nums] if len(nums) == 4 else None


def _parse_pos(s: str) -> list | None:
    s = s.strip().strip("[]")
    if not s:
        return []
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]
    except Exception:
        return None


# ═══════════════════════════════════════════════════════
#  Metrics helpers
# ═══════════════════════════════════════════════════════
ALL_CLASSES = ["orig", "face_swap", "face_attribute",
               "text_swap", "text_attribute",
               "face_swap&text_swap", "face_swap&text_attribute",
               "face_attribute&text_swap", "face_attribute&text_attribute"]


def box_iou(a: list, b: list) -> float | None:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return None
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def pos_prf(pred_pos: list | None, gt_pos: list) -> tuple:
    """Token-level precision / recall / F1."""
    if pred_pos is None:
        pred_pos = []
    if not isinstance(gt_pos, list):
        gt_pos = []
    if not gt_pos and not pred_pos:
        return (1.0, 1.0, 1.0)
    if not gt_pos or not pred_pos:
        return (0.0, 0.0, 0.0)
    sp, sg = set(pred_pos), set(gt_pos)
    tp = len(sp & sg)
    precision = tp / len(sp) if sp else 0.0
    recall = tp / len(sg) if sg else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return (precision, recall, f1)


def compute_eer(labels, scores):
    """Equal Error Rate from scores."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


# ═══════════════════════════════════════════════════════
#  Batched inference
# ═══════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    "You are a multimodal forensic assistant. Your task is to assess "
    "whether an image-caption pair is authentic or manipulated. Check "
    "both visual evidence and caption-image consistency. If the sample "
    "is fake, identify the manipulation category and provide exact "
    "grounding fields for the manipulated image region and/or manipulated "
    "text token positions. If a modality is not manipulated, output an "
    "empty list for that grounding field.\n\n"
    "Always respond strictly in this five-line format:\n"
    "Verdict: [REAL or FAKE]\n"
    "Category: [category]\n"
    "Fake Image Box: [box or []]\n"
    "Fake Text Pos: [positions or []]\n"
    "Evidence: [concise grounded explanation]"
)


def _build_messages(sample: dict, media_dir: str) -> list[dict]:
    caption = sample.get("text", "")
    tpl = ('Caption: "{text}"\nPlease assess whether this image-caption '
           'pair is authentic or manipulated.')
    prompt = f"<image>\n{tpl.format(text=caption)}"
    imgs = sample.get("images", [])
    if not imgs:
        raise ValueError("no image")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": os.path.join(media_dir, imgs[0])},
            {"type": "text", "text": prompt},
        ]},
    ]


def run_batch(model, processor, samples: list, media_dir: str,
              max_tokens: int) -> list[tuple[str, float | None]]:
    """Return list of (response_text, fake_score) for each sample."""
    from qwen_vl_utils import process_vision_info

    messages_list = [_build_messages(s, media_dir) for s in samples]
    texts = [processor.apply_chat_template(
        m, tokenize=False, add_generation_prompt=True) for m in messages_list]

    # Qwen3-VL batch: use process_vision_info per conversation then stack
    all_image_inputs = []
    for m in messages_list:
        imgs, _ = process_vision_info([m], return_video_kwargs=False)
        all_image_inputs.append(imgs[0] if imgs else None)

    # Process in smaller sub-batches to avoid OOM
    results = []
    for i in range(len(samples)):
        img_in = [all_image_inputs[i]] if all_image_inputs[i] is not None else None
        inputs = processor(
            text=[texts[i]], images=img_in,
            return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                output_scores=True, return_dict_in_generate=True)

        response = processor.decode(
            outputs.sequences[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True).strip()

        # Extract fake score from first token logits
        score = None
        if hasattr(outputs, "scores") and outputs.scores:
            first_logits = outputs.scores[0][0]  # vocab logits
            # Find logit for "FAKE" vs "REAL" tokens
            fake_id = processor.tokenizer.encode("FAKE", add_special_tokens=False)
            real_id = processor.tokenizer.encode("REAL", add_special_tokens=False)
            # Some tokenizers split differently; try single tokens
            try:
                fake_token = fake_id[0] if fake_id else None
                real_token = real_id[0] if real_id else None
            except Exception:
                fake_token = real_token = None

            if fake_token is not None and real_token is not None:
                logits_pair = torch.stack([
                    first_logits[fake_token],
                    first_logits[real_token]])
                probs = torch.softmax(logits_pair.float(), dim=0)
                score = float(probs[0].cpu())  # P(FAKE)
        results.append((response, score))
    return results


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True)
    p.add_argument("--base-model",
                   default="/root/autodl-tmp/hf-cache/models--Qwen--"
                           "Qwen3-VL-8B-Instruct/snapshots/"
                           "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
    p.add_argument("--test-data",
                   default="/root/autodl-tmp/DGM4_Project/training_models/"
                           "datasets/dgm4_instruct/test.json")
    p.add_argument("--media-dir", default="/root/autodl-tmp/datasets")
    p.add_argument("--output", default="eval_results.json")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=256)
    args = p.parse_args()

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    print(f"Loading base model: {args.base_model}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    print(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.adapter, trust_remote_code=True)
    gpu_used = torch.cuda.max_memory_allocated() / 1e9
    print(f"Model loaded. GPU peak: {gpu_used:.1f} GB")

    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    if args.limit > 0:
        test_data = test_data[:args.limit]

    print(f"Evaluating {len(test_data)} samples...")

    # ── accumulators ────────────────────────────────
    results_detail = []                   # per-sample details
    bin_labels, bin_scores = [], []       # for AUC / EER
    ml_labels, ml_scores = [], []         # for mAP / CF1 / OF1  (one-hot per class)
    ious, iou_flags_50, iou_flags_75 = [], [], []  # image grounding
    tok_precisions, tok_recalls = [], []  # text grounding (per-sample)

    per_class = defaultdict(lambda: {
        "count": 0, "verdict_ok": 0, "category_ok": 0, "parsed": 0})

    t0 = time.perf_counter()

    for i, sample in enumerate(test_data):
        gt_cls = str(sample.get("fake_cls", "")).lower()
        gt_verdict = "FAKE" if gt_cls != "orig" else "REAL"
        gt_box = sample.get("fake_image_box", [])
        gt_pos = sample.get("fake_text_pos", [])
        bin_label = 1 if gt_verdict == "FAKE" else 0

        per_class[gt_cls]["count"] += 1

        try:
            batch_out = run_batch(
                model, processor, [sample], args.media_dir, args.max_tokens)
            raw, score = batch_out[0]
            err = None
        except Exception as e:
            raw, score, err = "", None, str(e)

        parsed = parse_5line(raw) if raw else None

        # ── Binary classification ──
        if score is not None:
            bin_labels.append(bin_label)
            bin_scores.append(score)

        # ── Multi-label ──
        if parsed:
            pred_cls = parsed.get("category", "")
            ml_vec_gt = [1.0 if c == gt_cls else 0.0 for c in ALL_CLASSES]
            ml_vec_pred = [1.0 if c == pred_cls else 0.0 for c in ALL_CLASSES]
            ml_labels.append(ml_vec_gt)
            ml_scores.append(ml_vec_pred)

        # ── Collect per-class ──
        if parsed:
            per_class[gt_cls]["parsed"] += 1
            if parsed.get("verdict", "") == gt_verdict:
                per_class[gt_cls]["verdict_ok"] += 1
            if parsed.get("category", "") == gt_cls:
                per_class[gt_cls]["category_ok"] += 1

        # ── Image grounding ──
        if parsed:
            pred_box = parsed.get("fake_image_box_pred")
            if pred_box is not None:
                iou = box_iou(pred_box, gt_box)
                if iou is not None:
                    ious.append(iou)
                    iou_flags_50.append(1.0 if iou > 0.5 else 0.0)
                    iou_flags_75.append(1.0 if iou > 0.75 else 0.0)

        # ── Text grounding ──
        if parsed:
            pred_pos = parsed.get("fake_text_pos_pred")
            pr, rc, _ = pos_prf(pred_pos, gt_pos)
            tok_precisions.append(pr)
            tok_recalls.append(rc)

        results_detail.append({
            "id": sample.get("id", i),
            "gt_verdict": gt_verdict,
            "gt_category": gt_cls,
            "bin_score": score,
            "raw_output": raw[:500],
            "parsed": parsed,
            "error": err,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{len(test_data)}] "
                  f"{100*(i+1)/len(test_data):.1f}% | "
                  f"{elapsed/(i+1):.1f}s/sample")

    elapsed = time.perf_counter() - t0

    # ── Compute Table 2 metrics ─────────────────────
    metrics = {"samples": len(test_data),
               "inference_time_s": elapsed,
               "seconds_per_sample": elapsed / max(1, len(test_data))}

    # 1. Binary Classification
    if bin_labels and bin_scores:
        try:
            metrics["AUC"] = float(roc_auc_score(bin_labels, bin_scores))
            metrics["EER"] = float(compute_eer(
                np.array(bin_labels), np.array(bin_scores)))
        except Exception:
            metrics["AUC"] = None
            metrics["EER"] = None
        # ACC from per-class (verdict level)
        total_parsed = sum(v["parsed"] for v in per_class.values())
        total_correct = sum(v["verdict_ok"] for v in per_class.values())
        metrics["ACC"] = total_correct / max(1, total_parsed)
    else:
        metrics["AUC"] = metrics["EER"] = metrics["ACC"] = None

    # 2. Multi-Label Classification
    if ml_labels and ml_scores:
        ml_labels_a = np.array(ml_labels)    # (N, 9)
        ml_scores_a = np.array(ml_scores)    # (N, 9)
        try:
            metrics["mAP"] = float(average_precision_score(
                ml_labels_a, ml_scores_a, average="macro"))
            metrics["CF1"] = float(f1_score(
                ml_labels_a, ml_scores_a, average="macro", zero_division=0))
            metrics["OF1"] = float(f1_score(
                ml_labels_a, ml_scores_a, average="micro", zero_division=0))
        except Exception:
            metrics["mAP"] = metrics["CF1"] = metrics["OF1"] = None
    else:
        metrics["mAP"] = metrics["CF1"] = metrics["OF1"] = None

    # 3. Image Grounding
    if ious:
        metrics["IoUmean"] = float(np.mean(ious))
        metrics["IoU50"]  = float(np.mean(iou_flags_50))
        metrics["IoU75"]  = float(np.mean(iou_flags_75))
    else:
        metrics["IoUmean"] = metrics["IoU50"] = metrics["IoU75"] = None

    # 4. Text Grounding
    if tok_precisions:
        # Macro-average per-sample P/R/F1
        metrics["Tok_Precision"] = float(np.mean(tok_precisions))
        metrics["Tok_Recall"]    = float(np.mean(tok_recalls))
        avg_p = metrics["Tok_Precision"]
        avg_r = metrics["Tok_Recall"]
        metrics["Tok_F1"] = (2 * avg_p * avg_r / (avg_p + avg_r)
                             if (avg_p + avg_r) else 0.0)
    else:
        metrics["Tok_Precision"] = metrics["Tok_Recall"] = metrics["Tok_F1"] = None

    # ── Per-class breakdown ──
    metrics["per_class"] = {}
    for cls_name in sorted(per_class):
        pc = per_class[cls_name]
        pn = max(1, pc["parsed"])
        metrics["per_class"][cls_name] = {
            "count": pc["count"],
            "parsed": pc["parsed"],
            "verdict_acc": pc["verdict_ok"] / pn,
            "category_acc": pc["category_ok"] / pn,
        }

    # ── Save ──
    out_path = os.path.join(os.path.dirname(args.adapter), args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": metrics, "details": results_detail},
                  f, ensure_ascii=False, indent=2)

    # ── Print Table 2 ──
    print(f"\n{'='*65}")
    print("  DGM4 Table 2 — Full Evaluation Metrics")
    print(f"{'='*65}")
    print("  [1] Binary Classification")
    print(f"      AUC : {metrics.get('AUC', 'N/A')}")
    print(f"      EER : {metrics.get('EER', 'N/A')}")
    print(f"      ACC : {metrics.get('ACC', 'N/A')}")
    print("  [2] Multi-Label Classification")
    print(f"      mAP : {metrics.get('mAP', 'N/A')}")
    print(f"      CF1 : {metrics.get('CF1', 'N/A')}")
    print(f"      OF1 : {metrics.get('OF1', 'N/A')}")
    print("  [3] Image Grounding")
    print(f"      IoUmean : {metrics.get('IoUmean', 'N/A')}")
    print(f"      IoU50   : {metrics.get('IoU50', 'N/A')}")
    print(f"      IoU75   : {metrics.get('IoU75', 'N/A')}")
    print("  [4] Text Grounding")
    print(f"      Tok_Precision : {metrics.get('Tok_Precision', 'N/A')}")
    print(f"      Tok_Recall    : {metrics.get('Tok_Recall', 'N/A')}")
    print(f"      Tok_F1        : {metrics.get('Tok_F1', 'N/A')}")
    print(f"{'='*65}")
    print(f"  Time: {elapsed:.0f}s  |  "
          f"{elapsed/max(1,len(test_data)):.1f}s/sample")
    print(f"  Results: {out_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
