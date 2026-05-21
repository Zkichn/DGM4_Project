#!/usr/bin/env python3
"""Evaluate a Qwen3-VL DGM4 LoRA adapter with DGM4/HAMMER-style metrics.

This evaluator keeps the original DGM4 metric definitions where possible:

- Binary: AUC / EER / ACC.
- Multi-label: four atomic labels FS, FA, TS, TA with mAcc / CF1 / OF1.
- Image grounding: IoUmean / IoU50 / IoU75 over all samples; empty GT and
  empty prediction counts as IoU=1, matching the original empty-box handling.
- Text grounding: global token-position Precision / Recall / F1 from TP/FP/FN.

For generative models there is no native classifier head.  Binary fake score is
estimated by adding the prefix "Verdict:" after the prompt and comparing the
next-token probability of FAKE vs REAL.  Multi-label mAcc/CF1/OF1 are computed
from the generated atomic labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


ATOMIC_LABELS = ("face_swap", "face_attribute", "text_swap", "text_attribute")
ATOMIC_KEYS = ("FS", "FA", "TS", "TA")
VALID_CATEGORIES = {
    "orig",
    "face_swap",
    "face_attribute",
    "text_swap",
    "text_attribute",
    "face_swap&text_swap",
    "face_swap&text_attribute",
    "face_attribute&text_swap",
    "face_attribute&text_attribute",
}

SYSTEM_PROMPT = (
    "You are a multimodal forensic assistant. Assess whether an image-caption "
    "pair is authentic or manipulated. Check visual evidence and caption-image "
    "consistency. If the sample is fake, identify the manipulation category and "
    "provide exact grounding fields for the manipulated image region and/or "
    "manipulated text token positions. If a modality is not manipulated, output "
    "an empty list for that grounding field. Respond strictly in this five-line "
    "format:\n"
    "Verdict: [REAL or FAKE]\n"
    "Category: [orig, face_swap, face_attribute, text_swap, text_attribute, or a valid combined category joined by &]\n"
    "Fake Image Box: [box or []]\n"
    "Fake Text Pos: [positions or []]\n"
    "Evidence: [concise grounded explanation]"
)


def category_to_atoms(category: str | None) -> list[int]:
    category = (category or "orig").strip().lower()
    parts = set() if category == "orig" else {p.strip() for p in category.split("&") if p.strip()}
    return [1 if label in parts else 0 for label in ATOMIC_LABELS]


def atoms_to_category(atoms: list[int]) -> str:
    parts = [label for label, flag in zip(ATOMIC_LABELS, atoms) if flag]
    return "&".join(parts) if parts else "orig"


def parse_list_numbers(text: str, as_int: bool = False) -> list[int | float] | None:
    value = text.strip()
    if value == "[]":
        return []
    nums = re.findall(r"-?\d+(?:\.\d+)?", value)
    if not nums:
        return []
    if as_int:
        return [int(float(x)) for x in nums]
    return [float(x) for x in nums]


def parse_output(text: str) -> dict[str, Any] | None:
    parsed: dict[str, Any] = {}
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("verdict:"):
            parsed["verdict"] = line.split(":", 1)[1].strip().upper()
        elif lower.startswith("category:"):
            category = line.split(":", 1)[1].strip().lower()
            parsed["category"] = category if category in VALID_CATEGORIES else category
        elif lower.startswith("fake image box:"):
            box = parse_list_numbers(line.split(":", 1)[1].strip(), as_int=False)
            parsed["fake_image_box_pred"] = box if box is not None and len(box) in (0, 4) else None
        elif lower.startswith("fake text pos:"):
            parsed["fake_text_pos_pred"] = parse_list_numbers(line.split(":", 1)[1].strip(), as_int=True)
        else:
            for key in ATOMIC_KEYS:
                if lower.startswith(f"{key.lower()}:"):
                    value = line.split(":", 1)[1].strip().lower()
                    parsed[key] = 1 if value.startswith(("y", "1", "true")) else 0

    if "category" not in parsed and all(key in parsed for key in ATOMIC_KEYS):
        parsed["category"] = atoms_to_category([parsed[key] for key in ATOMIC_KEYS])
    if "verdict" not in parsed and "category" in parsed:
        parsed["verdict"] = "REAL" if parsed["category"] == "orig" else "FAKE"
    return parsed if "verdict" in parsed else None


def build_messages(sample: dict[str, Any], media_dir: str) -> list[dict[str, Any]]:
    image_rel = (sample.get("images") or [sample.get("image")])[0]
    image_path = os.path.join(media_dir, image_rel)

    conversations = sample.get("conversations") or []
    user_text = None
    if conversations:
        user_text = conversations[0].get("value")
    if not user_text:
        user_text = f'<image> Caption: "{sample.get("text", "")}"\nPlease assess whether this image-caption pair is authentic or manipulated.'
    user_text = user_text.replace("<image>", "").strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def multilabel_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    preds = scores >= 0
    per_class_acc = np.mean(preds == targets, axis=0)
    nc = np.sum(targets * preds, axis=0).astype(float)
    npred = np.sum(preds, axis=0).astype(float)
    ngt = np.sum(targets == 1, axis=0).astype(float)
    npred_safe = np.where(npred == 0, 1, npred)
    op = float(np.sum(nc) / np.sum(npred_safe))
    or_ = float(np.sum(nc) / np.sum(ngt))
    of1 = float((2 * op * or_) / (op + or_)) if (op + or_) else 0.0
    cp = float(np.sum(nc / npred_safe) / targets.shape[1])
    cr = float(np.sum(nc / ngt) / targets.shape[1])
    cf1 = float((2 * cp * cr) / (cp + cr)) if (cp + cr) else 0.0
    per_label = {}
    for idx, key in enumerate(ATOMIC_KEYS):
        p = float(nc[idx] / npred_safe[idx])
        r = float(nc[idx] / ngt[idx]) if ngt[idx] else 0.0
        f1 = float((2 * p * r) / (p + r)) if (p + r) else 0.0
        per_label[key] = {
            "accuracy": float(per_class_acc[idx]),
            "precision": p,
            "recall": r,
            "F1": f1,
            "support": int(ngt[idx]),
        }
    return {
        "mAcc": float(per_class_acc.mean()),
        "OP": op,
        "OR": or_,
        "OF1": of1,
        "CP": cp,
        "CR": cr,
        "CF1": cf1,
        "per_label": per_label,
    }


def box_iou_xyxy(pred: list[float] | None, gt: list[float] | None) -> float:
    pred = pred or []
    gt = gt or []
    if not pred and not gt:
        return 1.0
    if len(pred) != 4 or len(gt) != 4:
        return 0.0
    xa, ya = max(pred[0], gt[0]), max(pred[1], gt[1])
    xb, yb = min(pred[2], gt[2]), min(pred[3], gt[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_p = max(0.0, pred[2] - pred[0]) * max(0.0, pred[3] - pred[1])
    area_g = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
    union = area_p + area_g - inter
    return float(inter / union) if union > 0 else 0.0


def token_counts(pred: list[int] | None, gt: list[int] | None) -> tuple[int, int, int]:
    sp = set(pred or [])
    sg = set(gt or [])
    tp = len(sp & sg)
    fp = len(sp - sg)
    fn = len(sg - sp)
    return tp, fp, fn


def run_generate_batch(model, processor, samples: list[dict[str, Any]], media_dir: str, max_tokens: int) -> list[str]:
    from qwen_vl_utils import process_vision_info

    messages = [build_messages(sample, media_dir) for sample in samples]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    image_inputs, _ = process_vision_info(messages, return_video_kwargs=False)
    processor.tokenizer.padding_side = "left"
    inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return [processor.decode(seq[input_len:], skip_special_tokens=True).strip() for seq in outputs]


def verdict_scores_batch(model, processor, samples: list[dict[str, Any]], media_dir: str) -> list[float | None]:
    from qwen_vl_utils import process_vision_info

    messages = [build_messages(sample, media_dir) for sample in samples]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) + "Verdict:" for m in messages]
    image_inputs, _ = process_vision_info(messages, return_video_kwargs=False)
    processor.tokenizer.padding_side = "left"
    inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    fake_ids = processor.tokenizer.encode(" FAKE", add_special_tokens=False) or processor.tokenizer.encode("FAKE", add_special_tokens=False)
    real_ids = processor.tokenizer.encode(" REAL", add_special_tokens=False) or processor.tokenizer.encode("REAL", add_special_tokens=False)
    if not fake_ids or not real_ids:
        return [None for _ in samples]
    with torch.inference_mode():
        logits = model(**inputs).logits
    scores: list[float] = []
    # With left padding, the scored "Verdict:" prefix ends at the last column
    # for every row in the padded batch.
    last_idx = logits.shape[1] - 1
    for row in range(len(samples)):
        pair = torch.stack([logits[row, last_idx, fake_ids[0]], logits[row, last_idx, real_ids[0]]])
        probs = torch.softmax(pair.float(), dim=0)
        scores.append(float(probs[0].cpu()))
    return scores


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel

    print(f"Loading base model: {args.base_model}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    print(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.adapter, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    with open(args.test_data, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]

    details = []
    y_true_bin, y_score_bin = [], []
    y_true_ml, y_score_ml = [], []
    ious = []
    tok_tp = tok_fp = tok_fn = 0
    verdict_correct = parsed_count = 0
    t0 = time.perf_counter()

    for start in range(0, len(data), args.batch_size):
        batch = data[start : start + args.batch_size]
        try:
            outputs = run_generate_batch(model, processor, batch, args.media_dir, args.max_tokens)
            scores = verdict_scores_batch(model, processor, batch, args.media_dir)
        except Exception as exc:
            torch.cuda.empty_cache()
            print(f"[WARN] batch {start} failed: {exc}; retrying one by one")
            outputs, scores = [], []
            for sample in batch:
                try:
                    outputs.extend(run_generate_batch(model, processor, [sample], args.media_dir, args.max_tokens))
                    scores.extend(verdict_scores_batch(model, processor, [sample], args.media_dir))
                except Exception as one_exc:
                    print(f"[WARN] sample {sample.get('id')} failed: {one_exc}")
                    outputs.append("")
                    scores.append(None)

        for sample, raw, score in zip(batch, outputs, scores):
            gt_category = str(sample.get("fake_cls", "orig")).lower()
            gt_verdict = "REAL" if gt_category == "orig" else "FAKE"
            gt_bin = 0 if gt_verdict == "REAL" else 1
            parsed = parse_output(raw)
            if score is not None:
                y_true_bin.append(gt_bin)
                y_score_bin.append(score)
            if parsed:
                parsed_count += 1
                pred_verdict = parsed.get("verdict", "")
                pred_category = str(parsed.get("category", "orig")).lower()
                verdict_correct += int(pred_verdict == gt_verdict)
                gt_atoms = category_to_atoms(gt_category)
                pred_atoms = category_to_atoms(pred_category)
                y_true_ml.append(gt_atoms)
                y_score_ml.append([1.0 if flag else -1.0 for flag in pred_atoms])
                iou = box_iou_xyxy(parsed.get("fake_image_box_pred"), sample.get("fake_image_box", []))
                ious.append(iou)
                tp, fp, fn = token_counts(parsed.get("fake_text_pos_pred"), sample.get("fake_text_pos", []))
                tok_tp += tp
                tok_fp += fp
                tok_fn += fn
            details.append({"id": sample.get("id"), "gt_category": gt_category, "gt_verdict": gt_verdict, "bin_score": score, "raw_output": raw, "parsed": parsed})
        print(f"[{min(start + len(batch), len(data))}/{len(data)}] elapsed={time.perf_counter() - t0:.0f}s")

    metrics: dict[str, Any] = {"samples": len(data), "parsed_samples": parsed_count, "batch_size": args.batch_size}
    metrics["AUC"] = float(roc_auc_score(np.array(y_true_bin), np.array(y_score_bin))) if y_true_bin else None
    metrics["EER"] = compute_eer(np.array(y_true_bin), np.array(y_score_bin)) if y_true_bin else None
    metrics["ACC"] = float(verdict_correct / parsed_count) if parsed_count else None
    if y_true_ml:
        metrics.update(multilabel_metrics(np.array(y_score_ml), np.array(y_true_ml)))
    else:
        metrics.update({"mAcc": None, "OF1": None, "CF1": None})
    if ious:
        metrics["IoUmean"] = float(np.mean(ious))
        metrics["IoU50"] = float(np.mean(np.array(ious) > 0.5))
        metrics["IoU75"] = float(np.mean(np.array(ious) > 0.75))
    else:
        metrics["IoUmean"] = metrics["IoU50"] = metrics["IoU75"] = None
    tok_p = tok_tp / (tok_tp + tok_fp) if (tok_tp + tok_fp) else 0.0
    tok_r = tok_tp / (tok_tp + tok_fn) if (tok_tp + tok_fn) else 0.0
    metrics["Tok_Precision"] = float(tok_p)
    metrics["Tok_Recall"] = float(tok_r)
    metrics["Tok_F1"] = float((2 * tok_p * tok_r) / (tok_p + tok_r)) if (tok_p + tok_r) else 0.0
    metrics["inference_time_s"] = time.perf_counter() - t0

    result = {"summary": metrics, "details": details}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path = out_path.with_name(out_path.stem + "_summary.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote detailed results: {out_path}")
    print(f"Wrote summary metrics: {metrics_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="outputs/qwen3-vl-8b/dgm4-curriculum/stage3-final")
    parser.add_argument("--base-model", default="/root/autodl-tmp/hf-cache/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
    parser.add_argument("--test-data", default="datasets/dgm4_stage3_final/test.json")
    parser.add_argument("--media-dir", default="/root/autodl-tmp/datasets")
    parser.add_argument("--output", default="outputs/qwen3-vl-8b/dgm4-curriculum/stage3-final/eval_dgm4_12metrics.json")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
