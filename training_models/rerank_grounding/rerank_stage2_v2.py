#!/usr/bin/env python3
"""Rerank Stage2 grounding outputs with discriminative Qwen-VL scoring.

This script does not retrain the model. It reuses an existing Stage2 evaluation
JSON, builds token/box candidates from the generated grounding fields, scores
candidates with 0/1 next-token probabilities, replaces grounding predictions,
and recomputes the DGM4 metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from peft import PeftModel
from qwen_vl_utils import process_vision_info
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from eval_dgm4_qwen3vl import (
    ATOMIC_KEYS,
    box_iou_xyxy,
    category_to_atoms,
    multilabel_metrics,
    token_counts,
)

SYSTEM_TOKEN = (
    "You are a strict multimodal forensic token verifier. Given an image-caption pair, "
    "atomic manipulation labels, and one candidate caption token, decide whether that exact token is manipulated. "
    "Answer only 1 for manipulated or 0 for not manipulated."
)
SYSTEM_BOX = (
    "You are a strict multimodal forensic box verifier. Given an image-caption pair, atomic manipulation labels, "
    "and one candidate image box, decide whether that candidate box is the manipulated image region. "
    "Answer only 1 for correct manipulated region or 0 for not correct."
)

IMAGE_ATOMS = {"FS", "FA"}
TEXT_ATOMS = {"TS", "TA"}


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def caption_tokens(text: str) -> list[str]:
    return str(text or "").strip().split()


def has_image_manip(parsed: dict[str, Any]) -> bool:
    return any(int(parsed.get(k, 0) or 0) == 1 for k in IMAGE_ATOMS)


def has_text_manip(parsed: dict[str, Any]) -> bool:
    return any(int(parsed.get(k, 0) or 0) == 1 for k in TEXT_ATOMS)


def get_atoms_for_metrics(parsed: dict[str, Any]) -> list[int]:
    if all(key in parsed for key in ATOMIC_KEYS):
        return [int(parsed.get(key, 0) or 0) for key in ATOMIC_KEYS]
    return category_to_atoms(str(parsed.get("category", "orig")).lower())


def token_candidates(pred: list[int] | None, n_tokens: int, window: int, max_candidates: int) -> list[int]:
    pred = sorted(set(int(x) for x in (pred or []) if isinstance(x, int) or str(x).lstrip("-").isdigit()))
    cand: set[int] = set()
    for idx in pred:
        for off in range(-window, window + 1):
            j = idx + off
            if 0 <= j < n_tokens:
                cand.add(j)
    # If the generator returned no candidates for a text-manipulated sample, try all tokens up to max_candidates.
    if not cand and n_tokens:
        cand.update(range(min(n_tokens, max_candidates)))
    ordered = sorted(cand)
    if len(ordered) <= max_candidates:
        return ordered
    # Keep generated tokens first, then nearest neighbors, to cap runtime deterministically.
    priority = {idx: 0 for idx in pred}
    ordered.sort(key=lambda x: (priority.get(x, 1), min(abs(x - p) for p in pred) if pred else x, x))
    return sorted(ordered[:max_candidates])


def clamp_box(box: list[float], image_size: int = 512) -> list[float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(image_size), x1))
    y1 = max(0.0, min(float(image_size), y1))
    x2 = max(0.0, min(float(image_size), x2))
    y2 = max(0.0, min(float(image_size), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 - x1 < 1 or y2 - y1 < 1:
        return []
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def box_candidates(pred: list[float] | None, image_size: int = 512) -> list[list[float]]:
    if not pred or len(pred) != 4:
        return []
    x1, y1, x2, y2 = [float(v) for v in pred]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    cands = []
    # Original, scale variants, and small translations. These are cheap proposals around the generated box.
    for scale in [1.0, 0.85, 1.15, 1.3, 0.7]:
        nw, nh = w * scale, h * scale
        cands.append([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2])
    for dx, dy in [(-0.15*w, 0), (0.15*w, 0), (0, -0.15*h), (0, 0.15*h), (-0.1*w, -0.1*h), (0.1*w, 0.1*h)]:
        cands.append([x1 + dx, y1 + dy, x2 + dx, y2 + dy])
    out = []
    seen = set()
    for b in cands:
        cb = clamp_box(b, image_size=image_size)
        key = tuple(round(v, 1) for v in cb)
        if cb and key not in seen:
            out.append(cb)
            seen.add(key)
    return out


def image_path_for(sample: dict[str, Any], media_dir: str) -> str:
    rel = (sample.get("images") or [sample.get("image")])[0]
    return os.path.join(media_dir, rel)


def base_context(sample: dict[str, Any], parsed: dict[str, Any]) -> str:
    atoms = "\n".join(f"{k}: {int(parsed.get(k, 0) or 0)}" for k in ATOMIC_KEYS)
    toks = caption_tokens(sample.get("text", ""))
    token_lines = "\n".join(f"{i}: {tok}" for i, tok in enumerate(toks))
    return (
        f"Caption: \"{sample.get('text', '')}\"\n"
        f"Verdict: {parsed.get('verdict', '')}\n"
        f"{atoms}\n"
        f"Category: {parsed.get('category', '')}\n\n"
        f"Caption tokens:\n{token_lines}"
    )


def build_token_message(sample: dict[str, Any], parsed: dict[str, Any], idx: int, media_dir: str) -> list[dict[str, Any]]:
    toks = caption_tokens(sample.get("text", ""))
    tok = toks[idx] if 0 <= idx < len(toks) else ""
    prompt = (
        base_context(sample, parsed)
        + f"\n\nCandidate token index: {idx}\nCandidate token text: {tok}\n"
        + "Is this exact token manipulated? Answer only 0 or 1.\nAnswer:"
    )
    return [
        {"role": "system", "content": SYSTEM_TOKEN},
        {"role": "user", "content": [{"type": "image", "image": image_path_for(sample, media_dir)}, {"type": "text", "text": prompt}]},
    ]


def build_box_message(sample: dict[str, Any], parsed: dict[str, Any], box: list[float], media_dir: str) -> list[dict[str, Any]]:
    prompt = (
        base_context(sample, parsed)
        + f"\n\nCandidate image box in [x1, y1, x2, y2] pixel format: {box}\n"
        + "Is this candidate box the manipulated image region? Answer only 0 or 1.\nAnswer:"
    )
    return [
        {"role": "system", "content": SYSTEM_BOX},
        {"role": "user", "content": [{"type": "image", "image": image_path_for(sample, media_dir)}, {"type": "text", "text": prompt}]},
    ]


def score_messages(model, processor, messages: list[list[dict[str, Any]]], batch_size: int) -> list[float]:
    if not messages:
        return []
    one_ids = processor.tokenizer.encode(" 1", add_special_tokens=False) or processor.tokenizer.encode("1", add_special_tokens=False)
    zero_ids = processor.tokenizer.encode(" 0", add_special_tokens=False) or processor.tokenizer.encode("0", add_special_tokens=False)
    one_id, zero_id = one_ids[0], zero_ids[0]
    scores: list[float] = []
    processor.tokenizer.padding_side = "left"
    for start in range(0, len(messages), batch_size):
        batch = messages[start : start + batch_size]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        image_inputs, _ = process_vision_info(batch, return_video_kwargs=False)
        inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        last_idx = logits.shape[1] - 1
        pair = torch.stack([logits[:, last_idx, one_id], logits[:, last_idx, zero_id]], dim=1)
        probs = torch.softmax(pair.float(), dim=1)[:, 0]
        scores.extend(float(x) for x in probs.detach().cpu())
    return scores


def recompute_metrics(data: list[dict[str, Any]], details: list[dict[str, Any]]) -> dict[str, Any]:
    y_true_bin, y_score_bin = [], []
    y_true_ml, y_score_ml = [], []
    ious = []
    tok_tp = tok_fp = tok_fn = 0
    verdict_correct = parsed_count = 0
    for sample, detail in zip(data, details):
        parsed = detail.get("parsed") or {}
        gt_category = str(sample.get("fake_cls", "orig")).lower()
        gt_verdict = "REAL" if gt_category == "orig" else "FAKE"
        y_true_bin.append(0 if gt_verdict == "REAL" else 1)
        y_score_bin.append(float(detail.get("bin_score", 0.0)))
        if parsed:
            parsed_count += 1
            verdict_correct += int(str(parsed.get("verdict", "")).upper() == gt_verdict)
            gt_atoms = category_to_atoms(gt_category)
            pred_atoms = get_atoms_for_metrics(parsed)
            y_true_ml.append(gt_atoms)
            y_score_ml.append([1.0 if flag else -1.0 for flag in pred_atoms])
            ious.append(box_iou_xyxy(parsed.get("fake_image_box_pred"), sample.get("fake_image_box", [])))
            tp, fp, fn = token_counts(parsed.get("fake_text_pos_pred"), sample.get("fake_text_pos", []))
            tok_tp += tp; tok_fp += fp; tok_fn += fn
    labels = np.array(y_true_bin)
    scores = np.array(y_score_bin)
    metrics: dict[str, Any] = {
        "stage": "stage2-rerank",
        "samples": len(data),
        "parsed_samples": parsed_count,
        "AUC": float(roc_auc_score(labels, scores)) if len(set(y_true_bin)) > 1 else None,
        "EER": compute_eer(labels, scores) if len(set(y_true_bin)) > 1 else None,
        "ACC": verdict_correct / parsed_count if parsed_count else 0.0,
    }
    metrics.update(multilabel_metrics(np.array(y_score_ml), np.array(y_true_ml)) if y_true_ml else {})
    arr = np.array(ious)
    metrics["IoUmean"] = float(np.mean(arr)) if len(arr) else None
    metrics["IoU50"] = float(np.mean(arr > 0.5)) if len(arr) else None
    metrics["IoU75"] = float(np.mean(arr > 0.75)) if len(arr) else None
    tok_p = tok_tp / (tok_tp + tok_fp) if (tok_tp + tok_fp) else 0.0
    tok_r = tok_tp / (tok_tp + tok_fn) if (tok_tp + tok_fn) else 0.0
    metrics["Tok_Precision"] = float(tok_p)
    metrics["Tok_Recall"] = float(tok_r)
    metrics["Tok_F1"] = float((2 * tok_p * tok_r) / (tok_p + tok_r)) if (tok_p + tok_r) else 0.0
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--media-dir", required=True)
    ap.add_argument("--input-eval", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["token", "box", "both"], default="both")
    ap.add_argument("--score-batch-size", type=int, default=8)
    ap.add_argument("--token-window", type=int, default=1)
    ap.add_argument("--max-token-candidates", type=int, default=24)
    ap.add_argument("--token-threshold", type=float, default=0.5)
    ap.add_argument("--box-image-size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(Path(args.test_data).read_text(encoding="utf-8"))
    eval_obj = json.loads(Path(args.input_eval).read_text(encoding="utf-8"))
    details = eval_obj.get("details", eval_obj if isinstance(eval_obj, list) else [])
    if args.limit:
        data = data[: args.limit]
        details = details[: args.limit]
    assert len(data) == len(details), f"test data and eval details length mismatch: {len(data)} vs {len(details)}"

    print(f"Loading base model: {args.base_model}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.adapter, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    t0 = time.perf_counter()
    new_details = json.loads(json.dumps(details, ensure_ascii=False))

    token_scored = box_scored = 0
    for idx, (sample, detail) in enumerate(zip(data, new_details), start=1):
        parsed = detail.get("parsed") or {}
        if not parsed:
            continue
        if args.mode in {"token", "both"}:
            if has_text_manip(parsed):
                toks = caption_tokens(sample.get("text", ""))
                cands = token_candidates(parsed.get("fake_text_pos_pred"), len(toks), args.token_window, args.max_token_candidates)
                messages = [build_token_message(sample, parsed, c, args.media_dir) for c in cands]
                scores = score_messages(model, processor, messages, args.score_batch_size)
                kept = [c for c, s in zip(cands, scores) if s >= args.token_threshold]
                parsed["fake_text_pos_pred_original"] = parsed.get("fake_text_pos_pred")
                parsed["fake_text_pos_pred"] = kept
                parsed["token_rerank_scores"] = {str(c): s for c, s in zip(cands, scores)}
                token_scored += len(cands)
            else:
                parsed["fake_text_pos_pred_original"] = parsed.get("fake_text_pos_pred")
                parsed["fake_text_pos_pred"] = []
        if args.mode in {"box", "both"}:
            if has_image_manip(parsed):
                cands = box_candidates(parsed.get("fake_image_box_pred"), args.box_image_size)
                messages = [build_box_message(sample, parsed, b, args.media_dir) for b in cands]
                scores = score_messages(model, processor, messages, args.score_batch_size)
                best = None
                if cands and scores:
                    best_i = int(np.argmax(np.array(scores)))
                    best = cands[best_i]
                parsed["fake_image_box_pred_original"] = parsed.get("fake_image_box_pred")
                parsed["fake_image_box_pred"] = best or parsed.get("fake_image_box_pred")
                parsed["box_rerank_scores"] = [{"box": b, "score": s} for b, s in zip(cands, scores)]
                box_scored += len(cands)
            else:
                parsed["fake_image_box_pred_original"] = parsed.get("fake_image_box_pred")
                parsed["fake_image_box_pred"] = []
        detail["parsed"] = parsed
        if idx % 25 == 0 or idx == len(data):
            print(f"[{idx}/{len(data)}] token_candidates={token_scored} box_candidates={box_scored} elapsed={time.perf_counter()-t0:.0f}s")

    metrics = recompute_metrics(data, new_details)
    metrics.update({
        "mode": args.mode,
        "token_threshold": args.token_threshold,
        "token_window": args.token_window,
        "max_token_candidates": args.max_token_candidates,
        "token_candidates_scored": token_scored,
        "box_candidates_scored": box_scored,
        "rerank_time_s": time.perf_counter() - t0,
        "note": "Classification scores are reused from the original Stage2-v2 eval; grounding fields are reranked by 0/1 next-token probabilities."
    })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": metrics, "details": new_details}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out.with_name(out.stem + "_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
