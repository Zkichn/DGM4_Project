#!/usr/bin/env python3
"""Stage-aware evaluation for DGM4 curriculum checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from eval_dgm4_qwen3vl import (
    ATOMIC_KEYS,
    box_iou_xyxy,
    category_to_atoms,
    multilabel_metrics,
    parse_output,
    token_counts,
)


STAGE_SYSTEM = {
    "stage1": (
        "You are a multimodal forensic classifier. Determine whether the image-caption pair is REAL or FAKE, "
        "predict four atomic labels FS/FA/TS/TA as 0 or 1, and output the final category. "
        "Respond strictly in this six-line format:\n"
        "Verdict: [REAL or FAKE]\nFS: [0 or 1]\nFA: [0 or 1]\nTS: [0 or 1]\nTA: [0 or 1]\nCategory: [category]"
    ),
    "stage2": (
        "You are a multimodal forensic grounding assistant. Determine authenticity, atomic labels FS/FA/TS/TA, "
        "final category, and requested grounding fields. Mark only minimal manipulated text token positions and "
        "do not include unchanged surrounding words. Respond only with the requested fields."
    ),
}


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def build_messages(sample: dict[str, Any], media_dir: str, stage: str) -> list[dict[str, Any]]:
    image_rel = (sample.get("images") or [sample.get("image")])[0]
    image_path = os.path.join(media_dir, image_rel)
    conversations = sample.get("conversations") or []
    user_text = conversations[0].get("value") if conversations else None
    if not user_text:
        user_text = f'<image> Caption: "{sample.get("text", "")}"\nDetermine whether this image-caption pair is authentic or manipulated.'
    user_text = user_text.replace("<image>", "").strip()
    return [
        {"role": "system", "content": STAGE_SYSTEM[stage]},
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": user_text}]},
    ]


def generate_batch(model, processor, samples: list[dict[str, Any]], media_dir: str, stage: str, max_tokens: int) -> list[str]:
    messages = [build_messages(sample, media_dir, stage) for sample in samples]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    image_inputs, _ = process_vision_info(messages, return_video_kwargs=False)
    processor.tokenizer.padding_side = "left"
    inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return [processor.decode(seq[input_len:], skip_special_tokens=True).strip() for seq in outputs]


def verdict_scores_batch(model, processor, samples: list[dict[str, Any]], media_dir: str, stage: str) -> list[float]:
    messages = [build_messages(sample, media_dir, stage) for sample in samples]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) + "Verdict:" for m in messages]
    image_inputs, _ = process_vision_info(messages, return_video_kwargs=False)
    processor.tokenizer.padding_side = "left"
    inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    fake_ids = processor.tokenizer.encode(" FAKE", add_special_tokens=False) or processor.tokenizer.encode("FAKE", add_special_tokens=False)
    real_ids = processor.tokenizer.encode(" REAL", add_special_tokens=False) or processor.tokenizer.encode("REAL", add_special_tokens=False)
    with torch.inference_mode():
        logits = model(**inputs).logits
    last_idx = logits.shape[1] - 1
    scores: list[float] = []
    for row in range(len(samples)):
        pair = torch.stack([logits[row, last_idx, fake_ids[0]], logits[row, last_idx, real_ids[0]]])
        probs = torch.softmax(pair.float(), dim=0)
        scores.append(float(probs[0].cpu()))
    return scores


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
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

    data = json.loads(Path(args.test_data).read_text(encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]

    y_true_bin, y_score_bin = [], []
    y_true_ml, y_score_ml = [], []
    ious: list[float] = []
    tok_tp = tok_fp = tok_fn = 0
    parsed_count = verdict_correct = 0
    details = []
    t0 = time.perf_counter()

    for start in range(0, len(data), args.batch_size):
        batch = data[start : start + args.batch_size]
        try:
            outputs = generate_batch(model, processor, batch, args.media_dir, args.stage, args.max_tokens)
            scores = verdict_scores_batch(model, processor, batch, args.media_dir, args.stage)
        except Exception as exc:
            print(f"[WARN] batch {start} failed: {exc}; falling back to single sample")
            torch.cuda.empty_cache()
            outputs, scores = [], []
            for sample in batch:
                outputs.extend(generate_batch(model, processor, [sample], args.media_dir, args.stage, args.max_tokens))
                scores.extend(verdict_scores_batch(model, processor, [sample], args.media_dir, args.stage))

        for sample, raw, score in zip(batch, outputs, scores):
            gt_category = str(sample.get("fake_cls", "orig")).lower()
            gt_verdict = "REAL" if gt_category == "orig" else "FAKE"
            gt_bin = 0 if gt_verdict == "REAL" else 1
            y_true_bin.append(gt_bin)
            y_score_bin.append(float(score))
            parsed = parse_output(raw)
            if parsed:
                parsed_count += 1
                pred_verdict = str(parsed.get("verdict", "")).upper()
                verdict_correct += int(pred_verdict == gt_verdict)
                if "category" in parsed:
                    pred_atoms = category_to_atoms(str(parsed.get("category", "orig")).lower())
                elif all(key in parsed for key in ATOMIC_KEYS):
                    pred_atoms = [parsed[key] for key in ATOMIC_KEYS]
                else:
                    pred_atoms = category_to_atoms("orig")
                gt_atoms = category_to_atoms(gt_category)
                y_true_ml.append(gt_atoms)
                y_score_ml.append([1.0 if flag else -1.0 for flag in pred_atoms])
                if args.stage == "stage2":
                    ious.append(box_iou_xyxy(parsed.get("fake_image_box_pred"), sample.get("fake_image_box", [])))
                    tp, fp, fn = token_counts(parsed.get("fake_text_pos_pred"), sample.get("fake_text_pos", []))
                    tok_tp += tp
                    tok_fp += fp
                    tok_fn += fn
            details.append(
                {
                    "id": sample.get("id"),
                    "gt_category": gt_category,
                    "gt_verdict": gt_verdict,
                    "bin_score": score,
                    "raw_output": raw,
                    "parsed": parsed,
                }
            )
        print(f"[{min(start + len(batch), len(data))}/{len(data)}] elapsed={time.perf_counter() - t0:.0f}s")

    metrics: dict[str, Any] = {"stage": args.stage, "samples": len(data), "parsed_samples": parsed_count, "batch_size": args.batch_size}
    labels = np.array(y_true_bin)
    scores = np.array(y_score_bin)
    metrics["AUC"] = float(roc_auc_score(labels, scores)) if len(set(y_true_bin)) > 1 else None
    metrics["EER"] = compute_eer(labels, scores) if len(set(y_true_bin)) > 1 else None
    metrics["ACC"] = verdict_correct / parsed_count if parsed_count else 0.0
    metrics.update(multilabel_metrics(np.array(y_score_ml), np.array(y_true_ml)) if y_true_ml else {"mAcc": None, "OF1": None, "CF1": None})

    if args.stage == "stage2":
        arr = np.array(ious)
        metrics["IoUmean"] = float(np.mean(arr)) if len(arr) else None
        metrics["IoU50"] = float(np.mean(arr > 0.5)) if len(arr) else None
        metrics["IoU75"] = float(np.mean(arr > 0.75)) if len(arr) else None
        tok_p = tok_tp / (tok_tp + tok_fp) if (tok_tp + tok_fp) else 0.0
        tok_r = tok_tp / (tok_tp + tok_fn) if (tok_tp + tok_fn) else 0.0
        metrics["Tok_Precision"] = float(tok_p)
        metrics["Tok_Recall"] = float(tok_r)
        metrics["Tok_F1"] = float((2 * tok_p * tok_r) / (tok_p + tok_r)) if (tok_p + tok_r) else 0.0

    metrics["inference_time_s"] = time.perf_counter() - t0
    metrics["metric_note"] = (
        "Stage-aware evaluation. Stage1 reports only binary and atomic multilabel metrics; "
        "Stage2 additionally reports image/text grounding."
    )
    result = {"summary": metrics, "details": details}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_name(output.stem + "_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--media-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
