#!/usr/bin/env python3
"""Zero-shot (or LoRA) DGM4 evaluation for InternVL3.5-8B-HF.

Metric functions are imported VERBATIM from eval_dgm4_qwen3vl to guarantee an
identical 口径 (multilabel mAcc/CF1/OF1, IoU, token micro-F1, soft-prob AUC) with
the Qwen3-VL and LLaVA evaluations. Only the model class + prompt assembly are
InternVL-specific. Processed sample-by-sample (InternVL multi-tile pixel_values
make safe batching awkward); slower but robust for a one-off baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from eval_dgm4_qwen3vl import (
    SYSTEM_PROMPT, build_messages, parse_output, category_to_atoms,
    box_iou_xyxy, token_counts, multilabel_metrics, compute_eer,
)


def _image_of(sample: dict, media_dir: str) -> Image.Image:
    rel = (sample.get("images") or [sample.get("image")])[0]
    return Image.open(os.path.join(media_dir, rel)).convert("RGB")


def _text_of(processor, sample: dict, suffix: str = "") -> str:
    msgs = build_messages(sample, "")  # media_dir unused for text rendering
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text + suffix


def generate_one(model, processor, sample, media_dir, max_tokens) -> str:
    img = _image_of(sample, media_dir)
    inputs = processor(text=[_text_of(processor, sample)], images=[img], return_tensors="pt").to(model.device)
    ilen = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return processor.decode(out[0][ilen:], skip_special_tokens=True).strip()


def verdict_score_one(model, processor, sample, media_dir, fake_id, real_id) -> float | None:
    if fake_id is None or real_id is None:
        return None
    img = _image_of(sample, media_dir)
    inputs = processor(text=[_text_of(processor, sample, "Verdict:")], images=[img], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits
    last = logits.shape[1] - 1
    pair = torch.stack([logits[0, last, fake_id], logits[0, last, real_id]])
    return float(torch.softmax(pair.float(), dim=0)[0].cpu())


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if not args.no_adapter and args.adapter:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    else:
        print("Zero-shot mode: no LoRA adapter")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    tok = processor.tokenizer
    fk = tok.encode(" FAKE", add_special_tokens=False) or tok.encode("FAKE", add_special_tokens=False)
    rl = tok.encode(" REAL", add_special_tokens=False) or tok.encode("REAL", add_special_tokens=False)
    fake_id, real_id = (fk[0] if fk else None), (rl[0] if rl else None)

    with open(args.test_data, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]

    details, y_true_bin, y_score_bin, y_true_ml, y_score_ml, ious = [], [], [], [], [], []
    tok_tp = tok_fp = tok_fn = verdict_correct = parsed_count = 0
    per_class = defaultdict(lambda: {"n": 0, "verdict_correct": 0, "category_correct": 0})
    t0 = time.perf_counter()

    for i, sample in enumerate(data):
        gt_cls = str(sample.get("fake_cls", "orig")).lower()
        gt_verdict = "REAL" if gt_cls == "orig" else "FAKE"
        per_class[gt_cls]["n"] += 1
        try:
            raw = generate_one(model, processor, sample, args.media_dir, args.max_tokens)
            score = verdict_score_one(model, processor, sample, args.media_dir, fake_id, real_id)
        except Exception as exc:
            print(f"[WARN] sample {sample.get('id')} failed: {exc}")
            torch.cuda.empty_cache()
            raw, score = "", None
        parsed = parse_output(raw)
        if score is not None:
            y_true_bin.append(0 if gt_verdict == "REAL" else 1)
            y_score_bin.append(score)
        if parsed:
            parsed_count += 1
            pv = parsed.get("verdict", "")
            pc = str(parsed.get("category", "orig")).lower()
            verdict_correct += int(pv == gt_verdict)
            per_class[gt_cls]["verdict_correct"] += int(pv == gt_verdict)
            per_class[gt_cls]["category_correct"] += int(pc == gt_cls)
            y_true_ml.append(category_to_atoms(gt_cls))
            y_score_ml.append([1.0 if f else -1.0 for f in category_to_atoms(pc)])
            ious.append(box_iou_xyxy(parsed.get("fake_image_box_pred"), sample.get("fake_image_box", [])))
            a, b, c = token_counts(parsed.get("fake_text_pos_pred"), sample.get("fake_text_pos", []))
            tok_tp += a; tok_fp += b; tok_fn += c
        details.append({"id": sample.get("id"), "gt_category": gt_cls, "gt_verdict": gt_verdict,
                        "bin_score": score, "raw_output": raw, "parsed": parsed})
        if (i + 1) % 50 == 0 or i + 1 == len(data):
            print(f"[{i + 1}/{len(data)}] elapsed={time.perf_counter() - t0:.0f}s")

    m: dict[str, Any] = {"samples": len(data), "parsed_samples": parsed_count}
    m["AUC"] = float(roc_auc_score(np.array(y_true_bin), np.array(y_score_bin))) if len(set(y_true_bin)) > 1 else None
    m["EER"] = compute_eer(np.array(y_true_bin), np.array(y_score_bin)) if len(set(y_true_bin)) > 1 else None
    m["ACC"] = float(verdict_correct / parsed_count) if parsed_count else None
    if y_true_ml:
        m.update(multilabel_metrics(np.array(y_score_ml), np.array(y_true_ml)))
    if ious:
        m["IoUmean"] = float(np.mean(ious))
        m["IoU50"] = float(np.mean(np.array(ious) > 0.5))
        m["IoU75"] = float(np.mean(np.array(ious) > 0.75))
    tp_p = tok_tp / (tok_tp + tok_fp) if (tok_tp + tok_fp) else 0.0
    tp_r = tok_tp / (tok_tp + tok_fn) if (tok_tp + tok_fn) else 0.0
    m["Tok_Precision"], m["Tok_Recall"] = float(tp_p), float(tp_r)
    m["Tok_F1"] = float((2 * tp_p * tp_r) / (tp_p + tp_r)) if (tp_p + tp_r) else 0.0
    m["per_class"] = {c: {"n": v["n"], "verdict_acc": v["verdict_correct"] / v["n"] if v["n"] else 0.0,
                          "category_acc": v["category_correct"] / v["n"] if v["n"] else 0.0}
                      for c, v in per_class.items()}
    m["inference_time_s"] = time.perf_counter() - t0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": m, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in m.items() if k not in ("per_class",)}, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_path}")
    return {"summary": m}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="OpenGVLab/InternVL3_5-8B-HF")
    p.add_argument("--adapter", default=None)
    p.add_argument("--no-adapter", action="store_true", help="zero-shot: base model only")
    p.add_argument("--test-data", default="datasets/dgm4_stage3_final/test.json")
    p.add_argument("--media-dir", default="/root/autodl-tmp/datasets")
    p.add_argument("--output", default="outputs/internvl3_5-8b/dgm4-zeroshot/eval_dgm4_12metrics.json")
    p.add_argument("--max-tokens", type=int, default=220)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
