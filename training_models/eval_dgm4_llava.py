#!/usr/bin/env python3
"""Stage-aware evaluation for DGM4 curriculum LLaVA-NeXT checkpoints.

Mirrors the schema of eval_curriculum_stage_outputs.py (Qwen3 version) so the
result JSON (summary + details) is 1:1 readable by the downstream Excel
update pipeline and any metric recomputation script.

Metric functions are imported from eval_dgm4_qwen3vl to guarantee identical
formula for multilabel/IoU/Token across the two model evaluations.
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
from peft import PeftModel
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

from eval_dgm4_qwen3vl import (
    ATOMIC_KEYS,
    SYSTEM_PROMPT,
    box_iou_xyxy,
    category_to_atoms,
    multilabel_metrics,
    parse_output,
    token_counts,
)

# Set to the explicit-format SYSTEM_PROMPT for fair zero-shot eval (un-finetuned
# models cannot infer the output format from the vague stage prompt).
PROMPT_OVERRIDE = None


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
        "do not include unchanged surrounding words. Fake Text Pos must be a JSON-style list of integer 0-based "
        "word indices into the numbered caption, not image coordinates or copied caption text. Respond only with "
        "the requested fields."
    ),
}


def numbered_caption(text: str) -> str:
    tokens = str(text or "").split()
    return " ".join(f"[{idx}] {tok}" for idx, tok in enumerate(tokens))


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def build_conversation(sample: dict, stage: str):
    """Build chat conversation; LLaVA-Mistral has no system role, fold system into user."""
    image_rel = (sample.get("images") or [sample.get("image")])[0]
    conversations = sample.get("conversations") or []
    user_text = conversations[0].get("value") if conversations else None
    if not user_text:
        caption = sample.get("text", "")
        user_text = (
            "Caption: " + json.dumps(caption) + "\n"
            "Determine whether this image-caption pair is authentic or manipulated."
        )
    user_text = user_text.replace("<image>", "").strip()
    numbered = numbered_caption(sample.get("text", ""))
    if numbered:
        user_text = (
            f"{user_text}\n\nNumbered Caption: {numbered}\n"
            "Use only a JSON-style list of 0-based integer indices from Numbered Caption for Fake Text Pos, "
            "for example [2, 5]. Do not copy the numbered caption text or use image coordinates."
        )
    if stage == "stage2" or PROMPT_OVERRIDE:
        user_text = (
            f"{user_text}\n\n"
            "Output exactly these five lines. Always include Fake Image Box and Fake Text Pos; use [] when empty:\n"
            "Verdict: [REAL or FAKE]\n"
            "Category: [orig, face_swap, face_attribute, text_swap, text_attribute, or a valid combined category joined by &]\n"
            "Fake Image Box: [box or []]\n"
            "Fake Text Pos: [positions or []]\n"
            "Evidence: [concise grounded explanation]"
        )
    sys_prompt = PROMPT_OVERRIDE if PROMPT_OVERRIDE else STAGE_SYSTEM[stage]
    combined = sys_prompt + "\n\n" + user_text
    conv = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": combined},
        ],
    }]
    return conv, image_rel


def _prepare_inputs(processor, samples, media_dir, stage, suffix=""):
    prompts, images = [], []
    for sample in samples:
        conv, image_rel = build_conversation(sample, stage)
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
        if suffix:
            prompt = prompt + suffix
        prompts.append(prompt)
        images.append(Image.open(os.path.join(media_dir, image_rel)).convert("RGB"))
    processor.tokenizer.padding_side = "left"
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    return inputs


def run_generate_batch(model, processor, samples, media_dir, stage, max_tokens):
    inputs = _prepare_inputs(processor, samples, media_dir, stage)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return [processor.decode(seq[input_len:], skip_special_tokens=True).strip() for seq in outputs]


def verdict_scores_batch(model, processor, samples, media_dir, stage):
    inputs = _prepare_inputs(processor, samples, media_dir, stage, suffix="Verdict:")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    fake_ids = processor.tokenizer.encode(" FAKE", add_special_tokens=False) or processor.tokenizer.encode("FAKE", add_special_tokens=False)
    real_ids = processor.tokenizer.encode(" REAL", add_special_tokens=False) or processor.tokenizer.encode("REAL", add_special_tokens=False)
    if not fake_ids or not real_ids:
        return [None for _ in samples]
    with torch.inference_mode():
        logits = model(**inputs).logits
    last_idx = logits.shape[1] - 1
    scores = []
    for row in range(len(samples)):
        pair = torch.stack([logits[row, last_idx, fake_ids[0]], logits[row, last_idx, real_ids[0]]])
        probs = torch.softmax(pair.float(), dim=0)
        scores.append(float(probs[0].cpu()))
    return scores


def _run_with_oom_retry(fn, model, processor, batch, media_dir, stage, start_bs, **kwargs):
    """Try start_bs; halve on OOM down to single-sample."""
    bs = start_bs
    while bs >= 1:
        try:
            results = []
            for i in range(0, len(batch), bs):
                sub = batch[i : i + bs]
                results.extend(fn(model, processor, sub, media_dir, stage, **kwargs))
            return results
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("[OOM] inner bs=" + str(bs) + " failed, halving")
            if bs == 1:
                raise
            bs = max(bs // 2, 1)


def evaluate(args):
    global PROMPT_OVERRIDE
    if getattr(args, "explicit_prompt", False):
        PROMPT_OVERRIDE = SYSTEM_PROMPT
        print("Using explicit-format SYSTEM_PROMPT (zero-shot fair eval)")
    print("Loading base model: " + args.base_model)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if args.no_adapter:
        print("Zero-shot mode: no LoRA adapter")
    else:
        print("Loading LoRA adapter: " + args.adapter)
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
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
            outputs = _run_with_oom_retry(run_generate_batch, model, processor, batch, args.media_dir, args.stage, args.batch_size, max_tokens=args.max_tokens)
            scores = _run_with_oom_retry(verdict_scores_batch, model, processor, batch, args.media_dir, args.stage, args.batch_size)
        except Exception as exc:
            print("[WARN] batch " + str(start) + " all retries failed: " + str(exc))
            outputs = [""] * len(batch)
            scores = [None] * len(batch)

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
                if args.stage == "stage2":
                    iou = box_iou_xyxy(parsed.get("fake_image_box_pred"), sample.get("fake_image_box", []))
                    ious.append(iou)
                    tp, fp, fn = token_counts(parsed.get("fake_text_pos_pred"), sample.get("fake_text_pos", []))
                    tok_tp += tp
                    tok_fp += fp
                    tok_fn += fn
            details.append({
                "id": sample.get("id"),
                "gt_category": gt_category,
                "gt_verdict": gt_verdict,
                "bin_score": score,
                "raw_output": raw,
                "parsed": parsed,
            })
        print("[" + str(min(start + len(batch), len(data))) + "/" + str(len(data)) + "] elapsed=" + str(int(time.perf_counter() - t0)) + "s")

    metrics = {"samples": len(data), "parsed_samples": parsed_count, "batch_size": args.batch_size, "stage": args.stage}
    metrics["AUC"] = float(roc_auc_score(np.array(y_true_bin), np.array(y_score_bin))) if y_true_bin else None
    metrics["EER"] = compute_eer(np.array(y_true_bin), np.array(y_score_bin)) if y_true_bin else None
    metrics["ACC"] = float(verdict_correct / parsed_count) if parsed_count else None
    if y_true_ml:
        metrics.update(multilabel_metrics(np.array(y_score_ml), np.array(y_true_ml)))
    else:
        metrics.update({"mAcc": None, "OF1": None, "CF1": None})
    if args.stage == "stage2":
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

    # 9-class verdict/category accuracy breakdown
    per_class = defaultdict(lambda: {"n": 0, "verdict_correct": 0, "category_correct": 0})
    for d in details:
        cls = d["gt_category"]
        per_class[cls]["n"] += 1
        if d.get("parsed"):
            per_class[cls]["verdict_correct"] += int(d["parsed"].get("verdict", "") == d["gt_verdict"])
            per_class[cls]["category_correct"] += int(str(d["parsed"].get("category", "")).lower() == cls.lower())
    metrics["per_class"] = {
        cls: {
            "n": v["n"],
            "verdict_acc": v["verdict_correct"] / v["n"] if v["n"] else 0.0,
            "category_acc": v["category_correct"] / v["n"] if v["n"] else 0.0,
        }
        for cls, v in per_class.items()
    }
    metrics["inference_time_s"] = time.perf_counter() - t0
    metrics["metric_note"] = "LLaVA-NeXT-Mistral evaluation. Stage1 reports only binary+multilabel; Stage2 adds image+text grounding."

    result = {"summary": metrics, "details": details}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path = out_path.with_name(out_path.stem + "_summary.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Wrote detailed results: " + str(out_path))
    print("Wrote summary metrics: " + str(metrics_path))

    # ---- Post-eval hook: Stage2 pre-flight smoke + YAML decisions ----
    # Triggers only after Stage1-v2-chain eval (i.e. right before Stage2-v2-chain train).
    # Calls BOTH bs=2 and bs=4 decide scripts (each is sentinel-guarded so runs at most once):
    #   - bs=2 decide: usually already done (called from launcher injection before Stage1-v2 train)
    #   - bs=4 decide: new, decides if Stage2 can be upgraded from bs=2 to bs=4
    if "stage1-cls-v2-chain" in str(args.adapter):
        print("\n[POST-EVAL HOOK] Stage1-v2-chain done. Running Stage2 smoke decisions.")
        try:
            del model
        except Exception:
            pass
        try:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
        import subprocess
        decide_scripts = [
            "/root/autodl-tmp/DGM4_Project/training_models/run_stage2_bs2_smoke_decide.sh",
            "/root/autodl-tmp/DGM4_Project/training_models/run_stage2_bs4_smoke_decide.sh",
        ]
        for script in decide_scripts:
            name = os.path.basename(script)
            if os.path.exists(script):
                try:
                    rc = subprocess.run(["bash", script], timeout=1100, check=False)
                    print("[POST-EVAL HOOK] " + name + " exit: " + str(rc.returncode))
                except subprocess.TimeoutExpired:
                    print("[POST-EVAL HOOK] " + name + " TIMED OUT, skipping.")
                except Exception as e:
                    print("[POST-EVAL HOOK] " + name + " error: " + str(e))
            else:
                print("[POST-EVAL HOOK] not found: " + script)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["stage1", "stage2"])
    parser.add_argument("--adapter", required=False, default=None)
    parser.add_argument("--no-adapter", action="store_true", help="zero-shot: load base model only")
    parser.add_argument("--explicit-prompt", action="store_true", help="use explicit 6-line SYSTEM_PROMPT (fair zero-shot)")
    parser.add_argument("--base-model", default="/root/autodl-tmp/DGM4_Project/training_models/base_models/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--media-dir", default="/root/autodl-tmp/datasets")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=140)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
