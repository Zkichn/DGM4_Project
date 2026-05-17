#!/usr/bin/env python3
"""Evaluate trained DGM4 LoRA model with TRUE BATCH inference.

Difference vs eval_model.py:
- Processes multiple samples per forward pass (configurable --batch-size)
- Default --batch-size=24 (canonical; user-verified on RTX 4080 SUPER 32GB)
- On OOM at primary batch, retries failed batch in chunks of --oom-fallback-batch (default 16)
- Significant GPU utilization improvement (40% → 70%+ at batch=8, higher at batch=24)
- Estimated 3-5x speedup vs single-sample inference

All 12 Table 2 metrics retained (Binary / Multi-Label / Image / Text Grounding).
"""
from __future__ import annotations
import argparse, json, re, os, time
from collections import defaultdict
from typing import Any

import torch
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, roc_curve)


# ═══════════════════════════════════════════════════════
#  Parsing (same as eval_model.py)
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
    p = tp / len(sp) if sp else 0.0
    r = tp / len(sg) if sg else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return (p, r, f)


def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


# ═══════════════════════════════════════════════════════
#  Prompt / message building
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


def build_messages(sample: dict, media_dir: str) -> list[dict]:
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


# ═══════════════════════════════════════════════════════
#  TRUE BATCH inference
# ═══════════════════════════════════════════════════════
def run_batch_true(model, processor, samples: list, media_dir: str,
                    max_tokens: int) -> list[tuple[str, float | None]]:
    """
    True batched inference: pack N samples into a single forward pass.

    Returns: list of (response_text, fake_score) tuples.

    Key implementation details:
    1. Build all messages for the batch
    2. Apply chat template to each (gets text strings)
    3. process_vision_info collects all images for all conversations
    4. processor(...) handles padding with left-padding (essential for generation)
    5. model.generate(...) processes the whole batch in one forward pass per step
    6. Decode each sample by slicing out its generated portion
    """
    from qwen_vl_utils import process_vision_info

    # Build messages and texts for each sample
    messages_list = [build_messages(s, media_dir) for s in samples]
    texts = [processor.apply_chat_template(
        m, tokenize=False, add_generation_prompt=True) for m in messages_list]

    # Collect images for ALL conversations (returns flat list)
    image_inputs, _ = process_vision_info(
        messages_list, return_video_kwargs=False)

    # CRITICAL: Use left-padding for generation. Right-padding breaks gen.
    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"

    inputs = processor(
        text=texts, images=image_inputs,
        return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            output_scores=True, return_dict_in_generate=True)

    # Decode each sample
    sequences = outputs.sequences  # (batch, input_len + new_tokens)
    results = []

    # Pre-compute FAKE/REAL token IDs once
    fake_ids = processor.tokenizer.encode("FAKE", add_special_tokens=False)
    real_ids = processor.tokenizer.encode("REAL", add_special_tokens=False)
    fake_tok = fake_ids[0] if fake_ids else None
    real_tok = real_ids[0] if real_ids else None

    for i in range(len(samples)):
        # Generated portion is everything after input_len
        gen_ids = sequences[i, input_len:]
        response = processor.decode(
            gen_ids, skip_special_tokens=True).strip()

        # First-token score from logits
        score = None
        if (outputs.scores and fake_tok is not None and real_tok is not None):
            first_logits = outputs.scores[0][i]  # (vocab,)
            pair = torch.stack(
                [first_logits[fake_tok], first_logits[real_tok]])
            probs = torch.softmax(pair.float(), dim=0)
            score = float(probs[0].cpu())

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
    p.add_argument("--output", default="eval_results_batch.json")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=24,
                    help="True batch size for parallel inference. "
                         "Default 24, OOM falls back to --oom-fallback-batch.")
    p.add_argument("--oom-fallback-batch", type=int, default=16,
                    help="Chunk size to retry within a batch after a primary-batch "
                         "OOM, before final single-sample fallback.")
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

    # Set left padding for generation
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    gpu_used = torch.cuda.max_memory_allocated() / 1e9
    print(f"Model loaded. GPU peak after load: {gpu_used:.1f} GB")

    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    if args.limit > 0:
        test_data = test_data[:args.limit]

    n_total = len(test_data)
    print(f"Evaluating {n_total} samples with batch_size={args.batch_size}...")

    # ── accumulators ────────────────────────────────
    results_detail = []
    bin_labels, bin_scores = [], []
    ml_labels, ml_scores = [], []
    ious, iou_flags_50, iou_flags_75 = [], [], []
    tok_precisions, tok_recalls = [], []

    per_class = defaultdict(lambda: {
        "count": 0, "verdict_ok": 0, "category_ok": 0, "parsed": 0})

    t0 = time.perf_counter()
    processed = 0

    # ── Batch loop ──
    for batch_start in range(0, n_total, args.batch_size):
        batch = test_data[batch_start:batch_start + args.batch_size]

        try:
            batch_out = run_batch_true(
                model, processor, batch, args.media_dir, args.max_tokens)
        except Exception as e:
            # Primary batch failed (OOM or other). Retry in chunks of
            # --oom-fallback-batch, then single-sample as final fallback.
            torch.cuda.empty_cache()
            print(f"  [WARN] batch={len(batch)} failed: {e}, "
                  f"retrying chunks of {args.oom_fallback_batch}")
            batch_out = []
            for ck in range(0, len(batch), args.oom_fallback_batch):
                chunk = batch[ck:ck + args.oom_fallback_batch]
                try:
                    batch_out.extend(run_batch_true(
                        model, processor, chunk,
                        args.media_dir, args.max_tokens))
                except Exception as e2:
                    torch.cuda.empty_cache()
                    print(f"  [WARN] fallback chunk={len(chunk)} also failed: "
                          f"{e2}, retrying single-sample mode")
                    for s in chunk:
                        try:
                            r = run_batch_true(
                                model, processor, [s],
                                args.media_dir, args.max_tokens)
                            batch_out.append(r[0])
                        except Exception:
                            batch_out.append(("", None))

        # ── process each result ──
        for i, sample in enumerate(batch):
            gt_cls = str(sample.get("fake_cls", "")).lower()
            gt_verdict = "FAKE" if gt_cls != "orig" else "REAL"
            gt_box = sample.get("fake_image_box", [])
            gt_pos = sample.get("fake_text_pos", [])
            bin_label = 1 if gt_verdict == "FAKE" else 0

            per_class[gt_cls]["count"] += 1
            raw, score = batch_out[i]
            parsed = parse_5line(raw) if raw else None

            if score is not None:
                bin_labels.append(bin_label)
                bin_scores.append(score)

            if parsed:
                pred_cls = parsed.get("category", "")
                ml_labels.append(
                    [1.0 if c == gt_cls else 0.0 for c in ALL_CLASSES])
                ml_scores.append(
                    [1.0 if c == pred_cls else 0.0 for c in ALL_CLASSES])

                per_class[gt_cls]["parsed"] += 1
                if parsed.get("verdict", "") == gt_verdict:
                    per_class[gt_cls]["verdict_ok"] += 1
                if parsed.get("category", "") == gt_cls:
                    per_class[gt_cls]["category_ok"] += 1

                pred_box = parsed.get("fake_image_box_pred")
                if pred_box is not None:
                    iou = box_iou(pred_box, gt_box)
                    if iou is not None:
                        ious.append(iou)
                        iou_flags_50.append(1.0 if iou > 0.5 else 0.0)
                        iou_flags_75.append(1.0 if iou > 0.75 else 0.0)

                pred_pos = parsed.get("fake_text_pos_pred")
                pr, rc, _ = pos_prf(pred_pos, gt_pos)
                tok_precisions.append(pr)
                tok_recalls.append(rc)

            results_detail.append({
                "id": sample.get("id", batch_start + i),
                "gt_verdict": gt_verdict,
                "gt_category": gt_cls,
                "bin_score": score,
                "raw_output": raw[:500],
                "parsed": parsed,
            })

        processed += len(batch)
        elapsed = time.perf_counter() - t0
        if processed % (args.batch_size * 10) == 0 or processed >= n_total:
            print(f"  [{processed}/{n_total}] "
                  f"{100*processed/n_total:.1f}% | "
                  f"{elapsed/processed:.2f}s/sample | "
                  f"batch_throughput={args.batch_size/(elapsed/(processed/args.batch_size)):.1f} sample/s")

    elapsed = time.perf_counter() - t0

    # ── Compute all 12 metrics ──
    metrics = {"samples": n_total,
               "batch_size": args.batch_size,
               "inference_time_s": elapsed,
               "seconds_per_sample": elapsed / max(1, n_total)}

    # Binary
    if bin_labels and bin_scores:
        try:
            metrics["AUC"] = float(roc_auc_score(bin_labels, bin_scores))
            metrics["EER"] = float(compute_eer(
                np.array(bin_labels), np.array(bin_scores)))
        except Exception:
            metrics["AUC"] = metrics["EER"] = None
        total_parsed = sum(v["parsed"] for v in per_class.values())
        total_correct = sum(v["verdict_ok"] for v in per_class.values())
        metrics["ACC"] = total_correct / max(1, total_parsed)
    else:
        metrics["AUC"] = metrics["EER"] = metrics["ACC"] = None

    # Multi-Label
    if ml_labels and ml_scores:
        ml_l = np.array(ml_labels)
        ml_s = np.array(ml_scores)
        try:
            metrics["mAP"] = float(average_precision_score(
                ml_l, ml_s, average="macro"))
            metrics["CF1"] = float(f1_score(
                ml_l, ml_s, average="macro", zero_division=0))
            metrics["OF1"] = float(f1_score(
                ml_l, ml_s, average="micro", zero_division=0))
        except Exception:
            metrics["mAP"] = metrics["CF1"] = metrics["OF1"] = None
    else:
        metrics["mAP"] = metrics["CF1"] = metrics["OF1"] = None

    # Image grounding
    if ious:
        metrics["IoUmean"] = float(np.mean(ious))
        metrics["IoU50"]  = float(np.mean(iou_flags_50))
        metrics["IoU75"]  = float(np.mean(iou_flags_75))
    else:
        metrics["IoUmean"] = metrics["IoU50"] = metrics["IoU75"] = None

    # Text grounding
    if tok_precisions:
        metrics["Tok_Precision"] = float(np.mean(tok_precisions))
        metrics["Tok_Recall"]    = float(np.mean(tok_recalls))
        p_avg = metrics["Tok_Precision"]
        r_avg = metrics["Tok_Recall"]
        metrics["Tok_F1"] = (2 * p_avg * r_avg / (p_avg + r_avg)
                             if (p_avg + r_avg) else 0.0)
    else:
        metrics["Tok_Precision"] = metrics["Tok_Recall"] = metrics["Tok_F1"] = None

    # Per-class
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

    # Save
    out_path = os.path.join(os.path.dirname(args.adapter), args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": metrics, "details": results_detail},
                  f, ensure_ascii=False, indent=2)

    gpu_peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n{'='*65}")
    print(f"  DGM4 Table 2 — Full Evaluation Metrics (batch={args.batch_size})")
    print(f"{'='*65}")
    print("  [1] Binary Classification")
    print(f"      AUC : {metrics.get('AUC')}")
    print(f"      EER : {metrics.get('EER')}")
    print(f"      ACC : {metrics.get('ACC')}")
    print("  [2] Multi-Label Classification")
    print(f"      mAP : {metrics.get('mAP')}")
    print(f"      CF1 : {metrics.get('CF1')}")
    print(f"      OF1 : {metrics.get('OF1')}")
    print("  [3] Image Grounding")
    print(f"      IoUmean : {metrics.get('IoUmean')}")
    print(f"      IoU50   : {metrics.get('IoU50')}")
    print(f"      IoU75   : {metrics.get('IoU75')}")
    print("  [4] Text Grounding")
    print(f"      Tok_Precision : {metrics.get('Tok_Precision')}")
    print(f"      Tok_Recall    : {metrics.get('Tok_Recall')}")
    print(f"      Tok_F1        : {metrics.get('Tok_F1')}")
    print(f"{'='*65}")
    print(f"  Time: {elapsed:.0f}s | {elapsed/max(1,n_total):.2f}s/sample | "
          f"GPU peak: {gpu_peak:.1f} GB")
    print(f"  Results: {out_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
