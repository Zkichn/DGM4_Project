#!/usr/bin/env python3
"""Recompute DGM4 paper-style atomic multi-label metrics.

The generation evaluator stores one predicted category string per sample, such
as ``face_swap&text_swap``.  DGM4 Table 2 reports multi-label metrics over four
atomic manipulation labels instead:

    FS = face_swap, FA = face_attribute, TS = text_swap, TA = text_attribute

This script expands both ground-truth and predicted category strings into those
four atomic labels and recomputes mAP / CF1 / OF1 from saved eval_results JSON.
When only generated categories are available, mAP is computed from hard 0/1
predictions, so it should be reported as "hard mAP" rather than probability AP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support


ATOMIC_LABELS = ("face_swap", "face_attribute", "text_swap", "text_attribute")
ATOMIC_SHORT = {
    "face_swap": "FS",
    "face_attribute": "FA",
    "text_swap": "TS",
    "text_attribute": "TA",
}

DEFAULT_EXPERIMENTS = {
    "3ep baseline": "eval_results__lora-sft-fast.json",
    "r2 adapter 3+2ep": "eval_results__lora-sft-fast-r2.json",
    "5ep resume 3+2ep": "eval_results__lora-sft-5ep-resume.json",
}


def category_to_atoms(category: str | None) -> list[int]:
    """Expand a DGM4 category string to FS/FA/TS/TA multi-hot labels."""
    category = (category or "orig").strip().lower()
    if category == "orig":
        parts: set[str] = set()
    else:
        parts = {p.strip() for p in category.split("&") if p.strip()}
    return [1 if label in parts else 0 for label in ATOMIC_LABELS]


def load_eval(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_atomic_metrics(eval_data: dict[str, Any]) -> dict[str, Any]:
    y_true: list[list[int]] = []
    y_pred: list[list[int]] = []
    skipped = 0

    for row in eval_data.get("details", []):
        parsed = row.get("parsed")
        if not parsed:
            skipped += 1
            continue
        y_true.append(category_to_atoms(row.get("gt_category")))
        y_pred.append(category_to_atoms(parsed.get("category")))

    if not y_true:
        raise ValueError("No parsed samples found in eval result.")

    yt = np.asarray(y_true, dtype=np.int32)
    yp = np.asarray(y_pred, dtype=np.int32)

    per_label: dict[str, Any] = {}
    p, r, f1, support = precision_recall_fscore_support(
        yt, yp, average=None, zero_division=0
    )
    ap = average_precision_score(yt, yp, average=None)

    for idx, label in enumerate(ATOMIC_LABELS):
        per_label[label] = {
            "short": ATOMIC_SHORT[label],
            "support": int(support[idx]),
            "predicted_positive": int(yp[:, idx].sum()),
            "precision": float(p[idx]),
            "recall": float(r[idx]),
            "f1": float(f1[idx]),
            "hard_ap": float(ap[idx]),
        }

    exact_match = np.all(yt == yp, axis=1).mean()
    return {
        "samples": int(len(eval_data.get("details", []))),
        "parsed_samples": int(len(y_true)),
        "skipped_unparsed": int(skipped),
        "hard_mAP": float(average_precision_score(yt, yp, average="macro")),
        "CF1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "OF1_micro": float(f1_score(yt, yp, average="micro", zero_division=0)),
        "atomic_exact_match": float(exact_match),
        "per_label": per_label,
    }


def fmt(x: float) -> str:
    return f"{x:.4f}"


def build_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Atomic Multi-Label Metrics",
        "",
        "Ground truth and predicted category strings are expanded to four DGM4 atomic labels: FS, FA, TS, TA.",
        "Because saved eval files only contain generated hard categories, mAP here is hard mAP, not probability-based AP.",
        "",
        "## Overall",
        "",
        "| Experiment | samples | parsed | hard mAP | CF1 macro | OF1 micro | atomic exact match |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name, metrics in results.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(metrics["samples"]),
                    str(metrics["parsed_samples"]),
                    fmt(metrics["hard_mAP"]),
                    fmt(metrics["CF1_macro"]),
                    fmt(metrics["OF1_micro"]),
                    fmt(metrics["atomic_exact_match"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per Atomic Label",
            "",
            "| Experiment | Label | support | pred + | Precision | Recall | F1 | hard AP |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in results.items():
        for label in ATOMIC_LABELS:
            item = metrics["per_label"][label]
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        item["short"],
                        str(item["support"]),
                        str(item["predicted_positive"]),
                        fmt(item["precision"]),
                        fmt(item["recall"]),
                        fmt(item["f1"]),
                        fmt(item["hard_ap"]),
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("training_models/outputs/qwen3-vl-8b/dgm4-instruct"),
        help="Directory containing eval_results__*.json files.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <results-dir>/atomic_multilabel_metrics.json.",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=None,
        help="Output Markdown path. Defaults to <results-dir>/atomic_multilabel_metrics.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    out_json = args.json_output or results_dir / "atomic_multilabel_metrics.json"
    out_md = args.md_output or results_dir / "atomic_multilabel_metrics.md"

    results: dict[str, Any] = {}
    for name, filename in DEFAULT_EXPERIMENTS.items():
        path = results_dir / filename
        results[name] = compute_atomic_metrics(load_eval(path))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(results), encoding="utf-8")

    print(build_markdown(results))
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote Markdown: {out_md}")


if __name__ == "__main__":
    main()
