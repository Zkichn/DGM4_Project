#!/usr/bin/env python3
"""Build DGM4 curriculum v2 datasets.

v2 changes:
- Stage 1 predicts Verdict + four atomic labels + Category.
- Stage 2 keeps those fields and mixes grounding subtasks:
  full Image+Text grounding, image-only grounding, and text-only grounding.
- Train splits are deterministically resampled to emphasize weak/rare classes
  and text/both grounding samples.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CLASS_WEIGHTS = {
    "orig": 0.8,
    "face_swap": 1.0,
    "face_attribute": 1.2,
    "text_swap": 1.5,
    "text_attribute": 2.5,
    "face_swap&text_swap": 2.0,
    "face_attribute&text_swap": 2.5,
    "face_swap&text_attribute": 3.0,
    "face_attribute&text_attribute": 3.0,
}

GROUND_TEXT_BONUS = 1.5
GROUND_BOTH_BONUS = 1.5

ATOMIC_LABELS = ("face_swap", "face_attribute", "text_swap", "text_attribute")
ATOMIC_KEYS = ("FS", "FA", "TS", "TA")


def category_to_atoms(category: str) -> dict[str, int]:
    parts = set() if category == "orig" else set(category.split("&"))
    return {key: int(label in parts) for key, label in zip(ATOMIC_KEYS, ATOMIC_LABELS)}


def yes_no(value: int) -> str:
    return "1" if value else "0"


def stage1_user(sample: dict[str, Any]) -> str:
    return (
        f'<image> Caption: "{sample.get("text", "")}"\n'
        "Determine whether this image-caption pair is authentic or manipulated. "
        "Predict the four atomic manipulation labels and the final category."
    )


def stage2_user(sample: dict[str, Any], task: str) -> str:
    base = (
        f'<image> Caption: "{sample.get("text", "")}"\n'
        "Determine authenticity, atomic manipulation labels, and the final category. "
    )
    if task == "image":
        return base + "Then predict only the manipulated image box. If image is not manipulated, output []."
    if task == "text":
        return (
            base
            + "Then predict only the minimal manipulated text token positions. "
            "Do not include unchanged surrounding words. If text is not manipulated, output []."
        )
    return (
        base
        + "Then predict both manipulated image box and minimal manipulated text token positions. "
        "Do not include unchanged surrounding words. If a modality is not manipulated, output []."
    )


def stage1_answer(sample: dict[str, Any]) -> str:
    category = sample.get("fake_cls", "orig")
    atoms = category_to_atoms(category)
    verdict = "REAL" if category == "orig" else "FAKE"
    return "\n".join(
        [
            f"Verdict: {verdict}",
            f"FS: {yes_no(atoms['FS'])}",
            f"FA: {yes_no(atoms['FA'])}",
            f"TS: {yes_no(atoms['TS'])}",
            f"TA: {yes_no(atoms['TA'])}",
            f"Category: {category}",
        ]
    )


def stage2_answer(sample: dict[str, Any], task: str) -> str:
    lines = stage1_answer(sample).splitlines()
    if task in ("image", "full"):
        lines.append(f"Fake Image Box: {sample.get('fake_image_box', [])}")
    if task in ("text", "full"):
        lines.append(f"Fake Text Pos: {sample.get('fake_text_pos', [])}")
    return "\n".join(lines)


def convert_sample(sample: dict[str, Any], stage: str, task: str = "full") -> dict[str, Any]:
    item = copy.deepcopy(sample)
    if stage == "stage1":
        user = stage1_user(sample)
        assistant = stage1_answer(sample)
    elif stage == "stage2":
        user = stage2_user(sample, task)
        assistant = stage2_answer(sample, task)
        item["grounding_task"] = task
    else:
        raise ValueError(stage)
    item["conversations"] = [{"from": "user", "value": user}, {"from": "assistant", "value": assistant}]
    item["images"] = item.get("images") or [item.get("image")]
    return item


def weighted_copies(sample: dict[str, Any], stage: str, rng: random.Random) -> int:
    category = sample.get("fake_cls", "orig")
    weight = CLASS_WEIGHTS.get(category, 1.0)
    if stage == "stage2":
        if sample.get("fake_text_pos"):
            weight *= GROUND_TEXT_BONUS
        if sample.get("fake_image_box") and sample.get("fake_text_pos"):
            weight *= GROUND_BOTH_BONUS
    base = int(weight)
    frac = weight - base
    return base + int(rng.random() < frac)


def choose_stage2_task(sample: dict[str, Any], copy_index: int) -> str:
    has_box = bool(sample.get("fake_image_box"))
    has_text = bool(sample.get("fake_text_pos"))
    if has_box and has_text:
        return ("full", "image", "text", "full")[copy_index % 4]
    if has_box:
        return ("full", "image")[copy_index % 2]
    if has_text:
        return ("full", "text")[copy_index % 2]
    return "full"


def resample_train(data: list[dict[str, Any]], stage: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for sample in data:
        copies = weighted_copies(sample, stage, rng)
        for idx in range(copies):
            task = choose_stage2_task(sample, idx) if stage == "stage2" else "full"
            out.append(convert_sample(sample, stage, task))
    rng.shuffle(out)
    return out


def summarize(data: list[dict[str, Any]]) -> dict[str, Any]:
    classes = Counter(x.get("fake_cls", "orig") for x in data)
    tasks = Counter(x.get("grounding_task", "n/a") for x in data)
    atoms = Counter()
    for sample in data:
        for key, value in category_to_atoms(sample.get("fake_cls", "orig")).items():
            atoms[key] += value
    return {
        "samples": len(data),
        "classes": dict(classes),
        "atomic": dict(atoms),
        "grounding_tasks": dict(tasks),
    }


def write_dataset_info(out_dir: Path, prefix: str) -> None:
    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }

    info = {
        f"{prefix}_train": entry("train.json"),
        f"{prefix}_val": entry("val.json"),
        f"{prefix}_test": entry("test.json"),
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_stage(source_dir: Path, out_dir: Path, stage: str, prefix: str, seed: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"stage": stage, "source_dir": str(source_dir), "splits": {}}
    for split in ("train", "val", "test"):
        data = json.loads((source_dir / f"{split}.json").read_text(encoding="utf-8"))
        if split == "train":
            converted = resample_train(data, stage, seed)
        else:
            converted = [convert_sample(x, stage, "full") for x in data]
        (out_dir / f"{split}.json").write_text(json.dumps(converted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["splits"][split] = summarize(converted)
    write_dataset_info(out_dir, prefix)
    (out_dir / "sampling_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--seed", type=int, default=20260520)
    args = parser.parse_args()

    root = Path(args.root)
    datasets = root / "datasets"
    source = datasets / "dgm4_stage3_final"
    if not source.exists():
        source = datasets / "dgm4_instruct"

    reports = [
        build_stage(source, datasets / "dgm4_stage1_cls_v2", "stage1", "dgm4_stage1_cls_v2", args.seed),
        build_stage(source, datasets / "dgm4_stage2_grounding_v2", "stage2", "dgm4_stage2_grounding_v2", args.seed),
    ]
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
