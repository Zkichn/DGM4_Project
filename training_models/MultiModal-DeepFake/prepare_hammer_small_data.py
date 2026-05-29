#!/usr/bin/env python
import argparse
import json
from collections import Counter
from pathlib import Path

FIELDS = ["id", "image", "text", "fake_cls", "fake_image_box", "fake_text_pos", "mtcnn_boxes"]
VALID = {
    "orig", "face_swap", "face_attribute", "text_swap", "text_attribute",
    "face_swap&text_swap", "face_swap&text_attribute",
    "face_attribute&text_swap", "face_attribute&text_attribute",
}

def clean_item(item):
    row = {k: item.get(k) for k in FIELDS if k in item}
    row.setdefault("fake_cls", "orig")
    row.setdefault("fake_image_box", [])
    row.setdefault("fake_text_pos", [])
    row.setdefault("mtcnn_boxes", [])
    if row["fake_cls"] not in VALID:
        raise ValueError(f"bad fake_cls={row['fake_cls']!r} id={row.get('id')}")
    if not isinstance(row["fake_image_box"], list):
        row["fake_image_box"] = []
    if not isinstance(row["fake_text_pos"], list):
        row["fake_text_pos"] = []
    return row

def convert_one(src: Path, dst: Path):
    data = json.loads(src.read_text(encoding="utf-8"))
    out = [clean_item(x) for x in data]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    cnt = Counter(x["fake_cls"] for x in out)
    return {"samples": len(out), "class_counts": dict(sorted(cnt.items()))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", default="../datasets/dgm4_instruct")
    ap.add_argument("--output_dir", default="hammer_small_dgm4/metadata")
    args = ap.parse_args()
    source = Path(args.source_dir)
    output = Path(args.output_dir)
    summary = {}
    for split in ["train", "val", "test"]:
        summary[split] = convert_one(source / f"{split}.json", output / f"{split}.json")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
