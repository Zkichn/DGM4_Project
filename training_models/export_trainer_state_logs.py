#!/usr/bin/env python3
"""Export Trainer state history into centralized log files.

This does not recover deleted raw terminal logs. It collects the structured
loss/eval/save history that Hugging Face Trainer kept in trainer_state.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def iter_trainer_states(outputs_dir: Path):
    for path in sorted(outputs_dir.rglob("trainer_state.json")):
        yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="training_models directory")
    parser.add_argument("--outputs", default="outputs", help="outputs directory under root")
    parser.add_argument("--logs", default="logs", help="logs directory under root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outputs_dir = (root / args.outputs).resolve()
    logs_dir = (root / args.logs).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for state_path in iter_trainer_states(outputs_dir):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic utility
            rows.append(
                {
                    "source": str(state_path.relative_to(root)),
                    "error": repr(exc),
                }
            )
            continue

        rel = state_path.relative_to(root)
        output_dir = rel.parent
        for idx, item in enumerate(state.get("log_history", [])):
            row = {"source": str(rel), "output_dir": str(output_dir), "log_index": idx}
            for key, value in item.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    row[key] = value
                else:
                    row[key] = json.dumps(value, ensure_ascii=False)
            rows.append(row)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = logs_dir / f"trainer_state_history_{stamp}.json"
    csv_path = logs_dir / f"trainer_state_history_{stamp}.csv"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    if fieldnames:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    print(f"Exported {len(rows)} trainer-state log rows")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
