import argparse
import json
import re
from pathlib import Path


VERDICT_RE = re.compile(r"Verdict\s*:\s*(REAL|FAKE)", re.IGNORECASE)


def read_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def get_text(obj):
    for key in ("predict", "prediction", "generated_text", "response", "output"):
        value = obj.get(key)
        if isinstance(value, str):
            return value
    return json.dumps(obj, ensure_ascii=False)


def parse_verdict(text):
    match = VERDICT_RE.search(text or "")
    if not match:
        return None
    return match.group(1).upper()


def gold_from_sample(sample):
    fake_cls = sample.get("fake_cls", "")
    return "REAL" if fake_cls == "orig" else "FAKE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to generated_predictions.jsonl from LLaMA-Factory.")
    parser.add_argument("--gold", required=True, help="Path to the test.json file used for prediction.")
    parser.add_argument("--out", default=None, help="Optional JSON metrics output path.")
    args = parser.parse_args()

    preds = read_json_or_jsonl(Path(args.pred))
    gold = read_json_or_jsonl(Path(args.gold))
    n = min(len(preds), len(gold))
    if n == 0:
        raise SystemExit("No prediction/gold samples found.")

    tp = tn = fp = fn = invalid = 0
    examples = []

    for idx in range(n):
        gold_label = gold_from_sample(gold[idx])
        pred_text = get_text(preds[idx])
        pred_label = parse_verdict(pred_text)
        if pred_label is None:
            invalid += 1
        elif pred_label == "FAKE" and gold_label == "FAKE":
            tp += 1
        elif pred_label == "REAL" and gold_label == "REAL":
            tn += 1
        elif pred_label == "FAKE" and gold_label == "REAL":
            fp += 1
        elif pred_label == "REAL" and gold_label == "FAKE":
            fn += 1

        if pred_label != gold_label and len(examples) < 20:
            examples.append(
                {
                    "idx": idx,
                    "id": gold[idx].get("id"),
                    "image": gold[idx].get("image"),
                    "fake_cls": gold[idx].get("fake_cls"),
                    "gold": gold_label,
                    "pred": pred_label,
                    "prediction_text": pred_text,
                }
            )

    valid = tp + tn + fp + fn
    total = valid + invalid
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics = {
        "total_compared": total,
        "valid_predictions": valid,
        "invalid_predictions": invalid,
        "accuracy": accuracy,
        "fake_precision": precision,
        "fake_recall": recall,
        "fake_f1": f1,
        "confusion_matrix": {
            "tp_fake_as_fake": tp,
            "tn_real_as_real": tn,
            "fp_real_as_fake": fp,
            "fn_fake_as_real": fn,
        },
        "first_mismatches": examples,
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
