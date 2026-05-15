import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


# --- Default paths ---
# Cleaned JSONL file with structured rationale and grounding fields.
DEFAULT_INPUT_JSONL = (
    r"C:\Users\Zkichn\Desktop\learn_AI\DGM4_IDEA\versions"
    r"\v3_2026-05-15\data\train_label_v3_grounded.jsonl"
)

# Output ShareGPT-style SFT data. The output is a JSON array.
DEFAULT_OUTPUT_JSON = (
    r"C:\Users\Zkichn\Desktop\learn_AI\DGM4_IDEA\versions"
    r"\v3_2026-05-15\data\qwen_sft_data_v3_grounded_llamafactory.json"
)

# Image root directory. Only used by --require_existing_images.
DEFAULT_ROOT_DIR = "D:\\"
DEFAULT_SYSTEM_PROMPT_TXT = (
    r"C:\Users\Zkichn\Desktop\learn_AI\DGM4_IDEA\versions"
    r"\v3_2026-05-15\data\qwen_sft_system_prompt_v3.txt"
)


# --- Prompt pool ---
# System prompt carries the stable role, task boundary, and output schema.
# User prompts stay natural and diverse: they provide the image, caption, and request.
SYSTEM_PROMPT = (
    "You are a multimodal forensic assistant. Your task is to assess whether an "
    "image-caption pair is authentic or manipulated. Check both visual evidence "
    "and caption-image consistency. If the sample is fake, identify the manipulation "
    "category and provide exact grounding fields for the manipulated image region "
    "and/or manipulated text token positions. If a modality is not manipulated, "
    "output an empty list for that grounding field.\n\n"
    "Always respond strictly in this five-line format:\n"
    "Verdict: [REAL or FAKE]\n"
    "Category: [category]\n"
    "Fake Image Box: [box or []]\n"
    "Fake Text Pos: [positions or []]\n"
    "Evidence: [concise grounded explanation]"
)


USER_TEMPLATES = [
    (
        "Caption: \"{text}\"\n"
        "Please assess whether this image-caption pair is authentic or manipulated."
    ),
    (
        "Here is the caption: \"{text}\"\n"
        "Does the image naturally match the caption, or is there any sign of manipulation?"
    ),
    (
        "Caption: \"{text}\"\n"
        "Please check the visual content and the caption for possible forgery or inconsistency."
    ),
    (
        "The caption for this image is: \"{text}\"\n"
        "Can you determine whether this sample is real or fake?"
    ),
    (
        "Caption: \"{text}\"\n"
        "Analyze whether the image, the caption, or both have been manipulated."
    ),
    (
        "Please inspect this image with its caption: \"{text}\"\n"
        "Is the image-caption pair trustworthy?"
    ),
    (
        "Caption: \"{text}\"\n"
        "Check whether the caption is consistent with the image and whether any modality is manipulated."
    ),
    (
        "This image is paired with the caption: \"{text}\"\n"
        "Please perform a multimodal authenticity check."
    ),
    (
        "Caption: \"{text}\"\n"
        "Look for visual tampering, text tampering, or image-caption mismatch."
    ),
    (
        "Given this caption: \"{text}\"\n"
        "Please judge the authenticity of the image-caption pair."
    ),
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                print(f"[WARN] skip invalid JSON at line {line_no}: {exc}")
                continue
            if isinstance(obj, dict):
                data.append(obj)
    return data


def build_abs_image_path(root_dir: str, image_path: str) -> str:
    if os.path.isabs(image_path):
        return image_path.replace("\\", "/")
    return os.path.join(root_dir, image_path).replace("\\", "/")


def compact_text(text: str) -> str:
    return " ".join(str(text).split())


def has_required_rationale_format(rationale: str) -> bool:
    lines = rationale.splitlines()
    return (
        len(lines) == 5
        and lines[0].startswith("Verdict:")
        and lines[1].startswith("Category:")
        and lines[2].startswith("Fake Image Box:")
        and lines[3].startswith("Fake Text Pos:")
        and lines[4].startswith("Evidence:")
    )


def build_conversation_item(
    obj: Dict[str, Any],
    root_dir: str,
    rng: random.Random,
    valid_count: int,
    require_existing_images: bool,
    system_mode: str,
    keep_newlines: bool,
) -> Optional[Dict[str, Any]]:
    rationale = str(obj.get("rationale", "")).strip()
    if not rationale or "error" in obj:
        return None

    if not has_required_rationale_format(rationale):
        return None

    img_rel_path = str(obj.get("image", "")).strip()
    if not img_rel_path:
        return None

    abs_img_path = build_abs_image_path(root_dir, img_rel_path)
    if require_existing_images and not os.path.exists(abs_img_path):
        return None

    text_caption = str(obj.get("text", "")).strip()
    instruct = rng.choice(USER_TEMPLATES).format(text=text_caption)
    user_prompt = f"<image>\n{instruct}"
    if not keep_newlines:
        user_prompt = compact_text(user_prompt)
        rationale = compact_text(rationale)
        system_prompt = compact_text(SYSTEM_PROMPT)
    else:
        system_prompt = SYSTEM_PROMPT

    item: Dict[str, Any] = {
        "id": obj.get("id", f"sample_{valid_count}"),
        "image": obj.get("image", ""),
        "text": obj.get("text", ""),
        "fake_cls": obj.get("fake_cls", ""),
        "fake_modality": obj.get("fake_modality", ""),
        "fake_image_box": obj.get("fake_image_box", []),
        "fake_text_pos": obj.get("fake_text_pos", []),
        "conversations": [],
    }

    if system_mode == "per_sample":
        item["conversations"].append({"from": "system", "value": system_prompt})
    item["conversations"].append({"from": "user", "value": user_prompt})
    item["conversations"].append({"from": "assistant", "value": rationale})
    return item


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert grounded DGM4 JSONL into ShareGPT-style SFT data."
    )
    parser.add_argument("--input_jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--root_dir", default=DEFAULT_ROOT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means no limit.")
    parser.add_argument(
        "--system_mode",
        choices=["none", "per_sample"],
        default="none",
        help=(
            "none: LLaMA-Factory style, omit system from each sample and write it to --system_prompt_txt. "
            "per_sample: include the system prompt in every conversation."
        ),
    )
    parser.add_argument("--system_prompt_txt", default=DEFAULT_SYSTEM_PROMPT_TXT)
    parser.add_argument(
        "--keep_newlines",
        action="store_true",
        help="Keep newlines in prompts and assistant answers. Default compacts them into one line.",
    )
    parser.add_argument(
        "--require_existing_images",
        action="store_true",
        help="Skip samples whose resolved image path does not exist. Output still keeps relative image path.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Use 0 for compact JSON.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print("开始转换 SFT 训练数据...")
    print(f"输入文件: {args.input_jsonl}")
    print(f"输出文件: {args.output_json}")

    raw_data = load_jsonl(args.input_jsonl)
    sft_data: List[Dict[str, Any]] = []
    skipped = 0

    for obj in raw_data:
        if args.max_samples and len(sft_data) >= args.max_samples:
            break

        item = build_conversation_item(
            obj=obj,
            root_dir=args.root_dir,
            rng=rng,
            valid_count=len(sft_data),
            require_existing_images=args.require_existing_images,
            system_mode=args.system_mode,
            keep_newlines=args.keep_newlines,
        )
        if item is None:
            skipped += 1
            continue

        sft_data.append(item)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.system_mode == "none":
        system_prompt_path = Path(args.system_prompt_txt)
        system_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        system_prompt_path.write_text(compact_text(SYSTEM_PROMPT), encoding="utf-8")

    with output_path.open("w", encoding="utf-8") as out_f:
        if args.indent and args.indent > 0:
            json.dump(sft_data, out_f, ensure_ascii=False, indent=args.indent)
        else:
            json.dump(sft_data, out_f, ensure_ascii=False, separators=(",", ":"))

    print(f"转换完毕，共计有效对话数据: {len(sft_data)} 条")
    print(f"跳过样本: {skipped} 条")
    print(f"SFT 数据已生成: {output_path}")
    if args.system_mode == "none":
        print(f"System prompt 已单独保存: {args.system_prompt_txt}")


if __name__ == "__main__":
    main()
