import argparse
import asyncio
import base64
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from PIL import Image
from tqdm.asyncio import tqdm
from transformers import BertTokenizerFast
from prompt_builder import (
    BertTokenView,
    has_valid_box,
    build_prompt_v2,
    build_messages_multimodal
)


# -----------------------------
# Utilities: IO (json list / jsonl)  文件读写工具 (Utilities: IO)
# -----------------------------
def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(data)}")
    return data

# 全局文件锁，防止异步写入时文件内容错乱
file_lock = asyncio.Lock()
# 异步追加写入。把大模型生成的结果（obj）单行写入到 .jsonl 文件中
async def async_append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    async with file_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_done_ids_from_jsonl(path: str) -> set:
    """
    读取已经处理过的结果文件，提取出所有已完成的样本 id。
    这是为了实现断点续传功能，如果程序意外中断，下次重启时会自动跳过这些已处理的数据。
    """
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(obj["id"])
            except Exception:
                continue
    return done


# -----------------------------
# Utilities: image handling
# -----------------------------
def read_image_bytes(path: str) -> Tuple[bytes, str]:
    """
    直接读取图像的二进制数据，并根据后缀名推断 MIME 类型
    """
    suffix = os.path.splitext(path)[1].lower()
    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix in [".png"]:
        mime = "image/png"
    else:
        mime = "image/jpeg"

    with open(path, "rb") as f:
        b = f.read()
    return b, mime

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def crop_with_margin(img: Image.Image, bbox_xyxy: List[float], margin_ratio: float = 0.25) -> Image.Image:
   """"
   带边距的图像裁剪。如果数据集中标注了“图像篡改区域（bounding box）”，
   这个函数不仅会把那个区域裁下来，还会额外向外扩展一定的比例（默认 25%）。
   这样提供给大模型时，模型既能看到篡改细节，又能看到周围的真实背景，便于对比找出 PS 痕迹。
   """""
   if not (isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4):
        return img
   try:
        x1, y1, x2, y2 = [float(x) for x in bbox_xyxy]
   except Exception:
       return img

   w = max(1.0, x2 - x1)
   h = max(1.0, y2 - y1)
   mx = w * margin_ratio
   my = h * margin_ratio

   W, H = img.size
   cx1 = int(clamp(x1 - mx, 0, W))
   cy1 = int(clamp(y1 - my, 0, H))
   cx2 = int(clamp(x2 + mx, 0, W))
   cy2 = int(clamp(y2 + my, 0, H))

   if cx2 <= cx1 or cy2 <= cy1:
       return img
   return img.crop((cx1, cy1, cx2, cy2))

#将处理后（如裁剪后）的 PIL 图像对象重新转换为二进制流
def image_to_bytes(img: Image.Image, prefer: str = "JPEG") -> Tuple[bytes, str]:
    import io
    buf = io.BytesIO()
    if prefer.upper() == "PNG":
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    else:
        img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), "image/jpeg"
#将图像二进制流转换为 Base64 编码的 Data URL 格式（例如 data:image/jpeg;base64,...）。
# 这是目前大多数 OpenAI 兼容格式的视觉大模型接收图像的标准方式。
def to_data_url(img_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# -----------------------------
# Utilities: fake_image_box / fake_text_pos
# -----------------------------
#推断伪造模态。通过检查有没有合法的图片框（box）和文本位置（text_pos），
# 判断这个样本是“仅图片被改 (image)”、“仅文字被改 (text)”、“图文都被改 (both)”还是“都没改 (none)”。
def infer_fake_modality(ann: Dict[str, Any]) -> str:
    img_flag = has_valid_box(ann.get("fake_image_box", None))
    txt_flag = isinstance(ann.get("fake_text_pos", None), list) and len(ann["fake_text_pos"]) > 0

    if img_flag and txt_flag:
        return "both"
    if img_flag:
        return "image"
    if txt_flag:
        return "text"
    return "none"

#通过这个函数处理后，你就可以把生成的 marked_context 直接拼接在 Prompt 里喂给大模型。大模型一读到 - around pos 15: ...，
# 就能精准定位到是哪个具体的词出现了逻辑或事实错误，从而生成更准确的分析报告（Rationale）
def build_bert_token_view(
    tokenizer: BertTokenizerFast, text: str, fake_text_pos: List[int], window: int = 6, max_tokens_show: int = 120
) -> BertTokenView:
    enc = tokenizer(text, add_special_tokens=True, return_offsets_mapping=False)
    input_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    tokens_with_idx = [f"{i}:{tok}" for i, tok in enumerate(tokens[:max_tokens_show])]
    if len(tokens) > max_tokens_show:
        tokens_with_idx.append(f"... (total_tokens={len(tokens)})")

    contexts = []
    for p in fake_text_pos[:50]:
        try:
            p = int(p)
        except Exception:
            continue
        if p < 0 or p >= len(tokens):
            continue
        left = max(0, p - window)
        right = min(len(tokens), p + window + 1)
        seg = " ".join([f"{i}:{tokens[i]}" for i in range(left, right)])
        contexts.append(f"- around pos {p}: {seg}")

    marked_context = "\n".join(contexts) if contexts else "(no valid fake_text_pos context)"
    return BertTokenView(tokens_with_idx=tokens_with_idx, marked_context=marked_context)

# -----------------------------
# Async Bailian API call (修改后的请求函数)
# -----------------------------
async def async_bailian_chat_completion(
        session: aiohttp.ClientSession,
        api_base: str,
        api_key: str,  # <-- 新增：传入 API Key
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
) -> str:
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"  # <-- 新增：阿里云 API 鉴权头
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with session.post(url, headers=headers, json=payload) as response:
        if response.status != 200:
            text = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {text[:500]}")
        js = await response.json()

    try:
        content = js["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Bad response schema: {json.dumps(js)[:800]}")
    return (content or "").strip()


# -----------------------------
# Worker for a single annotation
# -----------------------------
# 这是处理单个图文样本的流水线。它串联了上面的所有步骤：
# 1.通过 semaphore 获取并发许可（防止一次性发起太多请求把显存撑爆）。
# 2.提取并判断伪造类型。
# 3.生成文本 Token 上下文。
# 4.读取原图（可选读取裁剪图）并生成 Base64。
# 5.生成 Prompt。
# 6.发送异步 API 请求（带有重试机制，失败会自动等待并重试，最多 3 次）。
# 7.成功或彻底失败后，将结果通过 async_append_jsonl 安全地写进本地文件。
async def process_single_annotation(
    ann: Dict[str, Any],
    args: argparse.Namespace,
    tokenizer: BertTokenizerFast,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
):
    async with semaphore: # 控制并发数量
        sample_id = ann.get("id", None)
        rel_img = ann.get("image", "")
        abs_img = os.path.join(args.root_dir, rel_img).replace("\\", "/")

        fake_cls = ann.get("fake_cls", "unknown")
        caption = ann.get("text", "")
        fake_image_box = ann.get("fake_image_box", [])
        fake_text_pos = ann.get("fake_text_pos", [])
        fake_modality = infer_fake_modality(ann)

        bert_view = None
        if fake_modality in ("text", "both"):
            pos_int = []
            if isinstance(fake_text_pos, list):
                for x in fake_text_pos:
                    try:
                        pos_int.append(int(x))
                    except Exception:
                        pass
            bert_view = build_bert_token_view(tokenizer, caption, pos_int, window=6, max_tokens_show=120)

        prompt_text = build_prompt_v2(
            fake_cls=str(fake_cls),
            fake_modality=fake_modality,
            caption=caption,
            fake_image_box=fake_image_box,
            fake_text_pos=fake_text_pos,
            bert_view=bert_view,
        )

        try:
            full_bytes, full_mime = read_image_bytes(abs_img)
            full_data_url = to_data_url(full_bytes, full_mime)
        except Exception as e:
            await async_append_jsonl(args.output_jsonl, {
                "id": sample_id, "image": rel_img, "text": caption,
                "fake_cls": fake_cls, "fake_modality": fake_modality,
                "rationale": "", "error": f"read_image_failed: {e}",
            })
            return

        crop_data_url = None
        if args.crop_bbox and has_valid_box(fake_image_box):
            try:
                img = Image.open(abs_img).convert("RGB")
                crop = crop_with_margin(img, fake_image_box, margin_ratio=args.bbox_margin)
                crop_bytes, crop_mime = image_to_bytes(crop, prefer="JPEG")
                crop_data_url = to_data_url(crop_bytes, crop_mime)
                prompt_text = "（提示：你会收到两张图，第2张是fake_image_box周围裁剪图，请优先分析第2张细节。）\n" + prompt_text
            except Exception:
                crop_data_url = None

        messages = build_messages_multimodal(prompt_text, full_data_url, crop_data_url)

        last_err = None
        for attempt in range(1, args.retry + 1):
            try:
                # <-- 修改这里：调用新函数，传入 api_key
                out = await async_bailian_chat_completion(
                    session=session,
                    api_base=args.api_base,
                    api_key=args.api_key,
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                if len(out) < 40:
                    raise RuntimeError(f"too_short_output: {out[:60]}")

                await async_append_jsonl(args.output_jsonl, {
                    "id": sample_id, "image": rel_img, "text": caption,
                    "fake_cls": fake_cls, "fake_modality": fake_modality,
                    "rationale": out,
                })
                return # 成功，退出重试循环
            except Exception as e:
                last_err = e
                await asyncio.sleep(2.0 * attempt)

        # 如果多次重试仍然失败
        if last_err is not None:
            await async_append_jsonl(args.output_jsonl, {
                "id": sample_id, "image": rel_img, "text": caption,
                "fake_cls": fake_cls, "fake_modality": fake_modality,
                "rationale": "", "error": f"local_vllm_failed: {last_err}",
            })


# -----------------------------
# Main Async Orchestrator
# -----------------------------
async def async_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, type=str)
    parser.add_argument("--root_dir", required=True, type=str)
    parser.add_argument("--output_jsonl", required=True, type=str)

    # <-- 修改点 1：使用阿里云百炼的 OpenAI 兼容地址
    parser.add_argument("--api_base", default="https://dashscope.aliyuncs.com/compatible-mode/v1", type=str)

    # <-- 修改点 2：使用百炼上的视觉模型（例如 qwen-vl-max 或 qwen-vl-plus） "qwen3-vl-235b-a22b-thinking"
    parser.add_argument("--model", default="gpt-5.5", type=str)

    # <-- 修改点 3：增加 API Key 读取，默认从环境变量读取，也可以通过命令行传入
    parser.add_argument("--api_key", default=os.environ.get("DASHSCOPE_API_KEY", ""), type=str, help="你的百炼 API Key")

    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--max_tokens", default=512, type=int)

    # <-- 修改点 4：🚨 调低并发数！云端 API 都有严格的 QPS 限制（每秒请求数）。
    # 如果你没有单独申请提额，开到 15 会立刻触发 "Rate Limit Exceeded" 报错。建议先从 2 甚至 1 开始测。
    parser.add_argument("--concurrency", default=1, type=int, help="百炼 API 并发控制，未提额前建议设为 2")

    parser.add_argument("--retry", default=3, type=int)
    parser.add_argument("--crop_bbox", action="store_true")
    parser.add_argument("--bbox_margin", default=0.25, type=float)
    parser.add_argument("--max_samples", default=10, type=int)
    parser.add_argument("--bert_name", default="bert-base-uncased", type=str)
    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_name)
    done_ids = load_done_ids_from_jsonl(args.output_jsonl)

    data = load_json_list(args.input_json)
    if args.max_samples and args.max_samples > 0:
        data = data[:args.max_samples]

    # 过滤掉已经处理过的数据
    tasks_to_process = [ann for ann in data if ann.get("id") not in done_ids]
    print(f"Total samples to process: {len(tasks_to_process)}")

    semaphore = asyncio.Semaphore(args.concurrency)
    
    # 增加 timeout 防止某些死请求一直挂起
    timeout = aiohttp.ClientTimeout(total=180) 
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            process_single_annotation(ann, args, tokenizer, session, semaphore)
            for ann in tasks_to_process
        ]
        
        # 使用 tqdm.asyncio.gather 显示进度条
        await tqdm.gather(*tasks, desc="Generating rationales (Async vLLM)")

if __name__ == "__main__":
    # 解决部分环境中 asyncio event loop 的报错问题
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(async_main())