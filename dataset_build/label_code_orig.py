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
#检查给定的图像边界框（bounding box）是否是一个合法的、有实际面积的坐标列表
def has_valid_box(box: Any) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return False
    return (x2 - x1) > 1e-6 and (y2 - y1) > 1e-6
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
#把坐标列表格式化成可读的字符串，塞给大模型看。
def format_box_xyxy(box: Any) -> str:
    if not has_valid_box(box):
        return "[]"
    x1, y1, x2, y2 = box
    return f"[xmin={x1}, ymin={y1}, xmax={x2}, ymax={y2}]"
#在使用 BERT 分词器（Tokenizer）对文本进行处理时，
#将“被标记为虚假/篡改的文本位置（索引）”转换成人类或大模型（LLM）容易阅读的上下文视图
@dataclass
class BertTokenView:
    tokens_with_idx: List[str]
    marked_context: str
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
# Prompting
# -----------------------------
def build_prompt(
    fake_cls: str,
    fake_modality: str,
    caption: str,
    fake_image_box: Any,
    fake_text_pos: Any,
    bert_view: Optional[BertTokenView],
) -> str:
    is_fake = (str(fake_cls).lower() != "orig")
    verdict = "FAKE" if is_fake else "REAL"

    box_str = format_box_xyxy(fake_image_box)
    pos_list = fake_text_pos if isinstance(fake_text_pos, list) else []

    bert_block = ""
    if bert_view is not None:
        bert_block = f"\n- Text Context for reference: \n{bert_view.marked_context}"

    if fake_modality == "text":
        modality_guidance = (
            "CRITICAL: The IMAGE is completely REAL. The manipulation is ONLY in the TEXT. "
            "DO NOT claim there are visual artifacts in the image. Instead, focus entirely on "
            "how the highlighted text tokens contradict the visual facts, exhibit unnatural sentiment, "
            "or introduce logical/entity mismatches with the image."
        )
    elif fake_modality == "image":
        modality_guidance = (
            "CRITICAL: The TEXT is completely REAL. The manipulation is ONLY in the IMAGE. "
            "DO NOT claim the text is illogical. Focus entirely on the visual artifacts "
            "(e.g., unnatural blending, lighting inconsistencies, texture degradation) specifically "
            "in or around the provided bounding box."
        )
    elif fake_modality == "both":
        modality_guidance = (
            "CRITICAL: BOTH the image AND the text are MANIPULATED. "
            "You must briefly point out the visual artifacts in the image AND the factual/semantic "
            "mismatch in the text."
        )
    else: 
        modality_guidance = (
            "CRITICAL: Both the image and the text are REAL. "
            "Explain how the text naturally, factually, and emotionally aligns with the unaltered visual evidence. "
            "State clearly that no manipulation is detected."
        )

    perspectives = [
        "Start directly with your observation, avoiding filler words.",
        "Focus on the relationship between the visual elements and the narrative of the text.",
        "Provide a concise, evidence-based assessment of the media's authenticity.",
    ]
    random_perspective = random.choice(perspectives)

    prompt = f"""You are an expert multimodal forensic analyst. Evaluate the provided image and caption.

Rules for your analysis:
1. Modality Constraint: {modality_guidance}
2. Be Concise & Natural: Write a short, free-flowing paragraph (around 40-80 words). DO NOT use numbered lists. Avoid repetitive templates.
3. Grounded Evidence: If bounding boxes or text token positions are provided, incorporate them naturally into your explanation to prove your point.
4. Tone: {random_perspective}

Inputs:
- Target Verdict: {verdict}
- Category: {fake_cls}
- Caption: "{caption}"
- fake_image_box: {box_str}
- fake_text_pos: {pos_list}{bert_block}

Output Format (Exactly 3 lines, no extra line breaks):
Verdict: {verdict}
Category: {fake_cls}
Evidence & Location: <Write your concise paragraph here based strictly on the Modality Constraint.>
"""
    return prompt

#将上面构建的文字 Prompt 和前面转好 Base64 的全图（如果有裁剪图，也会加上裁剪图）
# 组装成 OpenAI 标准的 [{"role": "system", ...}, {"role": "user", ...}] 消息体格式。
def build_messages_multimodal(
    prompt_text: str,
    full_img_data_url: str,
    crop_img_data_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    system_content = (
        "You are an expert multimodal forensic analyst. "
        "You must evaluate both the image and the text caption for inconsistencies or manipulations. "
        "Strictly follow the requested 3-line output format. "
        "Keep your explanations natural, highly diverse, and strictly grounded in the provided evidence. "
        "NEVER hallucinate artifacts."
    )
    system = {"role": "system", "content": system_content}

    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": full_img_data_url}},
    ]
    if crop_img_data_url is not None:
        user_content.append({"type": "image_url", "image_url": {"url": crop_img_data_url}})

    user = {"role": "user", "content": user_content}
    return [system, user]


# -----------------------------
# Async Local vLLM API call
# -----------------------------
#使用 aiohttp 异步发送 HTTP POST 请求到本地部署的 vLLM 服务器端点。
# 它负责把组装好的信息发过去，并解析返回的 JSON，提取出模型生成的文本内容。
async def async_local_vllm_chat_completion(
    session: aiohttp.ClientSession,
    api_base: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> str:
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
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

        prompt_text = build_prompt(
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
                out = await async_local_vllm_chat_completion(
                    session=session,
                    api_base=args.api_base,
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                if len(out) < 40: # 放宽对长度的限制，因为去掉了强制编号
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

    # Local vLLM API settings
    parser.add_argument("--api_base", default="http://192.168.33.5:8056/v1", type=str)
    parser.add_argument("--model", default="qwen3-vl", type=str)
    parser.add_argument("--temperature", default=0.7, type=float) # 默认调高以增加多样性
    parser.add_argument("--max_tokens", default=512, type=int)
    
    # 并发控制，A800 80G可以先从 15 试试水，太高可能会OOM或者排队超时
    parser.add_argument("--concurrency", default=15, type=int, help="Max concurrent requests to vLLM")
    parser.add_argument("--retry", default=3, type=int)
    parser.add_argument("--crop_bbox", action="store_true")
    parser.add_argument("--bbox_margin", default=0.25, type=float)
    parser.add_argument("--max_samples", default=0, type=int)
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