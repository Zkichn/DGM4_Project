import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

#在使用 BERT 分词器（Tokenizer）对文本进行处理时，
#将“被标记为虚假/篡改的文本位置（索引）”转换成人类或大模型（LLM）容易阅读的上下文视图
@dataclass
class BertTokenView:
    tokens_with_idx: List[str]
    marked_context: str

def has_valid_box(box: Any) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return False
    return (x2 - x1) > 1e-6 and (y2 - y1) > 1e-6

def format_box_xyxy(box: Any) -> str:
    if not has_valid_box(box):
        return "[]"
    x1, y1, x2, y2 = box
    return f"[xmin={x1}, ymin={y1}, xmax={x2}, ymax={y2}]"


# ==========================================
# Prompt 版本库
# ==========================================
def build_prompt_v1(
    fake_cls: str,
    fake_modality: str,
    caption: str,
    fake_image_box: Any,
    fake_text_pos: Any,
    bert_view: Optional[BertTokenView],
) -> str:
    """
    V1 版本：生成质量还可以，但是同质化比较严重
    """
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

def build_prompt_v2(
    fake_cls: str,
    fake_modality: str,
    caption: str,
    fake_image_box: Any,
    fake_text_pos: Any,
    bert_view: Optional[BertTokenView],
) -> str:
    """
    V2 版本：严格限制幻觉，强制输出 IOU 坐标，禁止 orig 捏造坐标。
    """
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
            "DO NOT claim the text is illogical. Focus entirely on the specific visual anomalies "
            "in or strictly around the provided bounding box."
        )
    elif fake_modality == "both":
        modality_guidance = (
            "CRITICAL: BOTH the image AND the text are MANIPULATED. "
            "You must point out one visual artifact in the image box AND the factual mismatch in the text."
        )
    else:
        modality_guidance = (
            "CRITICAL: Both the image and the text are REAL. "
            "Explain how the text naturally aligns with the unaltered visual evidence. "
            "State clearly that no manipulation is detected."
        )

    perspectives = [
        "Focus on color temperature, illumination direction, and ambient shadows.",
        "Focus on image grain, noise distribution, and texture sharpness.",
        "Focus on boundary blending, edge pixelation, or halo effects.",
        "Start directly with your observation using a blunt, clinical tone, avoiding filler words."
    ]
    random_perspective = random.choice(perspectives)

    prompt = f"""You are an expert multimodal forensic analyst. Evaluate the provided image and caption.

Rules for your analysis:
1. Modality Constraint: {modality_guidance}
2. Be Concise & Diverse: Write a short, free-flowing paragraph (around 40-80 words). DO NOT use numbered lists.
3. Grounding for IOU Extraction (CRITICAL): 
   - IF `fake_image_box` is NOT `[]`: You MUST explicitly quote the exact coordinates in your text.
   - IF `fake_image_box` is `[]` (e.g., in 'orig' category): You MUST NOT invent, hallucinate, or mention any coordinates.
   - Apply the same rule for `fake_text_pos` if provided.
4. Anti-Homogenization & Tone: {random_perspective} Diversify your vocabulary. You may use terms like "unnatural blending" or "lighting inconsistencies" if they are the most accurate, but avoid overusing them as generic templates. Prioritize highly specific and varied descriptive language.
Inputs:
- Target Verdict: {verdict}
- Category: {fake_cls}
- Caption: "{caption}"
- fake_image_box: {box_str}
- fake_text_pos: {pos_list}{bert_block}

Output Format (Exactly 3 lines, no extra line breaks):
Verdict: {verdict}
Category: {fake_cls}
Evidence & Location: <Write your concise paragraph here strictly following the rules>
"""
    return prompt

# 将组装 Message 的函数也放过来
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
        "NEVER hallucinate artifacts or bounding boxes."
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


# ... [上面是你贴给我的那段代码，保持原样] ...

if __name__ == "__main__":
    # ==========================================
    # 本地测试台 (本地运行此文件即可查看生成的 Prompt)
    # ==========================================

    # 构造一个虚拟的 BertTokenView 用于测试
    mock_bert_view = BertTokenView(
        tokens_with_idx=["0:[CLS]", "1:The", "2:man", "3:is", "4:running", "5:[SEP]"],
        marked_context="- around pos 2: 0:[CLS] 1:The 2:man 3:is 4:running"
    )

    print("=" * 60)
    print("▶ 测试用例 1: FAKE (造假图片 - face_swap)")
    print("=" * 60)
    prompt_fake = build_prompt_v1(
        fake_cls="face_swap",
        fake_modality="image",
        caption="What s the biggest surprise from the JPMorgan Senate report",
        fake_image_box=[159.0, 18.0, 208.0, 79.0],  # 有坐标框
        fake_text_pos=[],
        bert_view=None  # 因为是 image 造假，文本一般没有上下文
    )
    print(prompt_fake)
    print("\n\n")

    print("=" * 60)
    print("▶ 测试用例 2: REAL (真实原图 - orig)")
    print("=" * 60)
    prompt_real = build_prompt_v2(
        fake_cls="orig",
        fake_modality="none",
        caption="Tony Nicklinson with his wife Jane Nicklinson.",
        fake_image_box=[],  # orig 类别没有坐标框
        fake_text_pos=[],
        bert_view=None
    )
    print(prompt_real)