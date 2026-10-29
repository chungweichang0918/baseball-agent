"""gemini_inspect：讓 Gemini 看關鍵幀，回答一個具體的觀察重點。

YOLO 骨架沒有手指關鍵點，看不到握球、手套撐開這類細節，這些交給 Gemini 看圖判斷。
每個觀察都附信心等級與依據的圖片編號，報告引用時可以追溯。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from ..llm.gemini_client import GeminiClient, GeminiError, ImageInput, get_client, model_name
from ..llm.prompts import load_prompt
from ..schemas import ToolResult

MAX_IMAGES = 8
CONFIDENCE_ZH = {"low": "低", "medium": "中", "high": "高"}


class VisualObservation(BaseModel):
    aspect: str                                   # 觀察面向，例如「固定姿勢時的手套位置」
    finding: str
    confidence: Literal["low", "medium", "high"]
    images: list[int]                             # 依據的圖片編號（從 1 開始）


class InspectResult(BaseModel):
    answer: str
    observations: list[VisualObservation]
    limitations: list[str]


ImageLike = str | Path | ImageInput | np.ndarray


def _to_input(img: ImageLike) -> ImageInput:
    if isinstance(img, ImageInput):
        return img
    if isinstance(img, np.ndarray):
        return ImageInput.from_array(img)
    return ImageInput.from_path(img)


def _even_subset(n: int, k: int) -> list[int]:
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)}) if n > k else list(range(n))


def gemini_inspect(images: list[ImageLike], focus: str, captions: list[str] | None = None, context: str = "",
                   client: GeminiClient | None = None, model: str | None = None) -> ToolResult:
    if not images:
        return ToolResult(status="failed", summary="沒有可檢視的圖片", error="images 為空")
    captions = captions or [f"關鍵幀 {i}" for i in range(1, len(images) + 1)]
    if len(captions) != len(images):
        raise ValueError("captions 的數量必須與 images 相同")
    note = None
    if len(images) > MAX_IMAGES:
        keep = _even_subset(len(images), MAX_IMAGES)
        note = f"圖片共 {len(images)} 張，只送出均勻取樣的 {MAX_IMAGES} 張"
        images, captions = [images[i] for i in keep], [captions[i] for i in keep]

    client = client or get_client()
    model = model or model_name("pro")
    contents: list = [f"觀察重點：{focus}"]
    if context:
        contents.append(f"背景資訊：\n{context}")
    for i, (img, cap) in enumerate(zip(images, captions), start=1):
        contents += [f"圖 {i}：{cap}", _to_input(img)]
    try:
        result, resp = client.generate_structured(model, contents, InspectResult, system=load_prompt("inspect"))
    except GeminiError as e:
        return ToolResult(status="failed", summary="關鍵幀檢視失敗", error=str(e))

    n = len(images)
    for ob in result.observations:  # 模型偶爾會引用不存在的圖片編號，直接剔除
        ob.images = [i for i in ob.images if 1 <= i <= n]
    lines = [result.answer] + [
        f"- {ob.aspect}：{ob.finding}（信心{CONFIDENCE_ZH[ob.confidence]}，圖 {'、'.join(map(str, ob.images)) or '—'}）"
        for ob in result.observations
    ]
    lines += [f"限制：{x}" for x in result.limitations]
    if note:
        lines.append(f"※ {note}")
    return ToolResult(
        status="ok",
        summary="\n".join(lines),
        data=result.model_dump() | {
            "focus": focus,
            "captions": captions,
            "llm": {"model": resp.model, "cached": resp.cached, "latency_s": resp.latency_s, "usage": resp.usage},
        },
    )
