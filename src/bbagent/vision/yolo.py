"""YOLO 模型載入。

權重放在 config.models.dir。檔案不存在時，Ultralytics 會把官方權重
自動下載到指定路徑。
"""

from __future__ import annotations

from . import device as _device  # noqa: F401  確保 MPS fallback 在 import torch 前設定
from ..config import model_path


def load_yolo(kind: str):
    """kind 為 "detector" 或 "pose"。"""
    from ultralytics import YOLO

    path = model_path(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    return YOLO(str(path))
