"""推論裝置選擇。

MPS 有部分運算子尚未支援，必須在 import torch 之前設定
PYTORCH_ENABLE_MPS_FALLBACK=1，讓這些運算子自動退回 CPU。
因此任何會用到 torch 的進入點都要先 import 本模組。
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def available_devices() -> list[str]:
    """依偏好順序列出可用裝置，cpu 永遠在最後。"""
    import torch

    devices = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    return devices


def resolve_device(preference: str = "auto") -> str:
    available = available_devices()
    if preference == "auto":
        return available[0]
    if preference not in available:
        raise RuntimeError(f"要求的裝置 {preference!r} 不可用；可用裝置：{available}")
    return preference
