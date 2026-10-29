"""讀寫圖片檔。Windows 上 cv2.imread／imwrite 遇到中文路徑（例如「生成式AI作業」）會靜靜失敗，
所以改成先在記憶體中編碼、再用 Python 讀寫位元組。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path) -> np.ndarray | None:
    """讀不到或不是圖片時回傳 None，與 cv2.imread 相同。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path: str | Path, img: np.ndarray, params: list[int] | None = None) -> bool:
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img, params or [])
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True
