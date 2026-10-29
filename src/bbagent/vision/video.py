"""影片讀取與探測：基本資訊、場景切換。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

SCENE_CUT_THRESHOLD = 0.5   # 相鄰幀 HSV 直方圖的 Bhattacharyya 距離超過此值即視為切換（經驗值）
THUMB_WIDTH = 160           # 算直方圖前先縮小，速度快且不受雜訊影響


@dataclass
class VideoMeta:
    path: str
    fps: float
    frame_count: int            # 實際解碼到的幀數（容器中繼資料常不準）
    width: int
    height: int
    scene_cuts: list[int] = field(default_factory=list)  # 每個新場景的第一幀

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps

    def to_dict(self) -> dict:
        return asdict(self) | {"duration_s": round(self.duration_s, 3)}


def _open(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片：{path}")
    return cap


def _hist(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    thumb = cv2.resize(frame, (THUMB_WIDTH, max(1, round(h * THUMB_WIDTH / w))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def probe_video(path: str | Path, scene_threshold: float = SCENE_CUT_THRESHOLD) -> VideoMeta:
    """完整解碼一次影片，取得實際幀數並偵測場景切換。"""
    cap = _open(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cuts: list[int] = []
    prev = None
    n = 0
    width = height = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n == 0:  # 以實際畫面為準（手機直拍影片的旋轉資訊會影響中繼資料）
            height, width = frame.shape[:2]
        cur = _hist(frame)
        if prev is not None and cv2.compareHist(prev, cur, cv2.HISTCMP_BHATTACHARYYA) > scene_threshold:
            cuts.append(n)
        prev = cur
        n += 1
    cap.release()
    if n == 0:
        raise ValueError(f"影片無法解碼任何幀：{path}")
    return VideoMeta(str(path), fps, n, width, height, cuts)


def iter_frames(path: str | Path, stride: int = 1, start: int = 0,
                end: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """依 stride 取樣讀幀。跳過的幀只 grab 不解碼，比逐幀 read 快。"""
    if stride < 1:
        raise ValueError("stride 必須 ≥ 1")
    cap = _open(path)
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while end is None or idx < end:
            if (idx - start) % stride == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, frame
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()
