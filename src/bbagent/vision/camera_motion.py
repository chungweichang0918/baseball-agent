"""鏡頭移動偵測：只看人物以外的背景角點，估計取樣幀之間的平移與縮放。

人在動、鏡頭沒動時，背景特徵點不會移動；手持晃動、轉向或縮放時，整片背景會一起移動。
這比讓 LLM 看幾張縮圖判斷可靠得多（實測 flash-lite 會漏判）。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .video import iter_frames

SAMPLE_INTERVAL_S = 0.2
WORK_WIDTH = 480            # 縮小後再找特徵點，速度快也比較不受雜訊影響
MIN_MATCHES = 12
BOX_PAD = 0.1               # 人物框外擴比例，避免把人的邊緣當成背景
MOVING_SHIFT = 0.03         # 每秒平移超過畫面對角線的 3%，視為鏡頭在動（經驗值）
MOVING_SCALE = 0.03         # 每秒縮放變化超過 3%（經驗值）
MIN_VALID_PAIRS = 3

BoxesAt = Callable[[int], list[np.ndarray]]


@dataclass
class CameraMotion:
    pairs: int
    valid_pairs: int
    shift_rate: list[float] = field(default_factory=list)   # 每對取樣幀：每秒平移（對角線比例）
    scale_rate: list[float] = field(default_factory=list)   # 每對取樣幀：每秒縮放變化
    moving: bool | None = None                               # None 表示背景特徵不足，無法判斷

    @property
    def p80_shift(self) -> float | None:
        return float(np.percentile(self.shift_rate, 80)) if self.shift_rate else None

    @property
    def p80_scale(self) -> float | None:
        return float(np.percentile(self.scale_rate, 80)) if self.scale_rate else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shift_rate"] = [round(x, 4) for x in self.shift_rate]
        d["scale_rate"] = [round(x, 4) for x in self.scale_rate]
        d["p80_shift"] = None if self.p80_shift is None else round(self.p80_shift, 4)
        d["p80_scale"] = None if self.p80_scale is None else round(self.p80_scale, 4)
        return d


def _background_mask(shape: tuple[int, int], boxes: list[np.ndarray], s: float) -> np.ndarray:
    mask = np.full(shape, 255, np.uint8)
    for x0, y0, x1, y1 in boxes:
        px, py = (x1 - x0) * BOX_PAD, (y1 - y0) * BOX_PAD
        mask[max(0, int((y0 - py) * s)):int((y1 + py) * s) + 1, max(0, int((x0 - px) * s)):int((x1 + px) * s) + 1] = 0
    return mask


def estimate_camera_motion(video: str | Path, fps: float, boxes_at: BoxesAt | None = None) -> CameraMotion:
    """在背景上找角點，用金字塔光流追蹤到下一個取樣幀，再以 RANSAC 估計相似變換（平移＋縮放＋旋轉）。"""
    step = max(1, round(SAMPLE_INTERVAL_S * fps))
    result = CameraMotion(pairs=0, valid_pairs=0)
    prev = None
    for idx, frame in iter_frames(video, stride=step):
        h, w = frame.shape[:2]
        s = WORK_WIDTH / w
        gray = cv2.cvtColor(cv2.resize(frame, (WORK_WIDTH, round(h * s)), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        mask = _background_mask(gray.shape, boxes_at(idx) if boxes_at else [], s)
        if prev is not None:
            result.pairs += 1
            p_idx, p_gray, p_mask = prev
            pts = cv2.goodFeaturesToTrack(p_gray, maxCorners=400, qualityLevel=0.01, minDistance=7, mask=p_mask)
            if pts is not None and len(pts) >= MIN_MATCHES:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(p_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
                ok = status.ravel() == 1
                src, dst = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
                # 追到人物框裡的點不算（可能是被人擋住或跟著人移動）
                inside = (dst[:, 0] >= 0) & (dst[:, 0] < gray.shape[1]) & (dst[:, 1] >= 0) & (dst[:, 1] < gray.shape[0])
                src, dst = src[inside], dst[inside]
                keep = mask[dst[:, 1].astype(int), dst[:, 0].astype(int)] > 0
                src, dst = src[keep], dst[keep]
                if len(src) >= MIN_MATCHES:
                    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if M is not None and inliers is not None and int(inliers.sum()) >= MIN_MATCHES:
                        dt = (idx - p_idx) / fps
                        diag = math.hypot(*gray.shape)
                        result.shift_rate.append(math.hypot(M[0, 2], M[1, 2]) / diag / dt)
                        result.scale_rate.append(abs(math.hypot(M[0, 0], M[1, 0]) - 1) / dt)
                        result.valid_pairs += 1
        prev = (idx, gray, mask)

    if result.valid_pairs >= MIN_VALID_PAIRS:
        result.moving = result.p80_shift > MOVING_SHIFT or result.p80_scale > MOVING_SCALE
    return result


def boxes_from_scan(scan) -> BoxesAt:
    """用粗掃結果中所有人的偵測框當遮罩；不在取樣幀上時取最近的取樣幀。"""
    frames = scan.frame_indices
    tracks = list(scan.tracks.values())

    def boxes_at(idx: int) -> list[np.ndarray]:
        i = int(np.clip(np.searchsorted(frames, idx), 0, len(frames) - 1))
        if i > 0 and abs(frames[i - 1] - idx) < abs(frames[i] - idx):
            i -= 1
        return [t.boxes[i] for t in tracks if t.present[i]]
    return boxes_at
