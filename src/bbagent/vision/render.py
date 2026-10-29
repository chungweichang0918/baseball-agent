"""在畫面上畫骨架、排成連拍圖。左側骨架畫藍色、右側畫橘色，方便用眼睛檢查左右是否標反。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
LEFT = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT = {2, 4, 6, 8, 10, 12, 14, 16}
COLOR_L = (235, 140, 40)    # BGR 藍
COLOR_R = (40, 140, 255)    # BGR 橘
COLOR_C = (200, 200, 200)
TAGGED = (9, 10, 13, 14, 15, 16)   # 手腕、膝蓋、腳踝旁標上 L / R


def _color(a: int, b: int | None = None) -> tuple[int, int, int]:
    pts = {a} if b is None else {a, b}
    if pts <= LEFT:
        return COLOR_L
    if pts <= RIGHT:
        return COLOR_R
    return COLOR_C


def draw_skeleton(img: np.ndarray, kpts: np.ndarray, kconf: np.ndarray, min_conf: float = 0.3,
                  thickness: int = 3) -> np.ndarray:
    ok = np.isfinite(kpts).all(axis=1) & (np.nan_to_num(kconf) >= min_conf)
    pt = lambda k: (int(round(kpts[k, 0])), int(round(kpts[k, 1])))  # noqa: E731
    for a, b in SKELETON:
        if ok[a] and ok[b]:
            cv2.line(img, pt(a), pt(b), _color(a, b), thickness, cv2.LINE_AA)
    for k in np.flatnonzero(ok):
        cv2.circle(img, pt(k), thickness + 2, _color(k), -1, cv2.LINE_AA)
    scale = max(0.5, img.shape[0] / 1400)
    for k in TAGGED:
        if ok[k]:
            x, y = pt(k)
            cv2.putText(img, "L" if k in LEFT else "R", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, _color(k), 2, cv2.LINE_AA)
    return img


def draw_box(img: np.ndarray, box: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    if np.isfinite(box).all():
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
    return img


def caption(img: np.ndarray, text: str) -> np.ndarray:
    scale = max(0.5, img.shape[1] / 700)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (0, 0), (w + 16, h + 16), (0, 0, 0), -1)
    cv2.putText(img, text, (8, h + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def contact_sheet(tiles: list[np.ndarray], cols: int = 6, tile_width: int = 320) -> np.ndarray:
    """把多張畫面縮成相同寬度後排成網格。"""
    if not tiles:
        raise ValueError("沒有可排列的畫面")
    resized = [cv2.resize(t, (tile_width, round(t.shape[0] * tile_width / t.shape[1])), interpolation=cv2.INTER_AREA)
               for t in tiles]
    tile_h = max(t.shape[0] for t in resized)
    rows = -(-len(resized) // cols)
    sheet = np.zeros((rows * tile_h, cols * tile_width, 3), np.uint8)
    for i, t in enumerate(resized):
        r, c = divmod(i, cols)
        sheet[r * tile_h:r * tile_h + t.shape[0], c * tile_width:(c + 1) * tile_width] = t
    return sheet


def read_frames(video: str | Path, indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    frames = {}
    try:
        for idx in sorted(set(indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames[idx] = frame
    finally:
        cap.release()
    return frames
