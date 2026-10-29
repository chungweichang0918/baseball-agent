"""把單一主體的原始骨架轉成可分析的時間序列：補短缺口、平滑、算速度與動作能量。

速度一律以「身高/秒」為單位（身高 = 主體偵測框高度的中位數），
因此門檻值不受影片解析度與拍攝距離影響。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from ..vision.pose_scan import PoseTrack

KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8, "l_wrist": 9, "r_wrist": 10,
    "l_hip": 11, "r_hip": 12, "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}
ENERGY_KPTS = [KP[k] for k in (
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
)]

# 側拍時身體一側被另一側遮住，YOLO 常把左右標反。手、腳各自檢查
LR_GROUPS = {
    "arms": ([KP["l_shoulder"], KP["l_elbow"], KP["l_wrist"]], [KP["r_shoulder"], KP["r_elbow"], KP["r_wrist"]]),
    "legs": ([KP["l_hip"], KP["l_knee"], KP["l_ankle"]], [KP["r_hip"], KP["r_knee"], KP["r_ankle"]]),
}
SWAP_RATIO = 0.5          # 互換後與預測位置的距離小於原本的這個比例，才判定為標反

MIN_KPT_CONF = 0.3        # 信心低於此值的關鍵點視為缺漏
MAX_INTERP_GAP_S = 0.3    # 只補這麼短的缺口，更長的保留 NaN
SMOOTH_WINDOW_S = 0.2     # 位置平滑視窗
ENERGY_WINDOW_S = 0.1     # 動作能量再平滑一次，壓低雜訊


def side_kp(side: str, part: str) -> int:
    """side 為 "R" 或 "L"，part 例如 "wrist"。"""
    return KP[f"{side.lower()}_{part}"]


def other_side(side: str) -> str:
    return "L" if side == "R" else "R"


def _odd_window(seconds: float, fps: float, minimum: int = 5) -> int:
    w = max(minimum, round(seconds * fps))
    return w if w % 2 else w + 1


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """回傳 mask 中連續 True 區段的 [start, end)。"""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def interp_short_gaps(y: np.ndarray, t: np.ndarray, max_gap_s: float) -> np.ndarray:
    out = y.copy()
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return out
    for s, e in _runs(~finite):
        if s == 0 or e == len(y) or t[e] - t[s - 1] > max_gap_s:
            continue
        out[s:e] = np.interp(t[s:e], [t[s - 1], t[e]], [y[s - 1], y[e]])
    return out


def fix_lr_swaps(kpts: np.ndarray) -> tuple[np.ndarray, int]:
    """修正左右標反，讓同一條手臂、同一條腿在整段時間中保持同一個標記。

    逐幀比較「維持原標記」與「左右互換」哪一個更接近預測位置（前兩個有效幀的等速度外插），
    互換明顯更接近就交換。只保證前後一致，不保證「左」真的是解剖學上的左，
    所以分段時仍使用不分左右的訊號。回傳 (修正後座標, 有交換的幀數)。
    """
    out = kpts.copy()
    swapped = np.zeros(len(kpts), dtype=bool)
    for left, right in LR_GROUPS.values():
        idx, flipped = left + right, right + left
        hist: list[tuple[int, np.ndarray]] = []   # 最近兩個有效幀 (索引, 座標)
        for i in range(len(out)):
            cur = out[i, idx]
            if np.isnan(cur).any():
                continue
            if hist:
                j1, p1 = hist[-1]
                pred = p1
                if len(hist) == 2 and hist[0][0] == j1 - 1 and j1 == i - 1:
                    pred = 2 * p1 - hist[0][1]
                keep = np.linalg.norm(cur - pred, axis=1).sum()
                alt = np.linalg.norm(out[i, flipped] - pred, axis=1).sum()
                if alt < SWAP_RATIO * keep:
                    out[i, idx] = out[i, flipped]
                    swapped[i] = True
            hist = (hist + [(i, out[i, idx].copy())])[-2:]
    return out, int(swapped.sum())


def smooth(y: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """對每段連續的有效值分別做 Savitzky-Golay 平滑；太短的段落保持原值。"""
    out = y.copy()
    for s, e in _runs(np.isfinite(y)):
        if e - s >= window:
            out[s:e] = savgol_filter(y[s:e], window, polyorder)
    return out


@dataclass
class SubjectSeries:
    track_id: int
    times: np.ndarray           # (N,) 秒
    frame_indices: np.ndarray   # (N,) 原始幀號
    pos: np.ndarray             # (N, 17, 2) 平滑後的像素座標
    speed: np.ndarray           # (N, 17) 身高/秒
    vel: np.ndarray             # (N, 17, 2) 身高/秒
    energy: np.ndarray          # (N,) 主要關鍵點的平均速度
    body_h: float               # 像素
    lr_fixes: int = 0           # 修正了幾幀左右標反

    @property
    def fps(self) -> float:
        return float(1 / np.median(np.diff(self.times)))

    def seconds_to_samples(self, seconds: float) -> int:
        return max(1, round(seconds * self.fps))

    def hip_center(self) -> np.ndarray:
        return (self.pos[:, KP["l_hip"]] + self.pos[:, KP["r_hip"]]) / 2

    def knee_height(self, side: str) -> np.ndarray:
        """膝蓋高於髖部中心多少個身高（影像 y 軸向下，所以用 髖 − 膝）。"""
        return (self.hip_center()[:, 1] - self.pos[:, side_kp(side, "knee"), 1]) / self.body_h

    def max_speed(self, part: str) -> np.ndarray:
        """左右兩側中較快的那一個，不受左右標記影響。"""
        return np.fmax(self.speed[:, side_kp("L", part)], self.speed[:, side_kp("R", part)])

    def higher_knee(self) -> np.ndarray:
        return np.fmax(self.knee_height("L"), self.knee_height("R"))


def build_series(track: PoseTrack, times: np.ndarray, frame_indices: np.ndarray) -> SubjectSeries:
    raw = track.kpts.astype(float).copy()
    raw[~(track.kconf >= MIN_KPT_CONF)] = np.nan  # 也涵蓋 kconf 為 NaN（不在畫面中）的幀
    raw, lr_fixes = fix_lr_swaps(raw)
    body_h = float(np.nanmedian(track.heights))
    if not np.isfinite(body_h) or body_h <= 0:
        raise ValueError(f"追蹤對象 {track.track_id} 沒有有效的偵測框")

    fps = 1 / np.median(np.diff(times))
    window = _odd_window(SMOOTH_WINDOW_S, fps)
    pos = np.full_like(raw, np.nan)
    for k in range(raw.shape[1]):
        for d in range(2):
            pos[:, k, d] = smooth(interp_short_gaps(raw[:, k, d], times, MAX_INTERP_GAP_S), window)

    vel = np.gradient(pos, times, axis=0) / body_h
    speed = np.linalg.norm(vel, axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 整列都是 NaN 時會警告，結果本來就該是 NaN
        energy = np.nanmean(speed[:, ENERGY_KPTS], axis=1)
    finite = np.isfinite(energy)
    if finite.any():
        filled = np.where(finite, energy, 0.0)
        smoothed = uniform_filter1d(filled, _odd_window(ENERGY_WINDOW_S, fps, minimum=3))
        energy = np.where(finite, smoothed, np.nan)
    return SubjectSeries(track.track_id, times, frame_indices, pos, speed, vel, energy, body_h, lr_fixes)
