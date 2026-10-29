"""產生合成的投手骨架資料，用來測試分段與指標，不需要 YOLO 與真實影片。

每一球是一個 5 秒的區塊：先回到固定姿勢、靜止、抬腿、跨步、揮臂出手、收尾。
各關鍵點的位移用 cosine 緩動，所以速度在動作中段最快、頭尾漸停，接近真實動作。
"""

from __future__ import annotations

import numpy as np

from bbagent.analysis.keypoints import KP
from bbagent.vision.pose_scan import N_KPT, PoseScan, PoseTrack

BLOCK_S = 5.0
# 區塊內各階段的時間（秒），也是測試的標準答案
TRUTH_S = {"set": 1.5, "first_move": 2.5, "peak_leg_lift": 3.1,
           "foot_strike": 3.5, "release": 3.6, "follow_through": 4.15}
BOX_PAD = 20

# 右投手、面向 +x，固定姿勢時雙手在胸前（像素座標，y 向下）
BASE = {
    "nose": (500, 300), "l_eye": (495, 295), "r_eye": (505, 295), "l_ear": (490, 300), "r_ear": (510, 300),
    "l_shoulder": (490, 380), "r_shoulder": (510, 380), "l_elbow": (475, 430), "r_elbow": (525, 430),
    "l_wrist": (485, 450), "r_wrist": (515, 450), "l_hip": (495, 520), "r_hip": (505, 520),
    "l_knee": (495, 610), "r_knee": (505, 610), "l_ankle": (495, 700), "r_ankle": (505, 700),
}
# 影片開始時雙手垂在身體兩側，第一球的「回到固定姿勢」會把它們移到胸前
INITIAL_OFFSET = {"l_wrist": (-25, 110), "r_wrist": (25, 110), "l_elbow": (-10, 40), "r_elbow": (10, 40)}

# (關鍵點, 開始, 結束, 位移)：時間相對於區塊起點
PITCH_MOVES = [
    ("l_knee", 2.5, 3.1, (0, -150)), ("l_ankle", 2.5, 3.1, (0, -120)),          # 抬腿
    ("l_knee", 3.1, 3.5, (200, 150)), ("l_ankle", 3.1, 3.5, (220, 120)),        # 跨步、著地
    ("l_hip", 3.1, 3.5, (40, 0)), ("r_hip", 3.1, 3.5, (40, 0)),
    ("r_wrist", 3.1, 3.5, (-140, -60)), ("r_elbow", 3.1, 3.5, (-70, -30)),      # 手臂後拉
    ("r_wrist", 3.5, 3.7, (330, -80)), ("r_elbow", 3.5, 3.7, (165, -40)),       # 揮臂，3.6 秒速度最快
    ("r_wrist", 3.7, 4.15, (-20, 180)), ("r_elbow", 3.7, 4.15, (0, 90)),        # 收尾
]
SETTLE_S = (0.0, 1.5)

LR_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def _ease(u: np.ndarray) -> np.ndarray:
    return (1 - np.cos(np.pi * np.clip(u, 0, 1))) / 2


def _animate(moves_per_block: list, block_s: float, n_blocks: int, initial_offset: dict, fps: float, stride: int,
             noise_px: float, seed: int, mirror: bool, track_id: int):
    """每個區塊先把上一區塊累積的位移歸零（回到起始姿勢），再依 moves 移動關鍵點。"""
    rng = np.random.default_rng(seed)
    frames = np.arange(0, round((n_blocks * block_s + 0.5) * fps), stride)
    t = frames / fps

    names = sorted(KP, key=KP.get)
    pos = np.tile(np.array([BASE[n] for n in names], float), (len(t), 1, 1))
    cum = np.zeros((N_KPT, 2))
    for name, d in initial_offset.items():
        pos[:, KP[name]] += d
        cum[KP[name]] += d

    for b in range(n_blocks):
        T = b * block_s
        moves = [(k, SETTLE_S[0], SETTLE_S[1], -cum[k].copy()) for k in range(N_KPT) if cum[k].any()]
        cum[:] = 0
        for name, t0, t1, d in moves_per_block:
            moves.append((KP[name], t0, t1, np.array(d, float)))
            cum[KP[name]] += d
        for k, t0, t1, d in moves:
            pos[:, k] += _ease((t - (T + t0)) / (t1 - t0))[:, None] * d

    pos += rng.normal(0, noise_px, pos.shape)
    if mirror:  # 鏡像成面向 −x，並交換左右
        pos[..., 0] = 1000 - pos[..., 0]
        pos = pos[:, LR_SWAP]

    boxes = np.concatenate([pos.min(axis=1) - BOX_PAD, pos.max(axis=1) + BOX_PAD], axis=1)
    n = len(t)
    track = PoseTrack(track_id, boxes, np.full(n, 0.9), pos, np.full((n, N_KPT), 0.9))
    return track, t, frames


def make_pitcher(n_pitches: int = 2, fps: float = 60.0, stride: int = 2, noise_px: float = 1.0,
                 seed: int = 0, left_handed: bool = False, track_id: int = 1):
    """回傳 (PoseTrack, 取樣時間, 取樣幀號, 每一球的標準答案幀號)。"""
    track, t, frames = _animate(PITCH_MOVES, BLOCK_S, n_pitches, INITIAL_OFFSET, fps, stride, noise_px, seed,
                                left_handed, track_id)
    truth = [{ph: round((b * BLOCK_S + s) * fps) for ph, s in TRUTH_S.items()} for b in range(n_pitches)]
    return track, t, frames, truth


# 打者：每次揮棒 4 秒。準備姿勢 → 引棒（雙手後拉、前膝微抬）→ 跨步 → 揮棒（2.9 秒手速最快）→ 收棒
SWING_BLOCK_S = 4.0
SWING_TRUTH_S = {"stance": 1.5, "load": 2.0, "foot_plant": 2.8, "contact": 2.9, "follow_through": 3.5}
SWING_MOVES = [
    ("r_wrist", 2.0, 2.45, (-40, -15)), ("l_wrist", 2.0, 2.45, (-40, -15)),      # 引棒
    ("r_elbow", 2.0, 2.45, (-20, -5)), ("l_elbow", 2.0, 2.45, (-20, -5)),
    ("l_knee", 2.0, 2.45, (10, -30)),
    ("l_ankle", 2.4, 2.8, (180, 0)), ("l_knee", 2.45, 2.8, (80, 30)),           # 跨步、著地
    ("r_wrist", 2.8, 3.0, (300, 40)), ("l_wrist", 2.8, 3.0, (300, 40)),         # 揮棒
    ("r_elbow", 2.8, 3.0, (150, 20)), ("l_elbow", 2.8, 3.0, (150, 20)),
    ("l_hip", 2.8, 3.0, (30, 0)), ("r_hip", 2.8, 3.0, (30, 0)),
    ("r_wrist", 3.0, 3.5, (-60, -120)), ("l_wrist", 3.0, 3.5, (-60, -120)),     # 收棒
]
SWING_INITIAL_OFFSET = {"l_wrist": (-25, 110), "r_wrist": (25, 110)}


def make_batter(n_swings: int = 2, fps: float = 60.0, stride: int = 2, noise_px: float = 1.0,
                seed: int = 0, left_handed: bool = False, track_id: int = 1):
    """回傳 (PoseTrack, 取樣時間, 取樣幀號, 每次揮棒的標準答案幀號)。"""
    track, t, frames = _animate(SWING_MOVES, SWING_BLOCK_S, n_swings, SWING_INITIAL_OFFSET, fps, stride, noise_px,
                                seed, left_handed, track_id)
    truth = [{ph: round((b * SWING_BLOCK_S + s) * fps) for ph, s in SWING_TRUTH_S.items()} for b in range(n_swings)]
    return track, t, frames, truth


def make_scan(tracks: list[PoseTrack], frames: np.ndarray, fps: float = 60.0) -> PoseScan:
    return PoseScan(frames, fps, (1000, 800), {t.track_id: t for t in tracks}, {"synthetic": True})
