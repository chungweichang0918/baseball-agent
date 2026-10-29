"""從細掃的候選球中找出投球軌跡，估算球速與飛行時間。

YOLO 的 sports ball 在手機影片中誤報很多（手、手套、燈、地上的球），所以分三步：
  1. 過濾：去掉離投手手腕太近的候選（球還在手上，或其實是手）、在同一位置重複出現的候選（靜止物）
  2. RANSAC 擬合：x 隨時間等速、y 為二次曲線（重力加上透視），找出最多幀一致的軌跡
  3. 檢查：軌跡至少要有 MIN_POINTS 幀、持續 MIN_SPAN_S 秒、速度夠快才採用

球速只能粗估：單一鏡頭沒有校正，用投手身高（假設 ASSUMED_HEIGHT_M）把像素換成公尺，
而且只有側拍、鏡頭固定時，球的移動才大致與投手在同一個深度平面上。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

HAND_RADIUS = 0.12          # 離任一手腕在這個距離內（身高）的候選視為還在手上
STATIC_RADIUS_REL = 0.01    # 在這個距離內（畫面對角線比例）重複出現 …
STATIC_MIN_FRAMES = 3       # … 這麼多幀以上，視為靜止物
RANSAC_ITERS = 400
TOL_REL = 0.012             # 擬合容許誤差（畫面高度比例），至少 3 像素
MIN_POINTS = 5
MIN_SPAN_S = 0.06
MIN_SPEED_BODY = 5.0        # 身高/秒；球速 80 公里的球約 12 身高/秒，手套、手臂通常低於此值
ASSUMED_HEIGHT_M = 1.80
RELEASE_TO_PLATE_M = 16.8   # 投手板到本壘 18.44 公尺，扣掉約 1.6 公尺的出手延伸
PLAUSIBLE_KMH = (50, 170)


@dataclass
class TrajectoryFit:
    inliers: np.ndarray         # 候選的索引（每幀最多一個）
    t_ref: float
    coef_x: np.ndarray          # x = a + b·(t − t_ref)
    coef_y: np.ndarray          # y = c + d·(t − t_ref) + e·(t − t_ref)²
    residual_px: float

    def at(self, t: np.ndarray) -> np.ndarray:
        u = np.asarray(t, float) - self.t_ref
        return np.stack([self.coef_x[0] + self.coef_x[1] * u,
                         self.coef_y[0] + self.coef_y[1] * u + self.coef_y[2] * u ** 2], axis=-1)

    def speed_px(self, t: float) -> float:
        u = t - self.t_ref
        return float(np.hypot(self.coef_x[1], self.coef_y[1] + 2 * self.coef_y[2] * u))


@dataclass
class PitchBall:
    pitch: int
    found: bool
    points: int = 0
    start_s: float | None = None
    end_s: float | None = None
    visible_s: float | None = None
    start_after_release_s: float | None = None
    speed_body_s: float | None = None
    speed_kmh_est: float | None = None
    flight_time_est_s: float | None = None
    time_to_plate_est_s: float | None = None
    residual_px: float | None = None
    path: list[list[float]] = field(default_factory=list)     # 擬合後的 (t, x, y)，畫圖用
    removed: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def filter_candidates(xy: np.ndarray, wrists: np.ndarray, body_h: float,
                      frame_size: tuple[int, int]) -> tuple[np.ndarray, dict[str, int]]:
    """wrists 為 (N, 2, 2)：每個候選同一時刻兩個手腕的位置（可含 NaN）。回傳 (保留的遮罩, 各原因刪除數)。"""
    n = len(xy)
    keep = np.ones(n, bool)
    removed = {"near_hand": 0, "static": 0}
    if n == 0:
        return keep, removed
    d = np.linalg.norm(xy[:, None, :] - wrists, axis=2)               # (N, 2)
    near = np.nan_to_num(np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1), nan=np.inf) < HAND_RADIUS * body_h
    keep &= ~near
    removed["near_hand"] = int(near.sum())

    diag = float(np.hypot(*frame_size))
    pair = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2) < STATIC_RADIUS_REL * diag
    static = pair.sum(axis=1) >= STATIC_MIN_FRAMES
    removed["static"] = int((static & keep).sum())
    keep &= ~static
    return keep, removed


def _fit(t: np.ndarray, xy: np.ndarray, t_ref: float) -> tuple[np.ndarray, np.ndarray]:
    u = t - t_ref
    cx = np.polyfit(u, xy[:, 0], 1)[::-1]
    cy = np.polyfit(u, xy[:, 1], 2)[::-1] if len(u) >= 3 else np.r_[np.polyfit(u, xy[:, 1], 1)[::-1], 0.0]
    return cx, cy


def _one_per_frame(idx: np.ndarray, t: np.ndarray, res: np.ndarray) -> np.ndarray:
    best: dict[float, int] = {}
    for i in idx:
        if t[i] not in best or res[i] < res[best[t[i]]]:
            best[t[i]] = i
    return np.array(sorted(best.values(), key=lambda i: t[i]), int)


def fit_trajectory(t: np.ndarray, xy: np.ndarray, frame_h: int, seed: int = 0) -> TrajectoryFit | None:
    """RANSAC：每次取 3 個不同時間的候選擬合，保留一致幀數最多的軌跡，再用全部一致點重新擬合。"""
    if len(np.unique(t)) < MIN_POINTS:
        return None
    rng = np.random.default_rng(seed)
    tol = max(3.0, TOL_REL * frame_h)
    t_ref = float(t.min())
    best, best_score = None, 0
    for _ in range(RANSAC_ITERS):
        pick = rng.choice(len(t), 3, replace=False)
        if len(np.unique(t[pick])) < 3:
            continue
        cx, cy = _fit(t[pick], xy[pick], t_ref)
        fit = TrajectoryFit(pick, t_ref, cx, cy, 0.0)
        res = np.linalg.norm(fit.at(t) - xy, axis=1)
        inl = _one_per_frame(np.flatnonzero(res < tol), t, res)
        if len(inl) > best_score:
            best, best_score = inl, len(inl)
    if best is None or best_score < MIN_POINTS:
        return None
    for _ in range(2):   # 用一致點重新擬合，再更新一致點
        cx, cy = _fit(t[best], xy[best], t_ref)
        fit = TrajectoryFit(best, t_ref, cx, cy, 0.0)
        res = np.linalg.norm(fit.at(t) - xy, axis=1)
        best = _one_per_frame(np.flatnonzero(res < tol), t, res)
        if len(best) < MIN_POINTS:
            return None
    fit.inliers = best
    fit.residual_px = round(float(np.sqrt(np.mean(res[best] ** 2))), 2)
    return fit


def analyze_pitch_ball(pitch: int, t: np.ndarray, xy: np.ndarray, wrists: np.ndarray, body_h: float,
                       frame_size: tuple[int, int], release_s: float, delivery_time_s: float | None,
                       speed_ok: bool) -> PitchBall:
    """speed_ok：側拍且鏡頭固定時才換算球速與飛行時間。"""
    out = PitchBall(pitch, False)
    keep, out.removed = filter_candidates(xy, wrists, body_h, frame_size)
    t, xy = t[keep], xy[keep]
    fit = fit_trajectory(t, xy, frame_size[1])
    if fit is None:
        out.notes.append(f"找不到一致的球軌跡（過濾後剩 {len(t)} 個候選、{len(np.unique(t))} 幀）")
        return out
    ti = t[fit.inliers]
    span = float(ti[-1] - ti[0])
    speed_body = fit.speed_px(float(ti[0])) / body_h
    if span < MIN_SPAN_S or speed_body < MIN_SPEED_BODY:
        out.notes.append(f"候選軌跡太短或太慢（{span:.2f} 秒、{speed_body:.1f} 身高/秒），不像投出的球")
        return out
    out.found = True
    out.points = len(ti)
    out.start_s, out.end_s, out.visible_s = round(float(ti[0]), 3), round(float(ti[-1]), 3), round(span, 3)
    out.start_after_release_s = round(float(ti[0]) - release_s, 3)
    out.speed_body_s = round(speed_body, 1)
    out.residual_px = fit.residual_px
    out.path = [[round(float(x), 3) for x in (tt, *p)] for tt, p in zip(ti, fit.at(ti))]
    if out.start_after_release_s > 0.15:
        out.notes.append("軌跡比出手晚 0.15 秒以上才出現，可能抓到的是其他移動物體")
    if not speed_ok:
        out.notes.append("非側拍或鏡頭移動：只回報軌跡，不換算球速")
        return out
    mps = speed_body * ASSUMED_HEIGHT_M
    out.speed_kmh_est = round(mps * 3.6, 1)
    if not PLAUSIBLE_KMH[0] <= out.speed_kmh_est <= PLAUSIBLE_KMH[1]:
        out.notes.append(f"粗估球速 {out.speed_kmh_est} 公里不合理，不採用")
        out.speed_kmh_est = None
        return out
    out.flight_time_est_s = round(RELEASE_TO_PLATE_M / mps, 3)
    if delivery_time_s is not None:
        out.time_to_plate_est_s = round(delivery_time_s + out.flight_time_est_s, 3)
    return out
