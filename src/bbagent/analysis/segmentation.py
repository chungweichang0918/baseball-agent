"""投手動作分段：從骨架時間序列找出每一球的 6 個階段。

順序是先找最明確的「出手」，再往前推抬腿、著地、動作啟動與固定姿勢，往後找收尾：
  ⑤ 出手        兩手腕中較快者的速度峰值
  ③ 抬腿最高點   出手前（至少 min_lift_to_release_s），兩膝中較高者最高的時刻
  ④ 前腳著地     抬腿後兩腳踝中較快者衝到最快，之後速度掉到峰值一定比例以下的第一個時刻
  ② 動作啟動     抬腿期動作能量往回推，降到「動作門檻」以下的下一個時刻
  ① 固定姿勢     動作啟動之前，能量持續低於「靜止門檻」的那段時間的起點
  ⑥ 跟進收尾     出手後動作能量降回「收尾門檻」以下的第一個時刻

側拍時 YOLO 常把左右標反，所以分段只用「兩側取較大值」這類不分左右的訊號；
投球手以使用者指定為準，沒有指定時才依出手瞬間哪個手腕較快來判斷。
門檻都相對於該球自己的靜止基準與峰值，不寫死絕對數值。
預設值是用合成資料調出來的，拿到標註過的真實影片後要再校正。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .keypoints import SubjectSeries, side_kp

PITCH_PHASES = ["set", "first_move", "peak_leg_lift", "foot_strike", "release", "follow_through"]
PHASE_ZH = {"set": "固定姿勢", "first_move": "動作啟動", "peak_leg_lift": "抬腿最高點",
            "foot_strike": "前腳著地", "release": "出手", "follow_through": "跟進收尾"}


@dataclass
class PitchSegParams:
    min_pitch_interval_s: float = 2.0   # 兩球出手至少相隔幾秒
    release_peak_rel: float = 0.5       # 出手峰值至少要達到全片最大手腕速度的此比例
    min_release_speed: float = 2.0      # 出手峰值的絕對下限（身高/秒），排除靜止時雜訊造成的小峰值
    lift_window_s: float = 2.0          # 出手前多久內找抬腿最高點
    min_lift_to_release_s: float = 0.15 # 抬腿最高點至少比出手早這麼久（排除出手瞬間後腳離地）
    lift_energy_window_s: float = 0.6   # 抬腿最高點前多久內找抬腿期的能量峰值
    rest_window_s: float = 3.0          # 抬腿前多久內找靜止基準
    rest_percentile: float = 10.0       # 靜止基準取該區間能量的此百分位數
    move_rel: float = 0.2               # 動作門檻 = 靜止 + 此比例 ×（抬腿期峰值 − 靜止）
    set_rel: float = 0.1                # 靜止門檻（判斷固定姿勢）
    min_set_s: float = 0.3              # 靜止至少要持續這麼久才算固定姿勢
    set_spike_s: float = 0.15           # 靜止期中容許的短暫雜訊突波長度（骨架抖動造成）
    strike_rel: float = 0.2             # 前腳速度降到跨步峰值的此比例以下即視為著地
    follow_window_s: float = 2.5        # 出手後多久內找收尾
    follow_rel: float = 0.05            # 收尾門檻 = 靜止 + 此比例 ×（出手期峰值 − 靜止）


@dataclass
class PitchCycle:
    index: int                                  # 第幾球（從 1 開始）
    phases: dict[str, int | None]               # 原始幀號
    times: dict[str, float | None]              # 秒
    samples: dict[str, int | None] = field(repr=False, default_factory=dict)  # SubjectSeries 中的索引
    release_side: str | None = None             # 出手瞬間較快的手腕在骨架上的標記（"R" 或 "L"）
    warnings: list[str] = field(default_factory=list)


@dataclass
class PitchSegmentation:
    throwing_side: str                          # "R" 或 "L"
    side_source: str                            # "user"（使用者指定）或 "skeleton"（依骨架判斷）
    cycles: list[PitchCycle]
    params: PitchSegParams
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for c in d["cycles"]:
            c.pop("samples")
        return d


def detect_throwing_side(series: SubjectSeries) -> str:
    r = np.nanmax(series.speed[:, side_kp("R", "wrist")])
    l = np.nanmax(series.speed[:, side_kp("L", "wrist")])
    return "R" if np.nan_to_num(r) >= np.nan_to_num(l) else "L"


def _nanargmax(x: np.ndarray) -> int | None:
    return int(np.nanargmax(x)) if np.isfinite(x).any() else None


@dataclass
class MotionStart:
    rest: float                 # 靜止基準（動作能量）
    move: int | None            # 動作開始的樣本索引
    still: int | None           # 動作開始前靜止期的起點
    problems: list[str]         # early／short／no_rest／no_data


START_WARNINGS = {
    "early": "固定姿勢可能比搜尋範圍更早開始",
    "short": "動作啟動前沒有足夠長的靜止期（可能不是固定姿勢投球）",
    "no_rest": "搜尋範圍內找不到靜止期，無法判定動作啟動",
    "no_data": "動作能量資料不足，無法判定動作啟動與固定姿勢",
}


def find_motion_start(series: SubjectSeries, anchor: int, boundary: int, rest_window_s: float,
                      peak_window_s: float, rest_percentile: float, move_rel: float, set_rel: float,
                      spike_s: float, min_still_s: float) -> MotionStart:
    """從 anchor（例如抬腿最高點）往回找：動作能量降到「動作門檻」以下的下一個時刻是動作開始，
    再往回找能量持續低於「靜止門檻」的靜止期起點。投手與打者共用。"""
    win = series.seconds_to_samples
    energy = series.energy
    rest_lo = max(boundary, anchor - win(rest_window_s))
    window = energy[rest_lo:anchor + 1]
    rest = float(np.nanpercentile(window, rest_percentile)) if np.isfinite(window).any() else np.nan
    peak_lo = max(rest_lo, anchor - win(peak_window_s))
    peak_rel = _nanargmax(energy[peak_lo:anchor + 1])
    if not np.isfinite(rest) or peak_rel is None:
        return MotionStart(rest, None, None, ["no_data"])
    peak_i = peak_lo + peak_rel
    span = energy[peak_i] - rest
    move_thr, still_thr = rest + move_rel * span, rest + set_rel * span
    j = peak_i
    while j > rest_lo and energy[j - 1] >= move_thr:
        j -= 1
    if j <= rest_lo:
        return MotionStart(rest, None, None, ["no_rest"])
    # 往回找靜止期起點；短於 spike_s 的突波視為雜訊，不中斷靜止期
    q, spike, max_spike = j, 0, win(spike_s)
    for i in range(j - 1, rest_lo - 1, -1):
        if energy[i] < still_thr:
            q, spike = i, 0
        else:
            spike += 1
            if spike > max_spike:
                break
    if series.times[j] - series.times[q] < min_still_s:
        return MotionStart(rest, j, None, ["short"])
    return MotionStart(rest, j, q, ["early"] if q == rest_lo and rest_lo > 0 else [])


def find_settle(energy: np.ndarray, peak: int, hi: int, rest: float, rel: float, peak_window: int) -> int | None:
    """peak 之後動作能量降回「收尾門檻」以下的第一個時刻（搜尋到 hi 為止）。"""
    if hi <= peak:
        return None
    peak_e = np.nanmax(energy[peak:min(hi, peak + peak_window) + 1])
    if not (np.isfinite(rest) and np.isfinite(peak_e)):
        return None
    below = np.flatnonzero(energy[peak + 1:hi + 1] < rest + rel * (peak_e - rest))
    return peak + 1 + int(below[0]) if len(below) else None


def segment_pitcher(series: SubjectSeries, params: PitchSegParams | None = None,
                    throwing_side: str | None = None) -> PitchSegmentation:
    p = params or PitchSegParams()
    n = len(series.times)
    win = series.seconds_to_samples

    wrist_speed = np.nan_to_num(series.max_speed("wrist"))
    ankle_speed = series.max_speed("ankle")
    knee_h = series.higher_knee()
    energy = series.energy

    seg = PitchSegmentation(throwing_side or "R", "user" if throwing_side else "skeleton", [], p)
    if wrist_speed.max() <= 0:
        seg.warnings.append("投球手腕沒有任何有效的速度資料")
        return seg
    height = max(p.release_peak_rel * wrist_speed.max(), p.min_release_speed)
    releases, _ = find_peaks(wrist_speed, height=height, distance=win(p.min_pitch_interval_s))
    if len(releases) == 0:
        seg.warnings.append("找不到出手動作")

    boundary = 0  # 上一球結束的位置，下一球的搜尋不能越過它
    for k, r in enumerate(releases):
        r = int(r)
        warns: list[str] = []
        s: dict[str, int | None] = dict.fromkeys(PITCH_PHASES)
        s["release"] = r

        # ③ 抬腿最高點
        lo = max(boundary, r - win(p.lift_window_s))
        hi = r - win(p.min_lift_to_release_s)
        lift_rel = _nanargmax(knee_h[lo:hi + 1]) if hi >= lo else None
        lift = lo + lift_rel if lift_rel is not None else None
        s["peak_leg_lift"] = lift
        if lift is None:
            warns.append("膝蓋資料缺漏，無法判定抬腿最高點")

        # ④ 前腳著地
        if lift is not None:
            m_rel = _nanargmax(ankle_speed[lift:r + 1])
            if m_rel is not None:
                m = lift + m_rel
                below = np.flatnonzero(ankle_speed[m:r + 1] < p.strike_rel * ankle_speed[m])
                s["foot_strike"] = m + int(below[0]) if len(below) else None
            if s["foot_strike"] is None:
                warns.append("無法判定前腳著地")

        # ② 動作啟動、① 固定姿勢
        anchor = lift if lift is not None else r
        start = find_motion_start(series, anchor, boundary, p.rest_window_s, p.lift_energy_window_s,
                                  p.rest_percentile, p.move_rel, p.set_rel, p.set_spike_s, p.min_set_s)
        rest = start.rest
        s["first_move"], s["set"] = start.move, start.still
        warns += [START_WARNINGS[w] for w in start.problems]

        # ⑥ 跟進收尾
        next_r = int(releases[k + 1]) if k + 1 < len(releases) else n
        s["follow_through"] = find_settle(energy, r, min(n - 1, r + win(p.follow_window_s), next_r - 1),
                                          rest, p.follow_rel, win(0.3))
        if s["follow_through"] is None:
            warns.append("出手後找不到動作停下的時刻")

        found = [(ph, s[ph]) for ph in PITCH_PHASES if s[ph] is not None]
        for (p1, i1), (p2, i2) in zip(found, found[1:]):
            if i2 < i1:
                warns.append(f"階段順序異常：{p2} 早於 {p1}")

        r_speed, l_speed = (np.nan_to_num(series.speed[r, side_kp(x, "wrist")]) for x in ("R", "L"))
        seg.cycles.append(PitchCycle(
            index=k + 1,
            phases={ph: (int(series.frame_indices[i]) if i is not None else None) for ph, i in s.items()},
            times={ph: (round(float(series.times[i]), 3) if i is not None else None) for ph, i in s.items()},
            samples=s,
            release_side="R" if r_speed >= l_speed else "L",
            warnings=warns,
        ))
        boundary = s["follow_through"] if s["follow_through"] is not None else r + 1

    sides = [c.release_side for c in seg.cycles]
    detected = max(set(sides), key=sides.count) if sides else detect_throwing_side(series)
    if throwing_side:
        if detected != throwing_side:
            seg.warnings.append(f"骨架判斷為{'右' if detected == 'R' else '左'}手，與指定不同；"
                                "側拍時 YOLO 可能左右標反，以指定為準")
    else:
        seg.throwing_side = detected
        if len(set(sides)) > 1:
            seg.warnings.append("各球判斷的投球手不一致，骨架左右標記可能不可靠，建議用 --hand 指定")
    return seg
