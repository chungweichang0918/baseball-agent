"""打者揮棒分段：從骨架時間序列找出每一次揮棒的 5 個階段。

和投手一樣先找最明確的「擊球」，再往前、往後推：
  ④ 擊球        兩手腕中較快者的速度峰值（雙手握棒，揮棒最快的瞬間接近擊球點）
  ③ 跨步著地     擊球前，兩腳踝中較快者衝到最快、之後速度掉到峰值一定比例以下的第一個時刻
  ② 引棒        從著地（找不到就從擊球）往回推，動作能量降到「動作門檻」以下的下一個時刻
  ① 準備姿勢     引棒之前，能量持續低於「靜止門檻」的那段時間的起點
  ⑤ 收棒        擊球後動作能量降回「收尾門檻」以下的第一個時刻

沒有揮棒的球（看球、擺棒不揮）不會被偵測到。
跨步不明顯的打者（例如不跨步、只抬腳跟）找不到著地時，引棒改從擊球往回推。
門檻相對於每次揮棒自己的靜止基準與峰值；預設值用合成資料調出來，要用真實影片再校正。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .keypoints import SubjectSeries
from .segmentation import _nanargmax, find_motion_start, find_settle

SWING_PHASES = ["stance", "load", "foot_plant", "contact", "follow_through"]
SWING_PHASE_ZH = {"stance": "準備姿勢", "load": "引棒", "foot_plant": "跨步著地", "contact": "擊球",
                  "follow_through": "收棒"}
START_WARNINGS = {
    "early": "準備姿勢可能比搜尋範圍更早開始",
    "short": "引棒前沒有足夠長的靜止期",
    "no_rest": "搜尋範圍內找不到靜止期，無法判定引棒",
    "no_data": "動作能量資料不足，無法判定引棒與準備姿勢",
}


@dataclass
class SwingSegParams:
    min_swing_interval_s: float = 1.5   # 兩次揮棒至少相隔幾秒
    contact_peak_rel: float = 0.5       # 擊球峰值至少要達到全片最大手腕速度的此比例
    min_contact_speed: float = 2.5      # 擊球峰值的絕對下限（身高/秒），排除擺棒、整理護具等小動作
    stride_window_s: float = 1.0        # 擊球前多久內找跨步
    min_plant_to_contact_s: float = 0.03
    min_stride_speed: float = 0.6       # 腳踝峰值低於此值（身高/秒）視為沒有明顯跨步
    plant_rel: float = 0.2              # 前腳速度降到跨步峰值的此比例以下即視為著地
    rest_window_s: float = 2.5          # 往回找靜止基準的範圍
    load_energy_window_s: float = 0.5   # 錨點前多久內找動作能量峰值
    rest_percentile: float = 10.0
    move_rel: float = 0.1               # 引棒動作比投手抬腿細微，門檻比投手低
    still_rel: float = 0.05
    min_stance_s: float = 0.3
    spike_s: float = 0.15
    follow_window_s: float = 1.5
    follow_rel: float = 0.05


@dataclass
class SwingCycle:
    index: int                                  # 第幾次揮棒（從 1 開始）
    phases: dict[str, int | None]               # 原始幀號
    times: dict[str, float | None]              # 秒
    samples: dict[str, int | None] = field(repr=False, default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SwingSegmentation:
    batting_side: str | None                    # "R"（右打）、"L"（左打）或 None（未指定）
    cycles: list[SwingCycle]
    params: SwingSegParams
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for c in d["cycles"]:
            c.pop("samples")
        return d


def segment_batter(series: SubjectSeries, params: SwingSegParams | None = None,
                   batting_side: str | None = None) -> SwingSegmentation:
    p = params or SwingSegParams()
    n = len(series.times)
    win = series.seconds_to_samples
    hand_speed = np.nan_to_num(series.max_speed("wrist"))
    ankle_speed = series.max_speed("ankle")
    energy = series.energy

    seg = SwingSegmentation(batting_side, [], p)
    if hand_speed.max() <= 0:
        seg.warnings.append("手腕沒有任何有效的速度資料")
        return seg
    height = max(p.contact_peak_rel * hand_speed.max(), p.min_contact_speed)
    contacts, _ = find_peaks(hand_speed, height=height, distance=win(p.min_swing_interval_s))
    if len(contacts) == 0:
        seg.warnings.append("找不到揮棒動作（看球不揮的球不會被偵測）")

    boundary = 0
    for k, c in enumerate(contacts):
        c = int(c)
        warns: list[str] = []
        s: dict[str, int | None] = dict.fromkeys(SWING_PHASES)
        s["contact"] = c

        # ③ 跨步著地
        lo = max(boundary, c - win(p.stride_window_s))
        hi = c - win(p.min_plant_to_contact_s)
        m_rel = _nanargmax(ankle_speed[lo:hi + 1]) if hi >= lo else None
        if m_rel is not None and ankle_speed[lo + m_rel] >= p.min_stride_speed:
            m = lo + m_rel
            below = np.flatnonzero(ankle_speed[m:c + 1] < p.plant_rel * ankle_speed[m])
            s["foot_plant"] = m + int(below[0]) if len(below) else None
        if s["foot_plant"] is None:
            warns.append("找不到明顯的跨步著地（可能是不跨步的打法或腳被遮住）")

        # ② 引棒、① 準備姿勢
        anchor = s["foot_plant"] if s["foot_plant"] is not None else c
        start = find_motion_start(series, anchor, boundary, p.rest_window_s, p.load_energy_window_s,
                                  p.rest_percentile, p.move_rel, p.still_rel, p.spike_s, p.min_stance_s)
        s["load"], s["stance"] = start.move, start.still
        warns += [START_WARNINGS[w] for w in start.problems]

        # ⑤ 收棒
        next_c = int(contacts[k + 1]) if k + 1 < len(contacts) else n
        s["follow_through"] = find_settle(energy, c, min(n - 1, c + win(p.follow_window_s), next_c - 1),
                                          start.rest, p.follow_rel, win(0.3))
        if s["follow_through"] is None:
            warns.append("擊球後找不到動作停下的時刻")

        found = [(ph, s[ph]) for ph in SWING_PHASES if s[ph] is not None]
        for (p1, i1), (p2, i2) in zip(found, found[1:]):
            if i2 < i1:
                warns.append(f"階段順序異常：{p2} 早於 {p1}")

        seg.cycles.append(SwingCycle(
            index=k + 1,
            phases={ph: (int(series.frame_indices[i]) if i is not None else None) for ph, i in s.items()},
            times={ph: (round(float(series.times[i]), 3) if i is not None else None) for ph, i in s.items()},
            samples=s,
            warnings=warns,
        ))
        boundary = s["follow_through"] if s["follow_through"] is not None else c + 1
    return seg
