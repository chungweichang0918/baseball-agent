"""從多個追蹤對象中選出要分析的主體（投手或打者）。

投手與打者用同一個評分：出現比例 × 身形大小 × log(1 + 手腕活動量)。
單人練習影片（投球練習、打擊場）主體幾乎一定是畫面中最大、出現最久、手腕動作最劇烈的人；
轉播的中外野視角同時拍到投手與打者時，分數可能接近，此時 confident 為 False，報告會提醒。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..vision.pose_scan import PoseScan
from .keypoints import KP, build_series

MIN_COVERAGE = 0.2        # 出現比例低於此值的對象不列入候選
CONFIDENT_MARGIN = 0.3    # 第一名分數領先第二名的比例達此值才算有把握


@dataclass
class SubjectCandidate:
    track_id: int
    coverage: float     # 出現在多少比例的取樣幀中
    size: float         # 偵測框高度中位數 ÷ 畫面高度
    activity: float     # 手腕速度的第 95 百分位數（身高/秒）
    score: float


@dataclass
class SubjectChoice:
    track_id: int | None
    confident: bool
    reason: str
    candidates: list[SubjectCandidate]

    def to_dict(self) -> dict:
        return asdict(self)


def select_subject(scan: PoseScan, subject: str = "pitcher") -> SubjectChoice:
    if subject not in ("pitcher", "batter"):
        raise ValueError(f"不支援的分析對象：{subject}")
    frame_h = scan.frame_size[1]
    candidates = []
    for tid, track in scan.tracks.items():
        if track.coverage < MIN_COVERAGE:
            continue
        series = build_series(track, scan.times, scan.frame_indices)
        wrists = series.speed[:, [KP["l_wrist"], KP["r_wrist"]]]
        activity = float(np.nanpercentile(wrists, 95)) if np.isfinite(wrists).any() else 0.0
        size = series.body_h / frame_h
        score = track.coverage * size * float(np.log1p(activity))
        candidates.append(SubjectCandidate(tid, round(track.coverage, 3), round(size, 3),
                                           round(activity, 3), round(score, 4)))
    candidates.sort(key=lambda c: c.score, reverse=True)

    if not candidates:
        return SubjectChoice(None, False, "沒有出現時間足夠長的人物", candidates)
    best = candidates[0]
    if len(candidates) == 1:
        return SubjectChoice(best.track_id, True, "畫面中只有一個候選人物", candidates)
    margin = (best.score - candidates[1].score) / best.score if best.score > 0 else 0.0
    confident = margin >= CONFIDENT_MARGIN
    reason = (f"分數領先第二名 {margin:.0%}" if confident
              else f"分數只領先第二名 {margin:.0%}，主體識別可能有誤")
    return SubjectChoice(best.track_id, confident, reason, candidates)
