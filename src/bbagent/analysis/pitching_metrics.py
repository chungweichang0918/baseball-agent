"""投手指標：時間類（與鏡頭無關，最可靠）與姿勢類（以身高正規化）。

單一鏡頭沒有相機校正，像素無法換算成公尺，所以不算球速與絕對距離。
側拍時骨架左右可能標反，姿勢類指標一律用不分左右的方式取值：
抬腿看較高的膝蓋、跨步看兩踝距離、手臂角度用出手瞬間較快的那隻手腕與同側肩膀。
"""

from __future__ import annotations

import math

import numpy as np

from .keypoints import SubjectSeries, side_kp
from .segmentation import PitchSegmentation

# 時速約 145 公里、出手點到本壘約 16.8 公尺的飛行時間。沒有球軌跡時用它估算到本壘時間
DEFAULT_FLIGHT_TIME_S = 0.42

METRIC_INFO = {
    "set_duration_s":       ("固定姿勢停留時間", "秒", "all"),
    "delivery_time_s":      ("動作啟動到出手時間", "秒", "all"),
    "time_to_plate_est_s":  ("到本壘時間（估）", "秒", "all"),
    "leg_lift_height":      ("抬腿高度", "身高", "all"),
    "stride_length":        ("跨步長度", "身高", "side"),
    "arm_slot_deg":         ("出手手臂角度", "度", "all"),
    "release_forward":      ("出手點前後位置", "身高", "all"),
    "release_height":       ("出手點高度", "身高", "all"),
}


def _dt(series: SubjectSeries, a: int | None, b: int | None) -> float | None:
    if a is None or b is None:
        return None
    return round(float(series.times[b] - series.times[a]), 3)


def _finite(x: float) -> float | None:
    return round(float(x), 3) if np.isfinite(x) else None


def compute_pitching_metrics(series: SubjectSeries, seg: PitchSegmentation, camera_view: str = "side",
                             flight_time_s: float = DEFAULT_FLIGHT_TIME_S) -> dict:
    h = series.body_h
    hip = series.hip_center()
    knee_h = series.higher_knee()
    per_pitch = []
    for c in seg.cycles:
        s = c.samples
        m: dict[str, float | None] = dict.fromkeys(METRIC_INFO)
        m["set_duration_s"] = _dt(series, s["set"], s["first_move"])
        m["delivery_time_s"] = _dt(series, s["first_move"], s["release"])
        if m["delivery_time_s"] is not None:
            m["time_to_plate_est_s"] = round(m["delivery_time_s"] + flight_time_s, 3)

        if s["peak_leg_lift"] is not None:
            m["leg_lift_height"] = _finite(knee_h[s["peak_leg_lift"]])

        if s["foot_strike"] is not None and camera_view == "side":
            i = s["foot_strike"]
            gap = series.pos[i, side_kp("L", "ankle"), 0] - series.pos[i, side_kp("R", "ankle"), 0]
            m["stride_length"] = _finite(abs(gap) / h)

        r = s["release"]
        arm = c.release_side or seg.throwing_side
        shoulder = series.pos[r, side_kp(arm, "shoulder")]
        wrist = series.pos[r, side_kp(arm, "wrist")]
        dx, dy = wrist[0] - shoulder[0], shoulder[1] - wrist[1]  # 影像 y 軸向下
        if np.isfinite([dx, dy]).all():
            m["arm_slot_deg"] = round(math.degrees(math.atan2(dy, abs(dx))), 1)
        # 投球方向 = 出手瞬間手腕的水平速度方向，讓「前方」一律為正
        facing = np.sign(series.vel[r, side_kp(arm, "wrist"), 0]) or 1.0
        m["release_forward"] = _finite((wrist[0] - hip[r, 0]) * facing / h)
        m["release_height"] = _finite((hip[r, 1] - wrist[1]) / h)

        per_pitch.append({"index": c.index, **m})

    summary = {}
    for key, (name, unit, _views) in METRIC_INFO.items():
        values = [p[key] for p in per_pitch if p[key] is not None]
        if not values:
            continue
        arr = np.array(values)
        summary[key] = {
            "name": name, "unit": unit, "n": len(values),
            "mean": round(float(arr.mean()), 3),
            "std": round(float(arr.std(ddof=1)), 3) if len(values) > 1 else None,
            "min": round(float(arr.min()), 3), "max": round(float(arr.max()), 3),
        }
    notes = [f"到本壘時間為估計值：動作啟動到出手時間 + 固定飛行時間 {flight_time_s} 秒（尚未使用球軌跡）"]
    if camera_view != "side":
        notes.append("非側拍視角，不計算跨步長度")
    if series.lr_fixes:
        notes.append(f"修正了 {series.lr_fixes} 幀骨架左右標反")
    return {
        "throwing_side": seg.throwing_side,
        "side_source": seg.side_source,
        "camera_view": camera_view,
        "body_height_px": round(h, 1),
        "per_pitch": per_pitch,
        "summary": summary,
        "notes": notes,
    }
