"""打者指標：時間類（與鏡頭無關）與姿勢類（以身高正規化，只在側拍時可靠）。

沒有球棒偵測與相機校正，所以不算棒速、擊球仰角這類需要球棒或球軌跡的數值；
手速以手腕速度代替，單位是身高/秒，只能用來比較同一位打者的不同揮棒。
"""

from __future__ import annotations

import numpy as np

from .keypoints import KP, SubjectSeries, side_kp
from .pitching_metrics import _dt, _finite
from .swing import SwingSegmentation

METRIC_INFO = {
    "load_to_contact_s":  ("引棒到擊球時間", "秒", "all"),
    "stride_time_s":      ("引棒到跨步著地時間", "秒", "all"),
    "plant_to_contact_s": ("著地到擊球時間", "秒", "all"),
    "peak_hand_speed":    ("最大手速", "身高/秒", "all"),
    "stride_length":      ("跨步長度", "身高", "side"),
    "head_movement":      ("頭部移動量", "身高", "all"),
    "contact_hand_height": ("擊球時手的高度", "身高", "all"),
}

# 兩次揮棒差異超過這些值才標記（經驗值）
CONSISTENCY_THRESHOLDS = {
    "load_to_contact_s": 0.1,
    "plant_to_contact_s": 0.05,
    "stride_length": 0.05,
    "head_movement": 0.05,
    "contact_hand_height": 0.05,
}


def compute_batting_metrics(series: SubjectSeries, seg: SwingSegmentation, camera_view: str = "side") -> dict:
    h = series.body_h
    hip = series.hip_center()
    hand_speed = series.max_speed("wrist")
    nose = series.pos[:, KP["nose"]]
    wrists = (series.pos[:, KP["l_wrist"]] + series.pos[:, KP["r_wrist"]]) / 2    # 雙手握棒，取中點
    per_swing = []
    for c in seg.cycles:
        s = c.samples
        m: dict[str, float | None] = dict.fromkeys(METRIC_INFO)
        m["load_to_contact_s"] = _dt(series, s["load"], s["contact"])
        m["stride_time_s"] = _dt(series, s["load"], s["foot_plant"])
        m["plant_to_contact_s"] = _dt(series, s["foot_plant"], s["contact"])
        m["peak_hand_speed"] = _finite(hand_speed[s["contact"]])
        if s["foot_plant"] is not None and camera_view == "side":
            i = s["foot_plant"]
            gap = series.pos[i, side_kp("L", "ankle"), 0] - series.pos[i, side_kp("R", "ankle"), 0]
            m["stride_length"] = _finite(abs(gap) / h)
        start = s["load"] if s["load"] is not None else s["stance"]
        if start is not None:
            m["head_movement"] = _finite(np.linalg.norm(nose[s["contact"]] - nose[start]) / h)
        m["contact_hand_height"] = _finite((hip[s["contact"], 1] - wrists[s["contact"], 1]) / h)
        per_swing.append({"index": c.index, **m})

    summary = {}
    for key, (name, unit, _views) in METRIC_INFO.items():
        values = [p[key] for p in per_swing if p[key] is not None]
        if not values:
            continue
        arr = np.array(values)
        summary[key] = {
            "name": name, "unit": unit, "n": len(values),
            "mean": round(float(arr.mean()), 3),
            "std": round(float(arr.std(ddof=1)), 3) if len(values) > 1 else None,
            "min": round(float(arr.min()), 3), "max": round(float(arr.max()), 3),
        }
    notes = ["手速是手腕速度（身高/秒），沒有偵測球棒，不是棒速；只適合比較同一位打者的不同揮棒"]
    if camera_view != "side":
        notes.append("非側拍視角，不計算跨步長度；頭部移動量只反映畫面平面上的移動")
    if series.lr_fixes:
        notes.append(f"修正了 {series.lr_fixes} 幀骨架左右標反")
    return {
        "batting_side": seg.batting_side,
        "camera_view": camera_view,
        "body_height_px": round(h, 1),
        "per_swing": per_swing,
        "summary": summary,
        "notes": notes,
    }
