"""Tipping 比對：找出某一球與其他球明顯不同的動作特徵。

這裡只能指出「哪一球、哪個特徵不一樣」。要說成「洩漏球種」，還需要知道各球的球種，
而且 2–3 球的樣本在統計上不足，所以信心等級最高只到 medium，報告必須用「疑似」。
"""

from __future__ import annotations

import numpy as np

from .pitching_metrics import METRIC_INFO

# 差異超過這些值才標記（經驗值）。時間單位是秒，位置單位是身高，角度單位是度
THRESHOLDS = {
    "set_duration_s": 0.3,
    "delivery_time_s": 0.15,
    "leg_lift_height": 0.05,
    "arm_slot_deg": 8.0,
    "release_height": 0.05,
    "release_forward": 0.05,
}
MAD_K = 2.5   # 三球以上時，門檻再放寬到 2.5 倍穩健標準差，避免本來就不穩定的特徵被誤標


def compare_pitches(per_pitch: list[dict], thresholds: dict[str, float] | None = None,
                    info: dict[str, tuple] | None = None, unit_word: str = "球",
                    notes: list[str] | None = None) -> dict:
    """預設比較投手各球；打者的揮棒一致性傳入打者的門檻、指標說明與「次揮棒」。"""
    thresholds = THRESHOLDS if thresholds is None else thresholds
    info = METRIC_INFO if info is None else info
    n = len(per_pitch)
    result = {"n_pitches": n, "flags": [], "confidence": "low" if n < 3 else "medium",
              "notes": list(notes) if notes is not None else ["沒有球種標籤，只能指出動作差異，不能直接對應到球種"]}
    if n < 2:
        result["notes"].insert(0, f"至少需要 2 {unit_word}才能比較")
        return result

    for key, base_thr in thresholds.items():
        vals = [(p["index"], p[key]) for p in per_pitch if p.get(key) is not None]
        if len(vals) < 2:
            continue
        name, unit, _ = info[key]
        arr = np.array([v for _, v in vals], float)
        if len(vals) == 2:
            diff = float(arr[1] - arr[0])
            if abs(diff) > base_thr:
                result["flags"].append({
                    "metric": key, "name": name, "unit": unit, "pitch": None,
                    "values": {str(i): round(v, 3) for i, v in vals}, "diff": round(diff, 3), "threshold": base_thr,
                    "text": f"{name}兩{unit_word[-1]}相差 {abs(diff):.3g} {unit}（第 {vals[0][0]} {unit_word} "
                            f"{vals[0][1]:.3g}、第 {vals[1][0]} {unit_word} {vals[1][1]:.3g}）",
                })
            continue
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) * 1.4826
        thr = max(base_thr, MAD_K * mad)
        for idx, v in vals:
            dev = v - med
            if abs(dev) > thr:
                result["flags"].append({
                    "metric": key, "name": name, "unit": unit, "pitch": idx,
                    "values": {str(i): round(x, 3) for i, x in vals}, "diff": round(dev, 3), "threshold": round(thr, 3),
                    "text": f"第 {idx} {unit_word}的{name}與其他{unit_word[-1]}不同（{v:.3g} {unit}，"
                            f"中位數 {med:.3g}）",
                })
    return result
