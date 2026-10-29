"""把演算法的分段結果和人工標註（data/samples/labels/*.json）比對。"""

from __future__ import annotations

import numpy as np

from .segmentation import PITCH_PHASES, PitchCycle

DEFAULT_TOLERANCE = 3   # 第 2 階段完成標準：平均誤差 ≤ 3 幀


def compare_with_labels(cycles: list[PitchCycle], labels: dict, tolerance: int = DEFAULT_TOLERANCE) -> dict:
    """誤差 = 偵測幀號 − 標註幀號（正值表示偵測得太晚）。

    每個標註週期配對到共同階段平均誤差最小的偵測週期，每個偵測週期最多配對一次。
    """
    labeled = [c for c in labels["cycles"] if c.get("phases")]
    costs = []
    for li, lc in enumerate(labeled):
        for di, dc in enumerate(cycles):
            common = [ph for ph, f in lc["phases"].items() if dc.phases.get(ph) is not None]
            if common:
                costs.append((float(np.mean([abs(dc.phases[ph] - lc["phases"][ph]) for ph in common])), li, di))

    used_l, used_d, pairs = set(), set(), []
    for _, li, di in sorted(costs):
        if li in used_l or di in used_d:
            continue
        used_l.add(li)
        used_d.add(di)
        lc, dc = labeled[li], cycles[di]
        errors = {ph: dc.phases[ph] - f for ph, f in lc["phases"].items() if dc.phases.get(ph) is not None}
        missing = [ph for ph in lc["phases"] if dc.phases.get(ph) is None]
        pairs.append({"label_index": lc["index"], "detected_index": dc.index, "errors": errors, "missing": missing})
    pairs.sort(key=lambda p: p["label_index"])

    per_phase = {}
    for ph in PITCH_PHASES:
        errs = [p["errors"][ph] for p in pairs if ph in p["errors"]]
        n_missing = sum(ph in p["missing"] for p in pairs)
        if not errs and not n_missing:
            continue
        abs_errs = np.abs(errs) if errs else np.array([])
        per_phase[ph] = {
            "n": len(errs),
            "missing": n_missing,
            "mean_error": round(float(np.mean(errs)), 2) if errs else None,
            "mean_abs_error": round(float(abs_errs.mean()), 2) if errs else None,
            "max_abs_error": int(abs_errs.max()) if errs else None,
            "pass": bool(errs) and n_missing == 0 and float(abs_errs.mean()) <= tolerance,
        }
    unmatched_labels = [lc["index"] for li, lc in enumerate(labeled) if li not in used_l]
    extra_detected = [dc.index for di, dc in enumerate(cycles) if di not in used_d]
    return {
        "tolerance_frames": tolerance,
        "pairs": pairs,
        "per_phase": per_phase,
        "unmatched_labels": unmatched_labels,
        "extra_detected": extra_detected,
        "pass": bool(per_phase) and all(v["pass"] for v in per_phase.values())
                and not unmatched_labels and not extra_detected,
    }
