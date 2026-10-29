"""動作曲線圖：分段用的訊號與偵測到的階段時間。analyze_video.py 與 Agent 的 segment_motions 共用。

圖上文字用英文：matplotlib 預設字型沒有中文，換字型在 Mac 與 Colab 上都不一定有。
"""

from __future__ import annotations

from pathlib import Path

from .keypoints import SubjectSeries, side_kp
from .segmentation import PitchSegmentation
from .swing import SwingSegmentation

PHASE_COLORS = {"set": "tab:gray", "first_move": "tab:blue", "peak_leg_lift": "tab:green",
                "foot_strike": "tab:orange", "release": "tab:red", "follow_through": "tab:purple"}
SWING_PHASE_COLORS = {"stance": "tab:gray", "load": "tab:blue", "foot_plant": "tab:orange", "contact": "tab:red",
                      "follow_through": "tab:purple"}
LINE_COLORS = {"all": "black", "L": "tab:blue", "R": "tab:orange"}


def plot_motion_curves(series: SubjectSeries, seg: PitchSegmentation | SwingSegmentation, out: str | Path,
                       labels: dict | None = None, fps: float | None = None,
                       figsize: tuple[float, float] = (14, 10)) -> Path:
    """畫 4 條訊號與各球的階段（實線）；有標註時另外畫虛線，用來校正門檻。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = SWING_PHASE_COLORS if isinstance(seg, SwingSegmentation) else PHASE_COLORS
    t = series.times
    # 左右分開畫（藍 = L、橘 = R，與骨架疊圖相同），兩條線互相交錯的地方就是左右標反的跡象
    rows = [
        ("motion energy (body/s)", {"all": series.energy}),
        ("wrist speed (body/s)", {s: series.speed[:, side_kp(s, "wrist")] for s in "LR"}),
        ("knee height (body)", {s: series.knee_height(s) for s in "LR"}),
        ("ankle speed (body/s)", {s: series.speed[:, side_kp(s, "ankle")] for s in "LR"}),
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=figsize, sharex=True)
    for ax, (title, lines) in zip(axes, rows):
        for name, y in lines.items():
            ax.plot(t, y, color=LINE_COLORS[name], lw=1, label=None if name == "all" else name)
        if len(lines) > 1:
            ax.legend(loc="upper right", fontsize=7)
        ax.set_ylabel(title, fontsize=8)
        for c in seg.cycles:
            for ph, color in colors.items():
                if c.times[ph] is not None:
                    ax.axvline(c.times[ph], color=color, lw=1.5)
        if labels:  # 標註是原始幀號，用標註檔記錄的 fps 換算成秒，與偵測結果共用時間軸
            label_fps = labels.get("fps") or fps
            for lc in labels["cycles"]:
                for ph, f in lc.get("phases", {}).items():
                    ax.axvline(f / label_fps, color=colors[ph], lw=1.2, ls="--")
    axes[-1].set_xlabel("time (s)   solid = detected" + (", dashed = labeled" if labels else ""))
    handles = [plt.Line2D([], [], color=c, label=ph) for ph, c in colors.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(colors), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(out)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
