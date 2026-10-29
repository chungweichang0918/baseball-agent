"""render_keyframes：把每一球（或每次揮棒）的關鍵階段畫成疊圖（主體框、其他人框、骨架），給 UI 與 Gemini 使用。"""

from __future__ import annotations

from pathlib import Path

from ..analysis.segmentation import PHASE_ZH, PitchSegmentation
from ..analysis.swing import SWING_PHASE_ZH, SwingSegmentation
from ..schemas import ToolResult
from ..vision.imageio import imwrite
from ..vision.pose_scan import PoseScan
from ..vision.render import caption, draw_box, draw_skeleton, read_frames

KEY_PHASES = ["set", "peak_leg_lift", "foot_strike", "release", "follow_through"]
SWING_KEY_PHASES = ["stance", "load", "foot_plant", "contact", "follow_through"]
MAX_KEYFRAMES = 12
SUBJECT_COLOR = (60, 220, 60)
OTHER_COLOR = (150, 150, 150)


def render_keyframes(video: str | Path, scan: PoseScan, track_id: int, seg: PitchSegmentation | SwingSegmentation,
                     out_dir: str | Path) -> ToolResult:
    batter = isinstance(seg, SwingSegmentation)
    phases, zh, unit = (SWING_KEY_PHASES, SWING_PHASE_ZH, "次揮棒") if batter else (KEY_PHASES, PHASE_ZH, "球")
    items = [(c.index, ph, c.phases[ph], c.samples[ph], c.times[ph])
             for c in seg.cycles for ph in phases if c.phases[ph] is not None][:MAX_KEYFRAMES]
    if not items:
        return ToolResult(status="failed", summary="沒有可繪製的關鍵幀", error="動作分段沒有找到任何階段")

    out = Path(out_dir) / "keyframes"
    out.mkdir(parents=True, exist_ok=True)
    frames = read_frames(video, [f for _, _, f, _, _ in items])
    subject = scan.tracks[track_id]
    keyframes = []
    for pitch, phase, frame_idx, sample, t in items:
        if frame_idx not in frames:
            continue
        img = frames[frame_idx].copy()
        for tid, track in scan.tracks.items():
            if tid != track_id and track.present[sample]:
                draw_box(img, track.boxes[sample], OTHER_COLOR, 1)
        draw_box(img, subject.boxes[sample], SUBJECT_COLOR, 3)
        draw_skeleton(img, subject.kpts[sample], subject.kconf[sample])
        caption(img, f"P{pitch} {phase} t={t:.2f}s")   # OpenCV 只能畫 ASCII，中文說明放在 data
        path = out / f"p{pitch}_{phase}.jpg"
        if not imwrite(path, img):
            continue
        keyframes.append({"pitch": pitch, "phase": phase, "frame": frame_idx, "time_s": t, "path": str(path),
                          "caption": f"第 {pitch} {unit}｜{zh[phase]}｜{t:.2f} 秒"})

    return ToolResult(
        status="ok" if keyframes else "failed",
        summary=f"繪製 {len(keyframes)} 張關鍵幀：" + "、".join(k["caption"] for k in keyframes),
        data={"keyframes": keyframes},
        artifacts=[k["path"] for k in keyframes],
    )
