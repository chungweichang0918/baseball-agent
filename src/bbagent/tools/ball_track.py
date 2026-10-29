"""track_ball：在每一球出手後的短時間窗內細掃找球、擬合軌跡，粗估球速與到本壘時間，並畫出軌跡圖。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from ..analysis.ball_track import ASSUMED_HEIGHT_M, RELEASE_TO_PLATE_M, PitchBall, analyze_pitch_ball
from ..analysis.keypoints import KP, SubjectSeries
from ..analysis.segmentation import PitchSegmentation
from ..schemas import ToolResult
from ..vision.ball_scan import run_ball_scan
from ..vision.imageio import imwrite
from ..vision.render import caption, read_frames
from ..vision.video import VideoMeta

BEFORE_RELEASE_S = 0.05
AFTER_RELEASE_S = 0.6       # 球從出手到本壘約 0.4–0.6 秒
PATH_COLOR = (0, 230, 255)  # BGR 黃
POINT_COLOR = (0, 0, 255)


def wrists_at(series: SubjectSeries, t: np.ndarray) -> np.ndarray:
    """在任意時間點內插兩個手腕的位置，回傳 (N, 2, 2)；沒有資料的時間為 NaN。"""
    out = np.full((len(t), 2, 2), np.nan)
    for j, k in enumerate((KP["l_wrist"], KP["r_wrist"])):
        for d in range(2):
            y = series.pos[:, k, d]
            ok = np.isfinite(y)
            if ok.sum() >= 2:
                v = np.interp(t, series.times[ok], y[ok], left=np.nan, right=np.nan)
                out[:, j, d] = v
    return out


def _draw(video: str | Path, fps: float, ball: PitchBall, out_dir: Path) -> str | None:
    pts = np.array(ball.path)
    mid = int(round(pts[len(pts) // 2, 0] * fps))
    frame = read_frames(video, [mid]).get(mid)
    if frame is None:
        return None
    img = frame.copy()
    xy = pts[:, 1:].round().astype(int)
    cv2.polylines(img, [xy.reshape(-1, 1, 2)], False, PATH_COLOR, 2, cv2.LINE_AA)
    for x, y in xy:
        cv2.circle(img, (int(x), int(y)), 4, POINT_COLOR, -1, cv2.LINE_AA)
    speed = f" ~{ball.speed_kmh_est:.0f} km/h" if ball.speed_kmh_est else ""
    caption(img, f"P{ball.pitch} ball path {ball.points} pts{speed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"p{ball.pitch}_ball.jpg"
    return str(path) if imwrite(path, img) else None


def track_ball(video: str | Path, meta: VideoMeta, series: SubjectSeries, seg: PitchSegmentation, metrics: dict,
               out_dir: str | Path, speed_ok: bool, progress: Callable[[int, int], None] | None = None) -> ToolResult:
    fps = meta.fps
    windows = [(max(0, c.phases["release"] - round(BEFORE_RELEASE_S * fps)),
                min(meta.frame_count, c.phases["release"] + round(AFTER_RELEASE_S * fps))) for c in seg.cycles]
    if not windows:
        return ToolResult(status="failed", summary="沒有出手時間，無法找球", error="動作分段沒有找到出手")
    det, hit = run_ball_scan(video, fps, windows, progress=progress)
    frame_size = det.frame_size if det.frame_size[0] else (meta.width, meta.height)
    delivery = {p["index"]: p["delivery_time_s"] for p in metrics.get("per_pitch", [])}

    balls: list[PitchBall] = []
    artifacts = []
    for c, (a, b) in zip(seg.cycles, windows):
        m = (det.frames >= a) & (det.frames < b)
        t = det.times[m]
        ball = analyze_pitch_ball(c.index, t, det.xy[m], wrists_at(series, t), series.body_h, frame_size,
                                  c.times["release"], delivery.get(c.index), speed_ok)
        balls.append(ball)
        if ball.found:
            path = _draw(video, fps, ball, Path(out_dir) / "ball")
            if path:
                artifacts.append(path)

    lines = [f"細掃 {sum(b - a for a, b in windows)} 幀，候選球 {len(det.frames)} 個（{'命中快取' if hit else '已重新計算'}）"]
    for ball in balls:
        if not ball.found:
            lines.append(f"第 {ball.pitch} 球：找不到球軌跡")
        else:
            text = (f"第 {ball.pitch} 球：軌跡 {ball.points} 幀、可見 {ball.visible_s:.2f} 秒、"
                    f"出手後 {ball.start_after_release_s:+.2f} 秒開始")
            if ball.speed_kmh_est:
                text += f"、粗估球速 {ball.speed_kmh_est:.0f} 公里、飛行時間 {ball.flight_time_est_s:.2f} 秒"
            if ball.time_to_plate_est_s:
                text += f" → 到本壘時間（球軌跡估計）{ball.time_to_plate_est_s:.2f} 秒"
            lines.append(text)
        lines += [f"  ⚠️ {n}" for n in ball.notes]
    notes = [f"球速以投手身高 {ASSUMED_HEIGHT_M} 公尺換算像素，沒有相機校正，只能粗估",
             f"飛行時間假設出手點到本壘 {RELEASE_TO_PLATE_M} 公尺（成棒距離；少棒、壘球距離不同）"]
    lines += [f"※ {n}" for n in notes]

    found = [b for b in balls if b.found]
    status = "failed" if not found else ("ok" if len(found) == len(balls) and all(b.speed_kmh_est for b in found)
                                         else "degraded")
    data = {"cache_hit": hit, "candidates": int(len(det.frames)), "windows": windows,
            "pitches": [b.to_dict() for b in balls], "notes": notes}
    return ToolResult(status=status, summary="\n".join(lines), data=data, artifacts=artifacts,
                      error=None if found else "所有球都找不到可信的軌跡（球太小、太模糊或被遮住）")
