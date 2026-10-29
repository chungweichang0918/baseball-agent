from pathlib import Path

import cv2
import numpy as np
import pytest
from synthetic import make_pitcher

from bbagent.analysis.ball_track import ASSUMED_HEIGHT_M, analyze_pitch_ball, filter_candidates, fit_trajectory
from bbagent.analysis.keypoints import KP, build_series
from bbagent.analysis.pitching_metrics import compute_pitching_metrics
from bbagent.analysis.segmentation import segment_pitcher
from bbagent.tools import ball_track as ball_tool
from bbagent.vision.ball_scan import BallDetections

FPS = 60
FRAME = (1000, 800)
SPEED_BODY = 20.0      # 身高/秒，換算約 130 公里


def synthetic_detections(series, releases, seed=0, with_ball=True):
    """每一球出手後 0.02–0.2 秒的球（等速 + 輕微下墜），加上三種誤報：手上的球、靜止物、隨機雜點。"""
    rng = np.random.default_rng(seed)
    rows = []
    vx = SPEED_BODY * series.body_h
    for r_t in releases:
        start = round(r_t * FPS) - 3
        wrist = series.pos[np.argmin(abs(series.times - r_t)), KP["r_wrist"]]
        for f in range(start, start + 36):
            t = f / FPS
            u = t - r_t
            if with_ball and 0.02 <= u <= 0.2:
                rows.append((f, wrist[0] + vx * u + rng.normal(0, 1), wrist[1] + 400 * u ** 2 + rng.normal(0, 1), 8, 0.4))
            if u < 0.02:
                rows.append((f, wrist[0] + 3, wrist[1] - 2, 10, 0.6))                 # 還在手上
            rows.append((f, 120 + rng.normal(0, 0.5), 650 + rng.normal(0, 0.5), 9, 0.3))  # 地上的球
            if rng.random() < 0.3:
                rows.append((f, rng.uniform(0, FRAME[0]), rng.uniform(0, FRAME[1]), 7, 0.15))
    arr = np.array(rows, float)
    return BallDetections(arr[:, 0].astype(int), arr[:, 1:3], arr[:, 3], arr[:, 4], FPS, FRAME, [], {})


@pytest.fixture
def pitcher():
    track, t, frames, _ = make_pitcher(n_pitches=2, stride=1, fps=FPS)
    series = build_series(track, t, frames)
    seg = segment_pitcher(series, throwing_side="R")
    return series, seg, compute_pitching_metrics(series, seg)


def test_filters_remove_hand_and_static(pitcher):
    series, seg, _ = pitcher
    det = synthetic_detections(series, [c.times["release"] for c in seg.cycles])
    keep, removed = filter_candidates(det.xy, ball_tool.wrists_at(series, det.times), series.body_h, FRAME)
    assert removed["near_hand"] > 0 and removed["static"] > 0
    assert not np.any(np.hypot(det.xy[keep, 0] - 120, det.xy[keep, 1] - 650) < 5)


def test_fit_recovers_trajectory_among_outliers():
    rng = np.random.default_rng(1)
    t = np.arange(12) / FPS
    xy = np.stack([200 + 3000 * t, 300 + 50 * t + 500 * t ** 2], axis=1) + rng.normal(0, 1, (12, 2))
    noise_t = rng.choice(t, 20)
    noise = rng.uniform(0, 800, (20, 2))
    fit = fit_trajectory(np.r_[t, noise_t], np.r_[xy, noise], 800)
    assert fit is not None and len(fit.inliers) >= 11 and fit.residual_px < 3
    assert fit.speed_px(0.0) == pytest.approx(np.hypot(3000, 50), rel=0.03)
    assert fit_trajectory(noise_t, noise, 800) is None or len(fit_trajectory(noise_t, noise, 800).inliers) < 11


def test_analyze_pitch_ball_speed_and_time(pitcher):
    series, seg, metrics = pitcher
    c = seg.cycles[0]
    det = synthetic_detections(series, [c.times["release"]])
    ball = analyze_pitch_ball(1, det.times, det.xy, ball_tool.wrists_at(series, det.times), series.body_h, FRAME,
                              c.times["release"], metrics["per_pitch"][0]["delivery_time_s"], speed_ok=True)
    assert ball.found and ball.points >= 10, ball.notes
    assert ball.speed_kmh_est == pytest.approx(SPEED_BODY * ASSUMED_HEIGHT_M * 3.6, rel=0.05)
    assert ball.time_to_plate_est_s == pytest.approx(metrics["per_pitch"][0]["delivery_time_s"]
                                                      + ball.flight_time_est_s, abs=0.002)
    assert 0 < ball.start_after_release_s < 0.06

    rear = analyze_pitch_ball(1, det.times, det.xy, ball_tool.wrists_at(series, det.times), series.body_h, FRAME,
                              c.times["release"], 1.0, speed_ok=False)
    assert rear.found and rear.speed_kmh_est is None and any("不換算球速" in n for n in rear.notes)


def test_track_ball_tool(pitcher, tmp_path, monkeypatch):
    series, seg, metrics = pitcher
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, FRAME)
    for _ in range(round(10.5 * FPS)):
        writer.write(np.zeros((FRAME[1], FRAME[0], 3), np.uint8))
    writer.release()
    from bbagent.vision.video import probe_video

    meta = probe_video(video)
    releases = [c.times["release"] for c in seg.cycles]
    calls = []

    def fake_scan(path, fps, windows, **kw):
        calls.append(windows)
        return synthetic_detections(series, releases), False

    monkeypatch.setattr(ball_tool, "run_ball_scan", fake_scan)
    result = ball_tool.track_ball(video, meta, series, seg, metrics, tmp_path, speed_ok=True)
    assert result.status == "ok", result.summary
    assert len(calls[0]) == 2 and all(b - a == round(0.65 * FPS) for a, b in calls[0])
    assert "到本壘時間（球軌跡估計）" in result.summary and len(result.artifacts) == 2
    assert all(Path(p).exists() for p in result.artifacts)

    monkeypatch.setattr(ball_tool, "run_ball_scan",
                        lambda *a, **k: (synthetic_detections(series, releases, with_ball=False), True))
    missing = ball_tool.track_ball(video, meta, series, seg, metrics, tmp_path, speed_ok=True)
    assert missing.status == "failed" and "找不到球軌跡" in missing.summary
