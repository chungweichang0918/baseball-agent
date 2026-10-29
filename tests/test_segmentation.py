import numpy as np
import pytest
from synthetic import LR_SWAP, make_pitcher

from bbagent.analysis.evaluation import compare_with_labels
from bbagent.analysis.keypoints import build_series
from bbagent.analysis.segmentation import PITCH_PHASES, segment_pitcher
from bbagent.vision.pose_scan import PoseTrack

# 容許誤差（原始幀，60 fps）。起停類階段本來就模糊，給得比峰值類寬。
# 固定姿勢最寬：它的起點是「慢慢減速到停下」的尾端，最後零點幾秒的動作已經和骨架雜訊分不出來。
TOLERANCE = {"set": 8, "first_move": 4, "peak_leg_lift": 2, "foot_strike": 4,
             "release": 2, "follow_through": 6}


def segment(**kw):
    track, t, frames, truth = make_pitcher(**kw)
    return segment_pitcher(build_series(track, t, frames)), truth


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("left_handed", [False, True])
def test_synthetic_pitches(stride, left_handed):
    seg, truth = segment(n_pitches=3, stride=stride, left_handed=left_handed)
    assert seg.throwing_side == ("L" if left_handed else "R")
    assert len(seg.cycles) == 3
    for cycle, answer in zip(seg.cycles, truth):
        assert cycle.warnings == []
        for ph in PITCH_PHASES:
            assert cycle.phases[ph] is not None, ph
            err = cycle.phases[ph] - answer[ph]
            assert abs(err) <= TOLERANCE[ph], f"第 {cycle.index} 球 {ph}：誤差 {err} 幀"


def test_left_right_label_swaps_do_not_break_segmentation():
    """每一球從抬腿到出手後 0.3 秒，骨架左右標記互換（側拍常見）。"""
    track, t, frames, truth = make_pitcher(n_pitches=2, stride=1)
    for a in truth:
        lo, hi = a["peak_leg_lift"] - 10, a["release"] + 18
        track.kpts[lo:hi] = track.kpts[lo:hi][:, LR_SWAP]
    series = build_series(track, t, frames)
    assert series.lr_fixes > 0
    seg = segment_pitcher(series)
    assert len(seg.cycles) == 2 and seg.throwing_side == "R"
    for cycle, answer in zip(seg.cycles, truth):
        for ph in PITCH_PHASES:
            assert abs(cycle.phases[ph] - answer[ph]) <= TOLERANCE[ph], ph


def test_user_specified_hand_wins():
    seg, _ = segment(n_pitches=2)
    assert seg.side_source == "skeleton" and seg.throwing_side == "R"
    track, t, frames, _ = make_pitcher(n_pitches=2)
    forced = segment_pitcher(build_series(track, t, frames), throwing_side="L")
    assert forced.side_source == "user" and forced.throwing_side == "L"
    assert any("與指定不同" in w for w in forced.warnings)


def test_lift_is_not_placed_at_release():
    seg, _ = segment(n_pitches=2)
    for c in seg.cycles:
        assert c.samples["release"] - c.samples["peak_leg_lift"] >= 4   # 至少 0.15 秒（30 Hz 取樣）


def test_heavier_noise_still_finds_peaks():
    seg, truth = segment(n_pitches=2, noise_px=3.0, seed=7)
    assert len(seg.cycles) == 2
    for cycle, answer in zip(seg.cycles, truth):
        for ph in ("peak_leg_lift", "release"):
            assert abs(cycle.phases[ph] - answer[ph]) <= TOLERANCE[ph] + 2


def test_static_person_has_no_pitches():
    n = 300
    rng = np.random.default_rng(0)
    kpts = np.tile(np.linspace([400, 300], [420, 700], 17), (n, 1, 1)) + rng.normal(0, 1, (n, 17, 2))
    boxes = np.tile([380.0, 280, 440, 720], (n, 1))
    track = PoseTrack(1, boxes, np.full(n, 0.9), kpts, np.full((n, 17), 0.9))
    frames = np.arange(0, n * 2, 2)
    seg = segment_pitcher(build_series(track, frames / 60, frames))
    assert seg.cycles == []
    assert seg.warnings == ["找不到出手動作"]


def test_compare_with_labels():
    seg, truth = segment(n_pitches=2)
    labels = {"cycles": [{"index": i + 1, "phases": a} for i, a in enumerate(truth)]}
    good = compare_with_labels(seg.cycles, labels, tolerance=TOLERANCE["set"])
    assert good["pass"], good
    assert [p["detected_index"] for p in good["pairs"]] == [1, 2]

    shifted = {"cycles": [{"index": i + 1, "phases": {ph: f + 10 for ph, f in a.items()}}
                          for i, a in enumerate(truth)]}
    bad = compare_with_labels(seg.cycles, shifted, tolerance=4)
    assert not bad["pass"]
    assert bad["per_phase"]["release"]["mean_error"] == pytest.approx(-10, abs=2)

    extra = {"cycles": labels["cycles"][:1]}
    one = compare_with_labels(seg.cycles, extra, tolerance=4)
    assert one["extra_detected"] == [2] and not one["pass"]
