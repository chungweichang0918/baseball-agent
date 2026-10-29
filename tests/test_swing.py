import numpy as np
import pytest
from synthetic import make_batter

from bbagent.analysis.batting_metrics import CONSISTENCY_THRESHOLDS, METRIC_INFO, compute_batting_metrics
from bbagent.analysis.keypoints import build_series
from bbagent.analysis.subject import select_subject
from bbagent.analysis.swing import SWING_PHASES, segment_batter
from bbagent.analysis.tipping import compare_pitches
from bbagent.vision.pose_scan import PoseTrack

# 容許誤差（60 fps 原始幀）。引棒比投手抬腿細微，門檻設得低，偵測會稍晚
TOLERANCE = {"stance": 12, "load": 8, "foot_plant": 4, "contact": 2, "follow_through": 6}


def series_of(**kw):
    track, t, frames, truth = make_batter(**kw)
    return build_series(track, t, frames), truth


@pytest.mark.parametrize("left_handed", [False, True])
def test_segment_batter_matches_truth(left_handed):
    series, truth = series_of(n_swings=3, left_handed=left_handed)
    seg = segment_batter(series)
    assert len(seg.cycles) == 3 and not seg.warnings
    for cycle, answer in zip(seg.cycles, truth):
        assert not cycle.warnings, cycle.warnings
        for ph in SWING_PHASES:
            err = cycle.phases[ph] - answer[ph]
            assert abs(err) <= TOLERANCE[ph], f"第 {cycle.index} 次揮棒 {ph}：誤差 {err} 幀"


def test_no_swing_found_for_still_person():
    n = 200
    kpts = np.tile(np.linspace([400, 300], [420, 700], 17), (n, 1, 1))
    still = PoseTrack(1, np.tile([380.0, 280, 440, 720], (n, 1)), np.full(n, 0.9), kpts, np.full((n, 17), 0.9))
    t = np.arange(n) / 30
    seg = segment_batter(build_series(still, t, np.arange(n)))
    assert seg.cycles == [] and "找不到揮棒動作" in seg.warnings[0]


def test_batting_metrics():
    series, _ = series_of(n_swings=2)
    m = compute_batting_metrics(series, segment_batter(series, batting_side="L"))
    assert m["batting_side"] == "L" and len(m["per_swing"]) == 2
    s = m["summary"]
    # 合成資料：引棒 2.0 秒、著地 2.8 秒、擊球 2.9 秒
    assert s["plant_to_contact_s"]["mean"] == pytest.approx(0.1, abs=0.07)
    assert s["load_to_contact_s"]["mean"] == pytest.approx(0.9, abs=0.15)
    assert s["stride_length"]["mean"] > 0.3            # 前腳跨出 180 像素、身高約 445 像素
    assert s["peak_hand_speed"]["mean"] > 3
    assert s["head_movement"]["mean"] < 0.02           # 合成打者的頭沒有動
    assert set(s) <= set(METRIC_INFO)

    rear = compute_batting_metrics(series, segment_batter(series), camera_view="other")
    assert "stride_length" not in rear["summary"] and any("非側拍" in n for n in rear["notes"])


def test_swing_consistency_flags_outlier():
    per = [{"index": i, "load_to_contact_s": v, "plant_to_contact_s": 0.15} for i, v in enumerate([0.9, 0.92, 1.3], 1)]
    r = compare_pitches(per, CONSISTENCY_THRESHOLDS, METRIC_INFO, unit_word="次揮棒", notes=[])
    assert [f["pitch"] for f in r["flags"]] == [3] and "第 3 次揮棒" in r["flags"][0]["text"]
    one = compare_pitches(per[:1], CONSISTENCY_THRESHOLDS, METRIC_INFO, unit_word="次揮棒", notes=[])
    assert "至少需要 2 次揮棒才能比較" in one["notes"]


def test_select_batter_subject():
    from synthetic import make_scan

    batter, _, frames, _ = make_batter(n_swings=2, track_id=1)
    n = len(frames)
    bystander = PoseTrack(2, np.tile([100.0, 500, 160, 700], (n, 1)), np.full(n, 0.9),
                          np.tile(np.linspace([110, 510], [150, 690], 17), (n, 1, 1)), np.full((n, 17), 0.9))
    choice = select_subject(make_scan([batter, bystander], frames), "batter")
    assert choice.track_id == 1 and choice.confident
    with pytest.raises(ValueError):
        select_subject(make_scan([batter], frames), "umpire")
