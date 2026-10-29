import pytest
from synthetic import BOX_PAD, make_pitcher

from bbagent.analysis.keypoints import build_series
from bbagent.analysis.pitching_metrics import DEFAULT_FLIGHT_TIME_S, compute_pitching_metrics
from bbagent.analysis.segmentation import segment_pitcher

BODY_H = 700 - 295 + 2 * BOX_PAD   # 合成投手的偵測框高度：眼睛（最高點）到腳踝再加上下邊距


@pytest.fixture(scope="module")
def series_and_seg():
    track, t, frames, _ = make_pitcher(n_pitches=3)
    series = build_series(track, t, frames)
    return series, segment_pitcher(series)


def test_metrics_match_synthetic_geometry(series_and_seg):
    series, seg = series_and_seg
    m = compute_pitching_metrics(series, seg, "side")
    assert m["body_height_px"] == pytest.approx(BODY_H, abs=3)
    s = m["summary"]
    assert s["set_duration_s"]["mean"] == pytest.approx(1.0, abs=0.15)   # 固定姿勢起點最模糊
    assert s["delivery_time_s"]["mean"] == pytest.approx(1.1, abs=0.1)
    assert s["time_to_plate_est_s"]["mean"] == pytest.approx(1.1 + DEFAULT_FLIGHT_TIME_S, abs=0.1)
    assert s["leg_lift_height"]["mean"] == pytest.approx(60 / BODY_H, abs=0.02)      # 膝蓋高於髖部 60 px
    assert s["stride_length"]["mean"] == pytest.approx(210 / BODY_H, abs=0.03)      # 兩踝相距 210 px
    assert s["arm_slot_deg"]["mean"] == pytest.approx(45, abs=6)                     # 腕在肩前上方 (30, 30)
    assert s["release_height"]["mean"] == pytest.approx(170 / BODY_H, abs=0.03)
    assert s["release_forward"]["mean"] == pytest.approx(0, abs=0.05)
    assert s["arm_slot_deg"]["std"] < 3   # 三球動作相同，應該很一致
    assert len(m["per_pitch"]) == 3


def test_non_side_view_skips_stride(series_and_seg):
    series, seg = series_and_seg
    m = compute_pitching_metrics(series, seg, "center_field")
    assert "stride_length" not in m["summary"]
    assert any("跨步" in n for n in m["notes"])


def test_left_handed_release_forward_is_positive_direction():
    track, t, frames, _ = make_pitcher(n_pitches=2, left_handed=True)
    series = build_series(track, t, frames)
    m = compute_pitching_metrics(series, segment_pitcher(series))
    assert m["throwing_side"] == "L"
    assert m["summary"]["arm_slot_deg"]["mean"] == pytest.approx(45, abs=6)
    assert m["summary"]["release_forward"]["mean"] == pytest.approx(0, abs=0.05)
