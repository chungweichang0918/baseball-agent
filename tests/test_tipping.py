from bbagent.analysis.tipping import compare_pitches


def pitch(i, **kw):
    base = {"index": i, "set_duration_s": 1.0, "delivery_time_s": 1.1, "leg_lift_height": 0.13,
            "arm_slot_deg": 45.0, "release_height": 0.38, "release_forward": 0.0}
    return base | kw


def test_single_pitch_cannot_compare():
    r = compare_pitches([pitch(1)])
    assert r["flags"] == [] and r["confidence"] == "low" and "至少需要 2 球" in r["notes"][0]


def test_identical_pitches_have_no_flags():
    r = compare_pitches([pitch(1), pitch(2), pitch(3)])
    assert r["flags"] == [] and r["confidence"] == "medium"


def test_outlier_pitch_is_flagged():
    r = compare_pitches([pitch(1), pitch(2, arm_slot_deg=60.0), pitch(3), pitch(4, set_duration_s=1.05)])
    assert [(f["metric"], f["pitch"]) for f in r["flags"]] == [("arm_slot_deg", 2)]
    assert "第 2 球" in r["flags"][0]["text"]


def test_two_pitches_compare_difference():
    r = compare_pitches([pitch(1), pitch(2, set_duration_s=1.5, release_height=0.40)])
    assert [f["metric"] for f in r["flags"]] == ["set_duration_s"]       # 高度差 0.02 未超過門檻
    assert r["flags"][0]["pitch"] is None and r["confidence"] == "low"


def test_noisy_metric_raises_threshold():
    """三球以上且本來就不穩定的特徵，門檻會放寬，避免誤標。"""
    r = compare_pitches([pitch(1, arm_slot_deg=40), pitch(2, arm_slot_deg=50), pitch(3, arm_slot_deg=60),
                         pitch(4, arm_slot_deg=70)])
    assert r["flags"] == []


def test_missing_values_are_skipped():
    r = compare_pitches([pitch(1, arm_slot_deg=None), pitch(2, arm_slot_deg=None)])
    assert r["flags"] == []
