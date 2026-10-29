import numpy as np
from synthetic import LR_SWAP, make_pitcher

from bbagent.analysis.keypoints import fix_lr_swaps


def swap_labels(kpts: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """模擬 YOLO 在側拍時左右標反：[lo, hi) 這段的左右標記互換。"""
    out = kpts.copy()
    out[lo:hi] = out[lo:hi][:, LR_SWAP]
    return out


def test_clean_data_is_untouched():
    track, *_ = make_pitcher(n_pitches=2, stride=1)
    fixed, n = fix_lr_swaps(track.kpts)
    assert n == 0
    np.testing.assert_array_equal(fixed, track.kpts)


def test_swapped_stretch_is_restored():
    track, t, frames, truth = make_pitcher(n_pitches=2, stride=1)
    lo, hi = truth[0]["peak_leg_lift"] - 10, truth[0]["release"] + 18
    fixed, n = fix_lr_swaps(swap_labels(track.kpts, lo, hi))
    assert n == hi - lo
    np.testing.assert_allclose(fixed[:, 5:], track.kpts[:, 5:])   # 手腳都換回來（臉部不在檢查範圍）


def test_swapped_from_first_frame_stays_consistent():
    """一開始就標反時無法知道真正的左右，但整段應維持一致，不會在中途跳來跳去。"""
    track, *_ = make_pitcher(n_pitches=1, stride=1)
    all_swapped = swap_labels(track.kpts, 0, len(track.kpts))
    fixed, n = fix_lr_swaps(all_swapped)
    assert n == 0
    np.testing.assert_array_equal(fixed, all_swapped)


def test_missing_frames_are_skipped():
    track, *_ = make_pitcher(n_pitches=1, stride=1)
    kpts = track.kpts.copy()
    kpts[100:110, 9] = np.nan   # 左手腕短暫消失
    fixed, n = fix_lr_swaps(kpts)
    assert n == 0
    assert np.isnan(fixed[100:110, 9]).all()
