import numpy as np
from synthetic import make_pitcher, make_scan

from bbagent.analysis.subject import select_subject
from bbagent.vision.pose_scan import PoseScan, PoseTrack, merge_fragments


def _split(track: PoseTrack, start: int, end: int, new_id: int) -> tuple[PoseTrack, PoseTrack]:
    """把 track 拆成 [0, start) 與 [end, N) 兩段，中間留空。"""
    a = PoseTrack(track.track_id, track.boxes.copy(), track.conf.copy(), track.kpts.copy(), track.kconf.copy())
    b = PoseTrack(new_id, track.boxes.copy(), track.conf.copy(), track.kpts.copy(), track.kconf.copy())
    idx = np.arange(len(track.conf))
    for part, drop in ((a, idx >= start), (b, idx < end)):
        for arr in (part.boxes, part.conf, part.kpts, part.kconf):
            arr[drop] = np.nan
    return a, b


def _bystander(n: int, track_id: int, x: float = 100) -> PoseTrack:
    kpts = np.tile(np.linspace([x, 600], [x + 10, 700], 17), (n, 1, 1))
    boxes = np.tile([x - 10, 590, x + 20, 710], (n, 1)).astype(float)
    return PoseTrack(track_id, boxes, np.full(n, 0.8), kpts, np.full((n, 17), 0.8))


def test_save_load_roundtrip(tmp_path):
    track, t, frames, _ = make_pitcher(n_pitches=1)
    scan = make_scan([track, _bystander(len(t), 5)], frames)
    path = tmp_path / "scan.npz"
    scan.save(path)
    loaded = PoseScan.load(path)
    assert loaded.fps == scan.fps and loaded.frame_size == scan.frame_size
    assert sorted(loaded.tracks) == [1, 5]
    np.testing.assert_array_equal(loaded.tracks[1].kpts, track.kpts)

    empty = make_scan([], frames)
    empty.save(tmp_path / "empty.npz")
    assert PoseScan.load(tmp_path / "empty.npz").tracks == {}


def test_merge_fragments_joins_nearby_gap():
    track, t, frames, _ = make_pitcher(n_pitches=1, stride=2)
    a, b = _split(track, 50, 55, new_id=2)        # 5 個取樣幀 ≈ 0.17 秒的缺口
    scan = make_scan([a, b], frames)
    merged = merge_fragments(scan)
    assert list(merged.tracks) == [1]
    assert merged.tracks[1].present.sum() == track.present.sum() - 5
    assert scan.tracks[2].present.any()           # 原本的 scan 沒有被改到


def test_merge_fragments_keeps_far_or_late_tracks():
    track, t, frames, _ = make_pitcher(n_pitches=1, stride=2)
    a, b = _split(track, 50, 80, new_id=2)        # 30 個取樣幀 = 1 秒，超過 0.5 秒上限
    assert len(merge_fragments(make_scan([a, b], frames)).tracks) == 2

    a, b = _split(track, 50, 55, new_id=2)
    b.boxes += 600                                # 離太遠，不是同一個人
    b.kpts += 600
    assert len(merge_fragments(make_scan([a, b], frames)).tracks) == 2


def test_select_subject_prefers_big_active_person():
    track, t, frames, _ = make_pitcher(n_pitches=2)
    scan = make_scan([_bystander(len(t), 7), track], frames)
    choice = select_subject(scan)
    assert choice.track_id == 1 and choice.confident
    assert [c.track_id for c in choice.candidates] == [1, 7]


def test_select_subject_flags_ambiguity():
    t1, t, frames, _ = make_pitcher(n_pitches=2, track_id=1)
    t2, *_ = make_pitcher(n_pitches=2, track_id=2, seed=1)
    choice = select_subject(make_scan([t1, t2], frames))
    assert not choice.confident and "可能有誤" in choice.reason


def test_select_subject_ignores_brief_tracks():
    track, t, frames, _ = make_pitcher(n_pitches=1)
    brief = _bystander(len(t), 9)
    brief.conf[10:] = np.nan
    choice = select_subject(make_scan([track, brief], frames))
    assert [c.track_id for c in choice.candidates] == [1]
