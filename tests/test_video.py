import cv2
import numpy as np
import pytest

from bbagent.vision.cache import file_sha1
from bbagent.vision.video import iter_frames, probe_video


@pytest.fixture
def two_scene_video(tmp_path):
    """30 幀紅色畫面接 30 幀藍色畫面，第 30 幀是場景切換。"""
    path = tmp_path / "two_scenes.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(60):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:] = (0, 0, 220) if i < 30 else (220, 0, 0)
        frame[10:20, (i % 40):(i % 40) + 10] = 255     # 小幅移動的白塊，不應被當成切換
        writer.write(frame)
    writer.release()
    return path


def test_probe_video(two_scene_video):
    meta = probe_video(two_scene_video)
    assert meta.frame_count == 60
    assert (meta.width, meta.height) == (64, 48)
    assert meta.fps == pytest.approx(30)
    assert meta.scene_cuts == [30]
    assert meta.to_dict()["duration_s"] == pytest.approx(2.0)


def test_iter_frames_stride_and_range(two_scene_video):
    assert [i for i, _ in iter_frames(two_scene_video, stride=4)] == list(range(0, 60, 4))
    assert [i for i, _ in iter_frames(two_scene_video, stride=3, start=10, end=20)] == [10, 13, 16, 19]
    with pytest.raises(ValueError):
        next(iter_frames(two_scene_video, stride=0))


def test_probe_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "nope.mp4")


def test_file_sha1_tracks_content(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    first = file_sha1(p)
    assert file_sha1(p) == first
    p.write_bytes(b"hello world")
    assert file_sha1(p) != first
