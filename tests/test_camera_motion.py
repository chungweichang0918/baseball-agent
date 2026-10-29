import cv2
import numpy as np
import pytest

from bbagent.vision.camera_motion import estimate_camera_motion

W, H, FPS, N = 320, 240, 30, 60


def _texture(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    big = rng.integers(0, 255, (H + 200, W + 300, 3), dtype=np.uint8)
    return cv2.GaussianBlur(big, (0, 0), 1.5)


def _write(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()
    return path


PERSON = (100, 40, 220, 230)   # 畫面中央的「人」：x0, y0, x1, y1


def _with_person(bg: np.ndarray, i: int, rng) -> np.ndarray:
    """在固定的人物框內畫一塊每幀都在動的紋理，模擬投手的動作。"""
    f = bg.copy()
    x0, y0, x1, y1 = PERSON
    f[y0:y1, x0:x1] = np.roll(_texture(i)[:y1 - y0, :x1 - x0], shift=i * 7, axis=1)
    return f


def test_static_camera(tmp_path):
    bg = _texture()[:H, :W]
    rng = np.random.default_rng(1)
    path = _write(tmp_path / "static.avi", [_with_person(bg, i, rng) for i in range(N)])
    m = estimate_camera_motion(path, FPS, boxes_at=lambda idx: [np.array(PERSON, float)])
    assert m.valid_pairs >= 3 and m.moving is False, m.to_dict()


def test_panning_camera(tmp_path):
    big = _texture()
    rng = np.random.default_rng(1)
    frames = [_with_person(big[0:H, 4 * i:4 * i + W].copy(), i, rng) for i in range(N)]   # 每幀平移 4 px
    path = _write(tmp_path / "pan.avi", frames)
    m = estimate_camera_motion(path, FPS, boxes_at=lambda idx: [np.array(PERSON, float)])
    assert m.moving is True, m.to_dict()
    # 4 px/幀 × 30 fps = 120 px/s；畫面縮成寬 480 後放大 1.5 倍 → 180 px/s ÷ 對角線 600 = 0.30
    assert m.p80_shift == pytest.approx(0.30, rel=0.15)


def test_zooming_camera(tmp_path):
    base = _texture()[:H, :W]
    frames = []
    for i in range(N):
        s = 1 + 0.004 * i   # 每幀放大 0.4% → 每秒約 12%
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
        frames.append(cv2.warpAffine(base, M, (W, H), borderMode=cv2.BORDER_REFLECT))
    m = estimate_camera_motion(_write(tmp_path / "zoom.avi", frames), FPS)
    assert m.moving is True and m.p80_scale > 0.05, m.to_dict()


def test_featureless_background_is_undetermined(tmp_path):
    frames = [np.full((H, W, 3), 128, np.uint8) for _ in range(N)]
    m = estimate_camera_motion(_write(tmp_path / "flat.avi", frames), FPS)
    assert m.moving is None and m.valid_pairs == 0
