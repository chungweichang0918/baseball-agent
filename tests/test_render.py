import numpy as np
import pytest
from synthetic import make_pitcher

from bbagent.vision.render import COLOR_L, COLOR_R, caption, contact_sheet, draw_skeleton


def test_draw_skeleton_colors_left_and_right():
    track, *_ = make_pitcher(n_pitches=1, noise_px=0)
    img = draw_skeleton(np.zeros((800, 1000, 3), np.uint8), track.kpts[0], track.kconf[0])
    lx, ly = track.kpts[0, 13].round().astype(int)    # 左膝
    rx, ry = track.kpts[0, 14].round().astype(int)    # 右膝
    assert tuple(img[ly, lx]) == COLOR_L
    assert tuple(img[ry, rx]) == COLOR_R


def test_low_confidence_points_are_not_drawn():
    track, *_ = make_pitcher(n_pitches=1, noise_px=0)
    conf = np.zeros(17)
    img = draw_skeleton(np.zeros((800, 1000, 3), np.uint8), track.kpts[0], conf)
    assert img.sum() == 0


def test_contact_sheet_layout():
    tiles = [caption(np.full((1280, 720, 3), i * 20, np.uint8), f"#{i}") for i in range(8)]
    sheet = contact_sheet(tiles, cols=3, tile_width=180)
    assert sheet.shape == (3 * 320, 3 * 180, 3)   # 8 張排成 3 欄 → 3 列；720x1280 縮成寬 180 → 高 320
    with pytest.raises(ValueError):
        contact_sheet([])


def test_imageio_handles_non_ascii_paths(tmp_path):
    # Windows 上 cv2.imread／imwrite 遇到中文路徑會靜靜失敗（專案就放在「生成式AI作業」底下）
    from bbagent.vision.imageio import imread, imwrite

    path = tmp_path / "生成式AI作業" / "關鍵幀.jpg"
    path.parent.mkdir()
    img = np.full((20, 30, 3), 128, np.uint8)
    assert imwrite(path, img) and path.exists()
    back = imread(path)
    assert back.shape == img.shape and abs(int(back.mean()) - 128) <= 2
    assert imread(tmp_path / "沒有這個檔案.jpg") is None
