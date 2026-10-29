"""Pass 2 細掃：只在出手前後的短時間窗內逐幀找球。

球在畫面中很小（常常只有 5–15 像素），所以用較大的輸入尺寸與較低的信心門檻（config.vision.fine_scan），
誤報交給 analysis/ball_track.py 的過濾與軌跡擬合處理。結果依影片內容與參數快取。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import device as device_mod
from ..config import load_config, model_path
from .cache import cache_path
from .video import iter_frames

MAX_PER_FRAME = 5     # 每幀最多保留幾個候選（依信心排序）


@dataclass
class BallDetections:
    """所有時間窗內的候選球，每列一個偵測。"""

    frames: np.ndarray      # (N,) 原始幀號
    xy: np.ndarray          # (N, 2) 框中心
    size: np.ndarray        # (N,) 框邊長（寬高的幾何平均）
    conf: np.ndarray        # (N,)
    fps: float
    frame_size: tuple[int, int]
    windows: list[tuple[int, int]]
    params: dict

    @property
    def times(self) -> np.ndarray:
        return self.frames / self.fps

    def save(self, path: str | Path) -> None:
        meta = {"fps": self.fps, "frame_size": list(self.frame_size), "windows": self.windows, "params": self.params}
        np.savez_compressed(path, frames=self.frames, xy=self.xy, size=self.size, conf=self.conf,
                            meta=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path: str | Path) -> BallDetections:
        d = np.load(path)
        meta = json.loads(str(d["meta"]))
        return cls(d["frames"], d["xy"], d["size"], d["conf"], meta["fps"], tuple(meta["frame_size"]),
                   [tuple(w) for w in meta["windows"]], meta["params"])


def run_ball_scan(video: str | Path, fps: float, windows: list[tuple[int, int]], *, imgsz: int | None = None,
                  conf: float | None = None, device: str | None = None, use_cache: bool = True,
                  progress: Callable[[int, int], None] | None = None) -> tuple[BallDetections, bool]:
    """windows 是 [起始幀, 結束幀) 的清單。回傳 (候選球, 是否命中快取)。"""
    cfg = load_config()
    c = cfg["vision"]["fine_scan"]
    imgsz = imgsz or c["imgsz"]
    conf = conf if conf is not None else c["conf"]
    ball_class = cfg["vision"]["classes"]["sports_ball"]
    windows = [(int(a), int(b)) for a, b in windows]
    params = {"model": model_path("detector").name, "imgsz": imgsz, "conf": conf, "class": ball_class,
              "windows": windows}
    path = cache_path("ball_scan", video, params)
    if use_cache and path.exists():
        return BallDetections.load(path), True

    from .yolo import load_yolo

    model = load_yolo("detector")
    dev = device_mod.resolve_device(device or cfg["device"])
    total = sum(b - a for a, b in windows)
    rows: list[tuple[int, float, float, float, float]] = []
    frame_size = (0, 0)
    done = 0
    for start, end in windows:
        for idx, frame in iter_frames(video, 1, start, end):
            frame_size = (frame.shape[1], frame.shape[0])
            r = model.predict(frame, imgsz=imgsz, conf=conf, classes=[ball_class], device=dev, verbose=False)[0]
            if len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for j in np.argsort(-confs)[:MAX_PER_FRAME]:
                    x1, y1, x2, y2 = xyxy[j]
                    rows.append((idx, (x1 + x2) / 2, (y1 + y2) / 2, float(np.sqrt((x2 - x1) * (y2 - y1))),
                                 float(confs[j])))
            done += 1
            if progress:
                progress(done, total)

    arr = np.array(rows, float).reshape(-1, 5)
    det = BallDetections(arr[:, 0].astype(int), arr[:, 1:3], arr[:, 3], arr[:, 4], fps, frame_size, windows, params)
    det.save(path)
    return det, False
