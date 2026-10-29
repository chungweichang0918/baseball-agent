"""Pass 1 粗掃：YOLO pose + ByteTrack，輸出每個追蹤對象的骨架時間序列。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import device as device_mod
from ..config import load_config, model_path
from .cache import cache_path
from .video import iter_frames

N_KPT = 17
TRACKER = "bytetrack.yaml"


@dataclass
class PoseTrack:
    """單一追蹤對象。所有陣列的第一維對應 PoseScan.frame_indices；不在畫面中的幀為 NaN。"""

    track_id: int
    boxes: np.ndarray   # (N, 4) xyxy
    conf: np.ndarray    # (N,)
    kpts: np.ndarray    # (N, 17, 2)
    kconf: np.ndarray   # (N, 17)

    @classmethod
    def empty(cls, track_id: int, n: int) -> PoseTrack:
        return cls(track_id, np.full((n, 4), np.nan), np.full(n, np.nan),
                   np.full((n, N_KPT, 2), np.nan), np.full((n, N_KPT), np.nan))

    @property
    def present(self) -> np.ndarray:
        return ~np.isnan(self.conf)

    @property
    def coverage(self) -> float:
        return float(self.present.mean())

    @property
    def heights(self) -> np.ndarray:
        return self.boxes[:, 3] - self.boxes[:, 1]

    @property
    def centers(self) -> np.ndarray:
        return np.stack([(self.boxes[:, 0] + self.boxes[:, 2]) / 2, (self.boxes[:, 1] + self.boxes[:, 3]) / 2], axis=1)

    def span(self) -> tuple[int, int]:
        idx = np.flatnonzero(self.present)
        return int(idx[0]), int(idx[-1])


@dataclass
class PoseScan:
    frame_indices: np.ndarray        # (N,) 取樣到的原始幀號
    fps: float                       # 原始影片 fps
    frame_size: tuple[int, int]      # (寬, 高)
    tracks: dict[int, PoseTrack]
    params: dict

    @property
    def times(self) -> np.ndarray:
        return self.frame_indices / self.fps

    def save(self, path: str | Path) -> None:
        ids = sorted(self.tracks)
        n = len(self.frame_indices)

        def stack(attr: str, shape: tuple) -> np.ndarray:
            if not ids:
                return np.zeros((0, n, *shape))
            return np.stack([getattr(self.tracks[i], attr) for i in ids])

        np.savez_compressed(
            path,
            frame_indices=self.frame_indices,
            track_ids=np.array(ids, dtype=int),
            boxes=stack("boxes", (4,)),
            conf=stack("conf", ()),
            kpts=stack("kpts", (N_KPT, 2)),
            kconf=stack("kconf", (N_KPT,)),
            meta=np.array(json.dumps({"fps": self.fps, "frame_size": self.frame_size, "params": self.params})),
        )

    @classmethod
    def load(cls, path: str | Path) -> PoseScan:
        d = np.load(path)
        meta = json.loads(str(d["meta"]))
        tracks = {
            int(tid): PoseTrack(int(tid), d["boxes"][k], d["conf"][k], d["kpts"][k], d["kconf"][k])
            for k, tid in enumerate(d["track_ids"])
        }
        return cls(d["frame_indices"], meta["fps"], tuple(meta["frame_size"]), tracks, meta["params"])


def run_pose_scan(video: str | Path, fps: float, total_frames: int | None = None, *,
                  stride: int | None = None, imgsz: int | None = None, conf: float | None = None,
                  device: str | None = None, use_cache: bool = True,
                  progress: Callable[[int, int], None] | None = None) -> tuple[PoseScan, bool]:
    """回傳 (粗掃結果, 是否命中快取)。參數未指定時取 config.vision.coarse_scan。

    stride 預設依影片 fps 決定，讓取樣後約為 target_fps。投球出手只有約 0.1 秒，
    30 fps 影片若再隔幀取樣只剩 15 Hz，手腕速度峰值會被抹平。
    """
    cfg = load_config()
    c = cfg["vision"]["coarse_scan"]
    stride = stride or max(1, round(fps / c["target_fps"]))
    imgsz = imgsz or c["imgsz"]
    conf = conf if conf is not None else c["conf"]
    params = {"model": model_path("pose").name, "stride": stride, "imgsz": imgsz, "conf": conf, "tracker": TRACKER}
    path = cache_path("pose_scan", video, params)
    if use_cache and path.exists():
        return PoseScan.load(path), True

    from .yolo import load_yolo

    model = load_yolo("pose")
    dev = device_mod.resolve_device(device or cfg["device"])
    expected = -(-total_frames // stride) if total_frames else 0

    frame_indices: list[int] = []
    detections: dict[int, list[tuple[int, np.ndarray, float, np.ndarray, np.ndarray]]] = {}
    frame_size = (0, 0)
    for i, (idx, frame) in enumerate(iter_frames(video, stride)):
        if i == 0:
            frame_size = (frame.shape[1], frame.shape[0])
        r = model.track(frame, persist=True, tracker=TRACKER, imgsz=imgsz, conf=conf, device=dev, verbose=False)[0]
        frame_indices.append(idx)
        b = r.boxes
        if b.id is not None and r.keypoints is not None:
            ids = b.id.int().cpu().numpy()
            xyxy = b.xyxy.cpu().numpy()
            confs = b.conf.cpu().numpy()
            kxy = r.keypoints.xy.cpu().numpy()
            kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2])
            for j, tid in enumerate(ids):
                detections.setdefault(int(tid), []).append((i, xyxy[j], float(confs[j]), kxy[j], kc[j]))
        if progress:
            progress(i + 1, expected)

    n = len(frame_indices)
    tracks = {}
    for tid, rows in detections.items():
        t = PoseTrack.empty(tid, n)
        for i, box, cf, kxy, kc in rows:
            t.boxes[i], t.conf[i], t.kpts[i], t.kconf[i] = box, cf, kxy, kc
        tracks[tid] = t
    scan = PoseScan(np.array(frame_indices, dtype=int), fps, frame_size, tracks, params)
    scan.save(path)
    return scan, False


def merge_fragments(scan: PoseScan, max_gap_s: float = 0.5, max_dist: float = 0.5) -> PoseScan:
    """把被追蹤器切斷的同一個人接回來。

    條件：A 消失後 max_gap_s 秒內 B 才出現（兩者沒有同時在畫面中），
    且 B 出現的位置離 A 消失的位置不超過 max_dist 個身高。
    """
    tracks = {  # 複製陣列，合併時才不會改到原本的 scan
        k: PoseTrack(t.track_id, t.boxes.copy(), t.conf.copy(), t.kpts.copy(), t.kconf.copy())
        for k, t in scan.tracks.items() if t.present.any()
    }
    times = scan.times
    merged = True
    while merged:
        merged = False
        order = sorted(tracks, key=lambda k: tracks[k].span()[0])
        for b_id in order:
            b = tracks[b_id]
            b_start = b.span()[0]
            for a_id in order:
                if a_id == b_id:
                    continue
                a = tracks[a_id]
                a_end = a.span()[1]
                if a_end >= b_start or times[b_start] - times[a_end] > max_gap_s:
                    continue
                dist = np.linalg.norm(a.centers[a_end] - b.centers[b_start])
                if dist > max_dist * np.nanmedian(a.heights):
                    continue
                mask = b.present
                for attr in ("boxes", "conf", "kpts", "kconf"):
                    getattr(a, attr)[mask] = getattr(b, attr)[mask]
                del tracks[b_id]
                merged = True
                break
            if merged:
                break
    return replace(scan, tracks=tracks)
