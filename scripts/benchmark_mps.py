"""YOLO 推論基準測試：比較各裝置（mps / cpu …）的速度與結果一致性。

測兩個工作，參數取自 config.yaml：
  - pose：yolov8m-pose，Pass 1 粗掃設定（imgsz 640）
  - det ：yolov8m 偵測，Pass 2 精掃設定（imgsz 1280，只留 person / ball / bat / glove）

以 cpu 的結果為基準，計算其他裝置的框匹配率、平均 IoU 與關鍵點像素誤差，
另外統計每個裝置有幾幀偵測到棒球（評估球軌跡追蹤風險的早期指標）。

用法：
    python scripts/benchmark_mps.py data/samples/p_side_01.mp4
    python scripts/benchmark_mps.py clip.mp4 --frames 60 --start-sec 3 --devices mps,cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bbagent.vision import device as device_mod  # noqa: E402  必須在 torch 之前 import

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from bbagent.config import load_config, project_path  # noqa: E402
from bbagent.vision.imageio import imwrite  # noqa: E402
from bbagent.vision.yolo import load_yolo  # noqa: E402

MAX_WIDTH = 1920          # 超過此寬度的影片先縮小，避免整段讀進記憶體時爆量
IOU_MATCH = 0.5           # IoU 達此值才算同一個物件
KPT_CONF = 0.5            # 兩邊關鍵點信心都超過此值才比較座標
PASS_MATCH_RATE = 0.95    # 一致性判定門檻（經驗值）
PASS_MEAN_IOU = 0.90


@dataclass
class FrameDet:
    boxes: np.ndarray                 # (n, 4) xyxy
    cls: np.ndarray                   # (n,)
    conf: np.ndarray                  # (n,)
    kpts: np.ndarray | None = None    # (n, 17, 2)
    kpt_conf: np.ndarray | None = None  # (n, 17)


@dataclass
class RunResult:
    job: str
    device: str
    latencies_ms: list[float] = field(default_factory=list)
    frames: list[FrameDet] = field(default_factory=list)


def read_frames(video: Path, start_sec: float, n: int) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片：{video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            frame = cv2.resize(frame, (MAX_WIDTH, round(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit("讀不到任何幀，請檢查 --start-sec 是否超過影片長度")
    return frames, fps


def to_frame_det(r) -> FrameDet:
    b = r.boxes
    det = FrameDet(
        boxes=b.xyxy.cpu().numpy(),
        cls=b.cls.cpu().numpy().astype(int),
        conf=b.conf.cpu().numpy(),
    )
    if r.keypoints is not None and len(b):
        det.kpts = r.keypoints.xy.cpu().numpy()
        det.kpt_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
    return det


def run_job(job: str, kind: str, predict_kwargs: dict, dev: str,
            frames: list[np.ndarray], warmup: int, sample_dir: Path) -> RunResult:
    model = load_yolo(kind)
    kwargs = dict(predict_kwargs, device=dev, verbose=False)
    for f in frames[:warmup]:
        model.predict(f, **kwargs)

    result = RunResult(job=job, device=dev)
    best_idx, best_score, best_plot = 0, -1, None
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        r = model.predict(f, **kwargs)[0]
        det = to_frame_det(r)  # .cpu() 會等裝置算完，所以計時包含同步
        result.latencies_ms.append((time.perf_counter() - t0) * 1000)
        result.frames.append(det)
        # 保留一張示意圖：優先選偵測到球的幀，其次選物件最多的幀
        score = int((det.cls == 32).sum()) * 1000 + len(det.cls)
        if score > best_score:
            best_idx, best_score, best_plot = i, score, r.plot()
    imwrite(sample_dir / f"{job}_{dev}_frame{best_idx:04d}.jpg", best_plot)
    del model
    return result


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(ref: FrameDet, other: FrameDet) -> list[tuple[int, int, float]]:
    """同類別、IoU 由大到小貪婪配對。"""
    ious = iou_matrix(ref.boxes, other.boxes)
    ious[ref.cls[:, None] != other.cls[None, :]] = 0
    pairs = []
    used_r, used_o = set(), set()
    for flat in np.argsort(-ious, axis=None):
        i, j = np.unravel_index(flat, ious.shape)
        if ious[i, j] < IOU_MATCH:
            break
        if i in used_r or j in used_o:
            continue
        used_r.add(i)
        used_o.add(j)
        pairs.append((int(i), int(j), float(ious[i, j])))
    return pairs


def compare(ref: RunResult, other: RunResult) -> dict:
    matched = total = 0
    ious, conf_diffs, kpt_errs = [], [], []
    for fr, fo in zip(ref.frames, other.frames):
        pairs = greedy_match(fr, fo)
        matched += len(pairs)
        total += max(len(fr.cls), len(fo.cls))
        for i, j, iou in pairs:
            ious.append(iou)
            conf_diffs.append(abs(float(fr.conf[i] - fo.conf[j])))
            if fr.kpts is not None and fo.kpts is not None and fr.kpt_conf is not None and fo.kpt_conf is not None:
                mask = (fr.kpt_conf[i] > KPT_CONF) & (fo.kpt_conf[j] > KPT_CONF)
                if mask.any():
                    kpt_errs.extend(np.linalg.norm(fr.kpts[i][mask] - fo.kpts[j][mask], axis=1).tolist())
    match_rate = matched / total if total else 1.0
    mean_iou = float(np.mean(ious)) if ious else 1.0
    return {
        "match_rate": round(match_rate, 4),
        "mean_iou": round(mean_iou, 4),
        "mean_conf_diff": round(float(np.mean(conf_diffs)), 4) if conf_diffs else 0.0,
        "mean_kpt_err_px": round(float(np.mean(kpt_errs)), 2) if kpt_errs else None,
        "consistent": match_rate >= PASS_MATCH_RATE and mean_iou >= PASS_MEAN_IOU,
    }


def speed_stats(r: RunResult) -> dict:
    lat = np.array(r.latencies_ms)
    return {
        "frames": len(lat),
        "mean_ms": round(float(lat.mean()), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "fps": round(1000 / float(lat.mean()), 1),
        # 姿勢模型只有 person 類別，不統計球
        "ball_frames": sum(int((f.cls == 32).any()) for f in r.frames) if r.job == "det" else None,
        "mean_objects": round(float(np.mean([len(f.cls) for f in r.frames])), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, default=120, help="測試幀數（預設 120）")
    parser.add_argument("--start-sec", type=float, default=0.0, help="從第幾秒開始讀")
    parser.add_argument("--devices", default=None, help="逗號分隔，例如 mps,cpu；預設為所有可用裝置")
    parser.add_argument("--warmup", type=int, default=5, help="每個裝置的暖機幀數")
    parser.add_argument("--jobs", default="pose,det", help="要測的工作，逗號分隔")
    args = parser.parse_args()

    cfg = load_config()
    vcfg = cfg["vision"]
    classes = vcfg["classes"]
    job_defs = {
        "pose": ("pose", {"imgsz": vcfg["coarse_scan"]["imgsz"], "conf": vcfg["coarse_scan"]["conf"]}),
        "det": ("detector", {
            "imgsz": vcfg["fine_scan"]["imgsz"],
            "conf": vcfg["fine_scan"]["conf"],
            "classes": [classes["person"], classes["sports_ball"], classes["baseball_bat"], classes["baseball_glove"]],
        }),
    }
    jobs = [j.strip() for j in args.jobs.split(",")]
    unknown = set(jobs) - job_defs.keys()
    if unknown:
        raise SystemExit(f"未知的工作：{unknown}")

    available = device_mod.available_devices()
    devices = [d.strip() for d in args.devices.split(",")] if args.devices else available
    for d in devices:
        if d not in available:
            raise SystemExit(f"裝置 {d!r} 不可用；可用裝置：{available}")
    ref_dev = "cpu" if "cpu" in devices else devices[-1]

    frames, fps = read_frames(args.video, args.start_sec, args.frames)
    h, w = frames[0].shape[:2]
    out_dir = project_path(cfg["paths"]["runs"]) / "benchmark" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"影片 {args.video.name}：讀入 {len(frames)} 幀（{w}x{h}，{fps:.2f} fps）｜裝置 {devices}｜基準 {ref_dev}")

    report: dict = {
        "video": str(args.video), "resolution": [w, h], "fps": fps, "frames": len(frames),
        "start_sec": args.start_sec, "devices": devices, "reference": ref_dev, "jobs": {},
    }
    for job in jobs:
        kind, kwargs = job_defs[job]
        runs: dict[str, RunResult] = {}
        for dev in devices:
            print(f"[{job}] {dev}：執行中…", flush=True)
            runs[dev] = run_job(job, kind, kwargs, dev, frames, args.warmup, out_dir)
            s = speed_stats(runs[dev])
            ball = "" if s["ball_frames"] is None else f"，有球的幀 {s['ball_frames']}/{s['frames']}"
            print(f"[{job}] {dev}：{s['fps']} fps（平均 {s['mean_ms']} ms，p95 {s['p95_ms']} ms）{ball}")
        report["jobs"][job] = {
            "params": {k: v for k, v in kwargs.items()},
            "speed": {dev: speed_stats(r) for dev, r in runs.items()},
            "consistency": {dev: compare(runs[ref_dev], r) for dev, r in runs.items() if dev != ref_dev},
        }

    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_markdown(report)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"報告與示意圖已輸出到 {out_dir}")
    all_ok = all(c["consistent"] for j in report["jobs"].values() for c in j["consistency"].values())
    return 0 if all_ok else 1


def render_markdown(report: dict) -> str:
    lines = [
        "# YOLO 基準測試報告",
        "",
        f"- 影片：`{report['video']}`（{report['resolution'][0]}x{report['resolution'][1]}，"
        f"{report['fps']:.2f} fps，測 {report['frames']} 幀，從 {report['start_sec']} 秒開始）",
        f"- 一致性基準裝置：{report['reference']}",
        "",
    ]
    for job, data in report["jobs"].items():
        lines += [f"## {job}　`{data['params']}`", "",
                  "| 裝置 | FPS | 平均 ms | p95 ms | 平均物件數 | 有球的幀 |",
                  "|---|---|---|---|---|---|"]
        for dev, s in data["speed"].items():
            ball = "—" if s["ball_frames"] is None else f"{s['ball_frames']}/{s['frames']}"
            lines.append(f"| {dev} | {s['fps']} | {s['mean_ms']} | {s['p95_ms']} | {s['mean_objects']} | {ball} |")
        if data["consistency"]:
            lines += ["", f"與 {report['reference']} 的一致性：", "",
                      "| 裝置 | 框匹配率 | 平均 IoU | 信心差 | 關鍵點誤差 px | 判定 |",
                      "|---|---|---|---|---|---|"]
            for dev, c in data["consistency"].items():
                kpt = "—" if c["mean_kpt_err_px"] is None else c["mean_kpt_err_px"]
                verdict = "✅ 一致" if c["consistent"] else "❌ 不一致"
                lines.append(f"| {dev} | {c['match_rate']:.1%} | {c['mean_iou']:.3f} | "
                             f"{c['mean_conf_diff']:.3f} | {kpt} | {verdict} |")
        lines.append("")
    if len(report["devices"]) < 2:
        lines.append("⚠️ 只測了一個裝置，沒有做一致性比較。")
    else:
        lines.append(f"判定門檻：框匹配率 ≥ {PASS_MATCH_RATE:.0%} 且平均 IoU ≥ {PASS_MEAN_IOU}（經驗值）。")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
