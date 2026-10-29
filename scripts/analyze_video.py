"""分析一段投手影片：粗掃骨架 → 選主體 → 動作分段 → 指標，並與人工標註比對。

用法：
    python scripts/analyze_video.py data/samples/p_side_01.mp4
    python scripts/analyze_video.py clip.mp4 --view center_field --no-cache
    python scripts/analyze_video.py clip.mp4 --track-id 3     # 主體選錯時手動指定
    python scripts/analyze_video.py clip.mp4 --hand R         # 指定投球手（側拍時骨架左右可能標反）

data/samples/labels/ 中有同名標註檔時會自動比對。
輸出：終端機表格，以及 runs/analyze/<影片檔名>/ 下的 result.json、signals.png、
每一球出手前後的骨架連拍圖 skeleton_sheet_<n>.jpg。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bbagent.vision import device  # noqa: E402,F401  必須在 torch 之前 import

import numpy as np  # noqa: E402

from bbagent.analysis.evaluation import DEFAULT_TOLERANCE, compare_with_labels  # noqa: E402
from bbagent.analysis.keypoints import SubjectSeries, build_series  # noqa: E402
from bbagent.analysis.pitching_metrics import METRIC_INFO, compute_pitching_metrics  # noqa: E402
from bbagent.analysis.plots import plot_motion_curves  # noqa: E402
from bbagent.analysis.segmentation import PITCH_PHASES, PitchSegmentation, segment_pitcher  # noqa: E402
from bbagent.analysis.subject import select_subject  # noqa: E402
from bbagent.config import load_config, project_path  # noqa: E402
from bbagent.vision.pose_scan import PoseScan, merge_fragments, run_pose_scan  # noqa: E402
from bbagent.vision.imageio import imwrite  # noqa: E402
from bbagent.vision.render import caption, contact_sheet, draw_skeleton, read_frames  # noqa: E402
from bbagent.vision.video import probe_video  # noqa: E402

import cv2  # noqa: E402

SHEET_BEFORE_S = 0.8    # 連拍圖從出手前多久開始
SHEET_AFTER_S = 0.3     # 到出手後多久結束
SHEET_MAX_TILES = 24

PHASE_ZH = {"set": "固定姿勢", "first_move": "動作啟動", "peak_leg_lift": "抬腿最高點",
            "foot_strike": "前腳著地", "release": "出手", "follow_through": "跟進收尾"}


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title) * 2))


def fmt(v, unit: str = "") -> str:
    return "—" if v is None else f"{v}{unit}"


def print_segmentation(seg: PitchSegmentation, lr_fixes: int) -> None:
    source = "手動指定" if seg.side_source == "user" else "依骨架判斷"
    print(f"投球手：{'右' if seg.throwing_side == 'R' else '左'}手（{source}）｜偵測到 {len(seg.cycles)} 球"
          f"｜修正左右標反 {lr_fixes} 幀")
    for w in seg.warnings:
        print(f"  ⚠️ {w}")
    if not seg.cycles:
        return
    print("階段".ljust(7) + "".join(f"第 {c.index} 球".rjust(18) for c in seg.cycles))
    for ph in PITCH_PHASES:
        cells = [(f"幀 {c.phases[ph]}（{c.times[ph]:.2f}s）" if c.phases[ph] is not None else "—").rjust(18)
                 for c in seg.cycles]
        print(PHASE_ZH[ph].ljust(7) + "".join(cells))
    for c in seg.cycles:
        for w in c.warnings:
            print(f"  ⚠️ 第 {c.index} 球：{w}")


def print_metrics(metrics: dict) -> None:
    for key, s in metrics["summary"].items():
        per = "、".join(fmt(p[key]) for p in metrics["per_pitch"])
        spread = f"，標準差 {s['std']}" if s["std"] is not None else ""
        print(f"{s['name']}：平均 {s['mean']} {s['unit']}{spread}（各球：{per}）")
    missing = [METRIC_INFO[k][0] for k in METRIC_INFO if k not in metrics["summary"]]
    if missing:
        print(f"未能計算：{'、'.join(missing)}")
    for note in metrics["notes"]:
        print(f"  ※ {note}")


def print_comparison(cmp: dict) -> None:
    print(f"容許誤差：平均 ≤ {cmp['tolerance_frames']} 幀（誤差 = 偵測 − 標註，正值表示偵測太晚）")
    for ph, v in cmp["per_phase"].items():
        mark = "✅" if v["pass"] else "❌"
        detail = (f"平均誤差 {v['mean_error']:+}，平均絕對誤差 {v['mean_abs_error']}，最大 {v['max_abs_error']}"
                  if v["n"] else "全部未偵測到")
        miss = f"，{v['missing']} 球未偵測到" if v["missing"] else ""
        print(f"  {mark} {PHASE_ZH[ph]}：{detail}（{v['n']} 球{miss}）")
    if cmp["unmatched_labels"]:
        print(f"  ❌ 標註的第 {cmp['unmatched_labels']} 球沒有對應的偵測結果")
    if cmp["extra_detected"]:
        print(f"  ❌ 偵測到的第 {cmp['extra_detected']} 球在標註中不存在（可能是誤判）")
    print("整體：" + ("✅ 通過" if cmp["pass"] else "❌ 未通過"))


def save_skeleton_sheets(video: Path, scan: PoseScan, track_id: int, series: SubjectSeries,
                         seg: PitchSegmentation, out_dir: Path) -> list[Path]:
    """每一球輸出一張出手前後的連拍圖，畫的是 YOLO 的原始骨架（未修正左右），用來檢查標記。

    沒有偵測到投球時，改以手腕速度最快的時刻為中心輸出一張。
    """
    track = scan.tracks[track_id]
    centers = [c.samples["release"] for c in seg.cycles]
    if not centers and np.isfinite(series.max_speed("wrist")).any():
        centers = [int(np.nanargmax(series.max_speed("wrist")))]
    phase_at = {c.phases[ph]: ph for c in seg.cycles for ph in PITCH_PHASES if c.phases[ph] is not None}
    paths = []
    for k, r in enumerate(centers, start=1):
        lo = max(0, r - series.seconds_to_samples(SHEET_BEFORE_S))
        hi = min(len(series.times) - 1, r + series.seconds_to_samples(SHEET_AFTER_S))
        samples = np.unique(np.linspace(lo, hi, min(SHEET_MAX_TILES, hi - lo + 1)).round().astype(int))
        frames = read_frames(video, [int(scan.frame_indices[i]) for i in samples])
        tiles = []
        for i in samples:
            f = int(scan.frame_indices[i])
            if f not in frames:
                continue
            img = draw_skeleton(frames[f].copy(), track.kpts[i], track.kconf[i])
            tag = f" {phase_at[f]}" if f in phase_at else ""
            tiles.append(caption(img, f"#{f} {scan.times[i]:.2f}s{tag}"))
        if tiles:
            path = out_dir / f"skeleton_sheet_{k}.jpg"
            imwrite(path, contact_sheet(tiles))
            paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--subject", choices=["pitcher", "batter"], default="pitcher")
    parser.add_argument("--view", choices=["side", "center_field", "other"], default="side")
    parser.add_argument("--no-cache", action="store_true", help="忽略快取，重新跑 YOLO")
    parser.add_argument("--track-id", type=int, help="手動指定主體的追蹤編號")
    parser.add_argument("--hand", choices=["R", "L"], help="投手的投球手（側拍時骨架左右可能標反，建議指定）")
    parser.add_argument("--labels", type=Path, help="標註檔路徑（預設自動尋找同名檔）")
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE, help="容許的平均誤差（幀）")
    args = parser.parse_args()

    if args.subject == "batter":
        print("這個校正工具只處理投手；打者請用 scripts/run_agent.py --target batter（動作曲線圖在 runs/agent/ 底下）。")
        return 2

    cfg = load_config()
    section("影片")
    meta = probe_video(args.video)
    print(f"{args.video.name}：{meta.width}x{meta.height}，{meta.fps:.2f} fps，{meta.frame_count} 幀（{meta.duration_s:.2f} 秒）")
    if meta.scene_cuts:
        print(f"  ⚠️ 偵測到場景切換，位於第 {meta.scene_cuts} 幀。影片可能包含特寫或重播，建議只保留現場即時畫面。")

    section("Pass 1 粗掃")
    t0 = time.perf_counter()

    def progress(done: int, total: int) -> None:
        print(f"\r  姿勢模型：{done}/{total} 幀", end="", flush=True)

    scan, hit = run_pose_scan(args.video, meta.fps, meta.frame_count, use_cache=not args.no_cache, progress=progress)
    elapsed = time.perf_counter() - t0
    print(("\r  命中快取" if hit else "") + f"，耗時 {elapsed:.1f} 秒｜取樣 {len(scan.frame_indices)} 幀"
          f"（每 {scan.params['stride']} 幀取 1）｜追蹤到 {len(scan.tracks)} 個人")
    scan = merge_fragments(scan)
    print(f"  合併斷開的追蹤後剩 {len(scan.tracks)} 個人")

    section("主體識別")
    choice = select_subject(scan, args.subject)
    for c in choice.candidates[:5]:
        mark = "→" if c.track_id == choice.track_id else " "
        print(f"  {mark} 編號 {c.track_id}：出現 {c.coverage:.0%}，大小 {c.size:.2f}，手腕活動量 {c.activity:.2f}，分數 {c.score}")
    track_id = args.track_id if args.track_id is not None else choice.track_id
    if track_id is None or track_id not in scan.tracks:
        print(f"  ❌ 找不到可分析的主體（{choice.reason}）")
        return 1
    print(f"  選定編號 {track_id}" + ("（手動指定）" if args.track_id is not None else f"：{choice.reason}"))
    if not choice.confident and args.track_id is None:
        print("  ⚠️ 主體識別信心不足，請看 signals.png 或用 --track-id 手動指定")

    series = build_series(scan.tracks[track_id], scan.times, scan.frame_indices)
    seg = segment_pitcher(series, throwing_side=args.hand)
    metrics = compute_pitching_metrics(series, seg, args.view)

    section("動作分段")
    print_segmentation(seg, series.lr_fixes)
    section("投手指標")
    print_metrics(metrics)

    label_path = args.labels or project_path(cfg["paths"]["labels"]) / f"{args.video.stem}.json"
    labels = json.loads(label_path.read_text(encoding="utf-8")) if label_path.exists() else None
    comparison = None
    section("與人工標註比對")
    if labels is None:
        print(f"沒有標註檔（{label_path}），略過。")
    elif labels.get("subject") != args.subject:
        print(f"標註檔的主體是 {labels.get('subject')}，與 --subject {args.subject} 不同，略過。")
        labels = None
    else:
        comparison = compare_with_labels(seg.cycles, labels, args.tolerance)
        print_comparison(comparison)

    out_dir = project_path(cfg["paths"]["runs"]) / "analyze" / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_motion_curves(series, seg, out_dir / "signals.png", labels=labels, fps=meta.fps)
    sheets = save_skeleton_sheets(args.video, scan, track_id, series, seg, out_dir)
    result = {
        "video": meta.to_dict(),
        "pose_scan": {"params": scan.params, "cache_hit": hit, "seconds": round(elapsed, 2)},
        "subject": choice.to_dict() | {"used_track_id": track_id},
        "segmentation": seg.to_dict(),
        "metrics": metrics,
        "comparison": comparison,
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
                                         encoding="utf-8")
    print(f"\n輸出到 {out_dir}：result.json、signals.png" + "".join(f"、{p.name}" for p in sheets))
    if sheets:
        print("  連拍圖中藍色 = 骨架標記的左側、橘色 = 右側，可檢查 YOLO 是否把左右標反")
    return 0


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"無法序列化 {type(o)}")


if __name__ == "__main__":
    sys.exit(main())
