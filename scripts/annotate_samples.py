"""人工標註動作階段幀號，作為分段演算法的 ground truth。

用法：
    python scripts/annotate_samples.py data/samples/p_side_01.mp4 --subject pitcher --view side
    python scripts/annotate_samples.py data/samples/b_cf_01.mp4 --subject batter --view center_field

標註檔存在 data/samples/labels/<影片檔名>.json；檔案已存在時會載入並接續標註。

按鍵（在影片視窗中按）：
    d / a       下一幀 / 上一幀
    D / A       前進 / 後退 10 幀
    空白鍵       播放 / 暫停
    g           跳到指定幀（在終端機輸入）
    1–6         把目前幀標為目前週期的第 1–6 個階段
    n           新增一個週期並切換過去
    [ / ]       切換到上一個 / 下一個週期
    x           清除目前週期的所有階段
    k           （打者）切換目前週期是否有出棒
    t           設定目前週期的球種（在終端機輸入，例如 FF、SL、CH）
    w           存檔
    q / Esc     存檔並離開
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bbagent.config import load_config, project_path  # noqa: E402

PHASES = {
    "pitcher": ["set", "first_move", "peak_leg_lift", "foot_strike", "release", "follow_through"],
    "batter": ["stance", "load", "foot_down", "swing_start", "contact", "finish"],
}
PHASE_ZH = {
    "set": "固定姿勢", "first_move": "動作啟動", "peak_leg_lift": "抬腿最高點",
    "foot_strike": "前腳著地", "release": "出手", "follow_through": "跟進收尾",
    "stance": "準備站姿", "load": "預備動作", "foot_down": "前腳落地",
    "swing_start": "揮棒啟動", "contact": "擊球點", "finish": "收棒",
}
VIEWS = ["side", "center_field", "other"]
DISPLAY_WIDTH = 1280
WINDOW = "annotate"


class FrameReader:
    """依需求讀幀，連續前進時直接 read，跳躍時才 seek，並快取最近的幀。"""

    def __init__(self, path: Path, cache_size: int = 90):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"無法開啟影片：{path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.next_pos = 0
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cache_size = cache_size

    def get(self, idx: int) -> np.ndarray | None:
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        if idx != self.next_pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        self.next_pos = idx + 1
        if not ok:
            return None
        h, w = frame.shape[:2]
        if w > DISPLAY_WIDTH:
            frame = cv2.resize(frame, (DISPLAY_WIDTH, round(h * DISPLAY_WIDTH / w)), interpolation=cv2.INTER_AREA)
        self.cache[idx] = frame
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return frame


def new_cycle(index: int) -> dict:
    return {"index": index, "phases": {}, "pitch_type": None, "swung": None, "note": ""}


def load_or_init(label_path: Path, video: Path, subject: str, view: str, reader: FrameReader) -> dict:
    if label_path.exists():
        data = json.loads(label_path.read_text(encoding="utf-8"))
        if data["subject"] != subject:
            raise SystemExit(f"既有標註檔的 subject 是 {data['subject']}，與 --subject {subject} 不符")
        print(f"載入既有標註：{label_path}（{len(data['cycles'])} 個週期）")
        return data
    return {
        "video": video.name,
        "subject": subject,
        "camera_view": view,
        "fps": reader.fps,
        "frame_count": reader.count,
        "phase_names": PHASES[subject],
        "cycles": [new_cycle(1)],
        "updated_at": None,
    }


def save(data: dict, label_path: Path) -> None:
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    for c in data["cycles"]:  # 依階段順序排好，方便人工閱讀
        c["phases"] = {p: c["phases"][p] for p in data["phase_names"] if p in c["phases"]}
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已存檔：{label_path}")


def check_order(cycle: dict, phase_names: list[str]) -> str | None:
    marked = [(p, cycle["phases"][p]) for p in phase_names if p in cycle["phases"]]
    for (p1, f1), (p2, f2) in zip(marked, marked[1:]):
        if f2 < f1:
            return f"順序異常：{p2}({f2}) 早於 {p1}({f1})"
    return None


def draw_overlay(frame: np.ndarray, idx: int, data: dict, cur: int, playing: bool, fps: float) -> np.ndarray:
    img = frame.copy()
    cycle = data["cycles"][cur]
    lines = [
        f"frame {idx}/{data['frame_count'] - 1}  t={idx / fps:.3f}s  {'PLAY' if playing else 'PAUSE'}",
        f"cycle {cur + 1}/{len(data['cycles'])}  subject={data['subject']}  view={data['camera_view']}"
        f"  pitch_type={cycle['pitch_type']}  swung={cycle['swung']}",
    ]
    for k, p in enumerate(data["phase_names"], start=1):
        f = cycle["phases"].get(p)
        here = "  <== HERE" if f == idx else ""
        lines.append(f"[{k}] {p:<15} {'-' if f is None else f}{here}")
    warn = check_order(cycle, data["phase_names"])
    if warn:
        lines.append("! " + warn.replace("：", ": "))
    lines.append("a/d:±1 A/D:±10 space:play g:goto n:new [/]:cycle x:clear k:swung t:type w:save q:quit")

    pad, lh = 8, 22
    box_h = pad * 2 + lh * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], box_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    for i, text in enumerate(lines):
        color = (0, 255, 255) if "HERE" in text else (0, 0, 255) if text.startswith("!") else (255, 255, 255)
        cv2.putText(img, text, (pad, pad + lh * (i + 1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    # 底部時間軸：標出目前週期各階段的位置
    bar_y = img.shape[0] - 14
    cv2.line(img, (0, bar_y), (img.shape[1], bar_y), (180, 180, 180), 2)
    scale = img.shape[1] / max(data["frame_count"] - 1, 1)
    for c_i, c in enumerate(data["cycles"]):
        color = (0, 255, 255) if c_i == cur else (120, 120, 120)
        for f in c["phases"].values():
            cv2.circle(img, (round(f * scale), bar_y), 4, color, -1)
    cv2.circle(img, (round(idx * scale), bar_y), 6, (0, 0, 255), 2)
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--subject", choices=list(PHASES), required=True)
    parser.add_argument("--view", choices=VIEWS, default="side")
    args = parser.parse_args()

    reader = FrameReader(args.video)
    label_path = project_path(load_config()["paths"]["labels"]) / f"{args.video.stem}.json"
    data = load_or_init(label_path, args.video, args.subject, args.view, reader)
    phase_names = data["phase_names"]

    idx, cur, playing = 0, 0, False
    last_frame = None
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    delay = max(1, round(1000 / reader.fps))
    while True:
        frame = reader.get(idx)
        if frame is None:  # 超出影片實際長度（幀數中繼資料常有誤差）
            data["frame_count"] = idx
            idx, playing = max(idx - 1, 0), False
            frame = reader.get(idx) if last_frame is None else last_frame
            if frame is None:
                raise SystemExit("影片無法解碼任何幀")
        last_frame = frame
        cv2.imshow(WINDOW, draw_overlay(frame, idx, data, cur, playing, reader.fps))
        key = cv2.waitKey(delay if playing else 0) & 0xFF
        cycle = data["cycles"][cur]

        if key == 255:  # 播放中沒有按鍵
            idx = min(idx + 1, data["frame_count"] - 1)
            if idx == data["frame_count"] - 1:
                playing = False
            continue
        ch = chr(key)
        if ch == "d":
            idx = min(idx + 1, data["frame_count"] - 1)
        elif ch == "a":
            idx = max(idx - 1, 0)
        elif ch == "D":
            idx = min(idx + 10, data["frame_count"] - 1)
        elif ch == "A":
            idx = max(idx - 10, 0)
        elif ch == " ":
            playing = not playing
        elif ch == "g":
            try:
                idx = min(max(int(input("跳到第幾幀：")), 0), data["frame_count"] - 1)
            except ValueError:
                print("請輸入整數")
        elif ch in "123456" and int(ch) <= len(phase_names):
            phase = phase_names[int(ch) - 1]
            cycle["phases"][phase] = idx
            print(f"週期 {cur + 1}：{PHASE_ZH[phase]}（{phase}）= 第 {idx} 幀")
            warn = check_order(cycle, phase_names)
            if warn:
                print(f"  ⚠️ {warn}")
        elif ch == "n":
            data["cycles"].append(new_cycle(len(data["cycles"]) + 1))
            cur = len(data["cycles"]) - 1
        elif ch == "[":
            cur = max(cur - 1, 0)
        elif ch == "]":
            cur = min(cur + 1, len(data["cycles"]) - 1)
        elif ch == "x":
            cycle["phases"].clear()
        elif ch == "k" and data["subject"] == "batter":
            cycle["swung"] = not cycle["swung"] if cycle["swung"] is not None else True
        elif ch == "t":
            value = input("球種（留空表示未知）：").strip()
            cycle["pitch_type"] = value or None
        elif ch == "w":
            save(data, label_path)
        elif ch == "q" or key == 27:
            break

    save(data, label_path)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
