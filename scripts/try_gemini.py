"""單獨試跑 Gemini 工具，確認金鑰、型號與提示詞都正常。

用法：
    python scripts/try_gemini.py overview data/samples/p_other_01.mp4
    python scripts/try_gemini.py inspect data/samples/p_other_01.mp4 --frames 115,139,144 --focus "出手時投球手臂的角度"
    python scripts/try_gemini.py inspect runs/analyze/p_other_01/skeleton_sheet_1.jpg --focus "抬腿高度"
    python scripts/try_gemini.py stream "用三句話說明什麼是 RE24"

加上 --no-cache 會略過快取、一定呼叫 API（會消耗額度）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bbagent.llm.gemini_client import GeminiClient, GeminiError, model_name  # noqa: E402
from bbagent.schemas import ToolResult  # noqa: E402
from bbagent.tools.gemini_inspect import gemini_inspect  # noqa: E402
from bbagent.tools.scout_overview import scout_overview  # noqa: E402
from bbagent.vision.render import read_frames  # noqa: E402
from bbagent.vision.video import probe_video  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi"}


def show(result: ToolResult) -> int:
    print(f"\n狀態：{result.status}")
    print(result.summary)
    if result.error:
        print(f"錯誤：{result.error}")
    llm = result.data.get("llm")
    if llm:
        src = "快取重播" if llm["cached"] else f"呼叫 API，{llm['latency_s']} 秒"
        print(f"\n型號 {llm['model']}｜{src}｜用量 {llm['usage'] or '—'}")
    return 0 if result.status == "ok" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ov = sub.add_parser("overview", help="判斷影片視角、主體與拍攝條件")
    p_ov.add_argument("video", type=Path)
    p_in = sub.add_parser("inspect", help="看關鍵幀回答觀察重點")
    p_in.add_argument("source", type=Path, nargs="+", help="一支影片（搭配 --frames）或多張圖片")
    p_in.add_argument("--frames", help="影片幀號，逗號分隔")
    p_in.add_argument("--focus", required=True, help="觀察重點")
    p_in.add_argument("--context", default="", help="背景資訊（賽況、已知數據）")
    p_st = sub.add_parser("stream", help="測試串流輸出")
    p_st.add_argument("prompt")
    for p in (p_ov, p_in, p_st):
        p.add_argument("--no-cache", action="store_true", help="略過快取，一定呼叫 API")
        p.add_argument("--model", help="覆寫 config 中的型號")
        p.add_argument("--thinking", choices=["minimal", "low", "medium", "high"],
                       help="模型思考程度（只適用 Gemini 3 系列），越低越快")
    args = parser.parse_args()

    client = GeminiClient(
        cache_mode="off" if args.no_cache else None,
        thinking_level=args.thinking,
        on_retry=lambda n, wait, reason: print(f"  ⏳ 第 {n} 次重試：{reason}，{wait:g} 秒後再試", flush=True),
    )
    print(f"呼叫 Gemini 中（單次逾時 {client.timeout_s:g} 秒，最多重試 {client.max_retries} 次，"
          f"思考程度 {client.thinking_level or '模型預設'}）…", flush=True)
    if args.cmd == "overview":
        return show(scout_overview(args.video, client=client, model=args.model))

    if args.cmd == "inspect":
        first = args.source[0]
        if first.suffix.lower() in VIDEO_EXTS:
            if not args.frames:
                parser.error("輸入影片時必須指定 --frames")
            meta = probe_video(first)
            idx = [int(x) for x in args.frames.split(",")]
            frames = read_frames(first, idx)
            missing = [i for i in idx if i not in frames]
            if missing:
                parser.error(f"讀不到這些幀：{missing}（影片共 {meta.frame_count} 幀）")
            images = [frames[i] for i in idx]
            captions = [f"第 {i} 幀（{i / meta.fps:.2f}s）" for i in idx]
        else:
            images, captions = list(args.source), [p.name for p in args.source]
        return show(gemini_inspect(images, args.focus, captions, args.context, client=client, model=args.model))

    model = args.model or model_name("fast")
    try:
        for chunk in client.stream(model, [args.prompt]):
            print(chunk, end="", flush=True)
    except GeminiError as e:
        print(f"\n錯誤：{e}")
        return 1
    r = client.last_response
    timing = "快取重播" if r.cached else f"呼叫 API，第一段文字 {r.first_chunk_s} 秒、全部完成 {r.latency_s} 秒"
    print(f"\n\n型號 {r.model}｜{timing}｜用量 {r.usage or '—'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
