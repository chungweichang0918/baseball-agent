"""Gradio 介面：上傳影片、填入賽況，即時顯示 Agent 的規劃、工具呼叫、關鍵幀、動作曲線、報告與審查過程。

介面只消費 run_agent 產生的 AgentEvent（經 RunView 整理），不碰 Agent 內部狀態。
同時支援 Gradio 5 與 6：兩版參數不同的地方用 _supported() 過濾。
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import gradio as gr

from ..agent.orchestrator import run_agent
from ..config import gemini_api_key, load_config, project_path
from ..llm.gemini_client import GeminiClient
from ..schemas import GameState
from ..vision.render import read_frames
from ..vision.video import probe_video
from .diagram import situation_svg
from .view import RunView

DELTA_INTERVAL_S = 0.15     # 報告串流時更新畫面的最短間隔，避免每個字都重繪
CSS = Path(__file__).with_name("studio.css").read_text(encoding="utf-8")
CURVES_HELP = """動作能量、手腕速度、膝蓋高度、腳踝速度（以身高為單位）。直線是偵測到的動作階段：

- 投手：灰 = 固定姿勢、藍 = 動作啟動、綠 = 抬腿最高點、橘 = 前腳著地、紅 = 出手、紫 = 跟進收尾
- 打者：灰 = 準備姿勢、藍 = 引棒、橘 = 跨步著地、紅 = 擊球、紫 = 收棒"""
HEADER = """# ⚾ 棒球戰術 Agent
上傳 10–15 秒的投球或打擊影片並填入賽況。Agent 會先**規劃**要用哪些工具，再**執行** YOLO 骨架分析與 Gemini 看圖，
結合 RE24 得分期望值**撰寫**戰術報告，最後由 **Critic 審查**，不合格就退回修正。"""

OUTS = [("無人出局", 0), ("一出局", 1), ("兩出局", 2)]
BASES = [("一壘", "1"), ("二壘", "2"), ("三壘", "3")]


def _supported(fn, **kwargs) -> dict:
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def make_game(inning, half, outs, bases, balls, strikes, score_diff, role, target, hand, batter_hand,
              notes) -> GameState:
    bases = set(bases or [])
    return GameState(inning=int(inning), half=half, outs=int(outs), on_first="1" in bases, on_second="2" in bases,
                     on_third="3" in bases, balls=int(balls), strikes=int(strikes), score_diff=int(score_diff or 0),
                     our_role=role, analysis_target=target or "auto", pitcher_hand=hand or None,
                     batter_hand=batter_hand or None, notes=notes or "")


def preview_situation(*game_inputs) -> str:
    try:
        return situation_svg(make_game(*game_inputs))
    except ValueError as e:
        return f"<p>賽況有誤：{e}</p>"


def preview_video(path: str | None) -> tuple:
    """上傳後顯示影片中段的一張畫面與基本資訊；讀不到就提早告訴使用者。"""
    if not path:
        return gr.update(value=None, visible=False), ""
    try:
        meta = probe_video(path)
    except (FileNotFoundError, ValueError) as e:
        return gr.update(value=None, visible=False), f"❌ 讀不到這個影片：{e}"
    frame = read_frames(path, [meta.frame_count // 2]).get(meta.frame_count // 2)
    info = f"{meta.width}×{meta.height}｜{meta.fps:.0f} fps｜{meta.duration_s:.1f} 秒"
    if meta.duration_s > 20:
        info += "\n\n⚠️ 影片超過 20 秒，分析會比較久；建議剪成 10–15 秒、包含 2–3 球"
    if meta.scene_cuts:
        info += "\n\n⚠️ 偵測到場景切換，可能含特寫或重播"
    image = None if frame is None else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return gr.update(value=image, visible=image is not None), info


def _outputs(view: RunView, tab=None) -> tuple:
    return (f"**{view.status}**", view.plan_md(), list(view.messages), view.keyframes or None, view.curves,
            view.report_md(), view.review_md(), view.files() or None, tab if tab is not None else gr.update())


def analyze(video, inning, half, outs, bases, balls, strikes, score_diff, role, target, hand, batter_hand, notes,
            demo_fault, no_cache) -> Iterator[tuple]:
    if not video:
        raise gr.Error("請先上傳影片")
    try:
        game = make_game(inning, half, outs, bases, balls, strikes, score_diff, role, target, hand, batter_hand,
                         notes)
    except ValueError as e:
        raise gr.Error(f"賽況有誤：{e}") from e

    view = RunView()
    client = GeminiClient(cache_mode="off" if no_cache else None)
    yield _outputs(view, gr.Tabs(selected="process"))
    last, on_report = 0.0, False
    try:
        for ev in run_agent(Path(video), game, client=client, demo_fault=demo_fault):
            view.apply(ev)
            tab = None
            if ev.stage == "synthesizer" and not on_report:   # 開始寫報告時切到報告分頁，看串流
                tab, on_report = gr.Tabs(selected="report"), True
            if ev.kind == "report_delta" and time.monotonic() - last < DELTA_INTERVAL_S:
                continue
            last = time.monotonic()
            yield _outputs(view, tab)
    except Exception as e:  # noqa: BLE001  介面不能因為未預期的錯誤卡住
        view.status = f"❌ 發生未預期的錯誤：{type(e).__name__}: {e}"
        view.messages.append({"role": "assistant", "content": str(e),
                              "metadata": {"title": "❌ 未預期的錯誤", "status": "done"}})
    yield _outputs(view, gr.Tabs(selected="report") if view.finished else None)


def build_app() -> gr.Blocks:
    from .workspace import build_workspace
    return build_workspace()


def launch_options(demo: gr.Blocks) -> dict:
    """launch() 需要的樣式與檔案權限。Gradio 6 的 css、theme 改在 launch() 設定；Colab 筆記本也用這個。"""
    runs = project_path(load_config()["paths"]["runs"])
    runs.mkdir(parents=True, exist_ok=True)
    return _supported(demo.launch, allowed_paths=[str(runs)], css=CSS, theme=getattr(demo, "studio_theme", None))


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False, *,
           root_path: str | None = None, max_file_size: str = "200mb") -> None:
    demo = build_app()
    demo.queue(default_concurrency_limit=1)     # YOLO 與 Gemini 額度都有限，一次只跑一個分析
    demo.launch(server_name=host, server_port=port, share=share,
                root_path=root_path, max_file_size=max_file_size, **launch_options(demo))
