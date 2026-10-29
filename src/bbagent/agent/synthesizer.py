"""Synthesizer：根據證據清單與關鍵幀，串流撰寫戰術建議報告。"""

from __future__ import annotations

from collections.abc import Generator

from ..llm.gemini_client import GeminiClient, GeminiError, ImageInput, model_name
from ..llm.prompts import load_prompt
from ..schemas import AgentEvent
from .planner import hands_zh
from .state import FOCUS_ZH, AgentState

MAX_REPORT_IMAGES = 4


def synthesizer_input(state: AgentState, feedback: list[str] | None, previous: str | None) -> str:
    g = state.game
    plan = state.plan
    lines = [
        f"【賽況】{g.describe_zh()}｜{hands_zh(g)}",
        f"【分析對象】{FOCUS_ZH[state.focus]}（{'每次揮棒' if state.focus == 'batter' else '每一球'}的動作）",
        f"【使用者補充】{g.notes or '（無）'}",
        f"【分析計畫】{plan.rationale if plan else '（無）'}",
    ]
    if plan and plan.tactical_questions:
        lines.append("【要回答的戰術問題】" + "；".join(plan.tactical_questions))
    lines.append("【證據清單】")
    lines += [ev.brief() for ev in state.evidence.values()]
    if feedback:
        lines.append("【審查意見：請逐條修正】")
        lines += [f"- {f}" for f in feedback]
    if previous:
        lines += ["【上一版報告】", previous]
    return "\n".join(lines)


def fallback_report(state: AgentState, error: str) -> str:
    """Gemini 無法使用時的備援：直接整理證據，至少讓使用者看到分析結果。"""
    g = state.game
    re24 = state.latest("re24_evaluate")
    evidence_rows = [f"| {i} | {ev.tool}：{ev.result.summary.splitlines()[0]} | — | [{ev.id}] |"
                     for i, ev in enumerate(state.evidence.values(), start=1)]
    return "\n".join([
        "# ⚾ 戰術建議報告",
        f"**情境**：{g.describe_zh()}",
        f"**分析對象**：{FOCUS_ZH[state.focus]}",
        "",
        "## 1. 結論",
        f"- 主建議：Gemini 無法使用（{error}），以下僅列出各工具的原始結果，請自行判讀。",
        "",
        "## 2. 影片觀察",
        "| # | 觀察 | 數據 | 證據 |",
        "|---|---|---|---|",
        *evidence_rows,
        "",
        "## 3. 數據依據（RE24）",
        re24.result.summary if re24 else "（沒有 RE24 結果）",
        "",
        "## 4. 戰術建議與推論",
        "（Gemini 無法使用，沒有自動推論）",
        "",
        "## 5. 信心與限制",
        "- 本報告未經 LLM 整合，影片視角、鏡頭移動等限制請參考第 2 節各工具的結果。",
    ])


class Synthesizer:
    def __init__(self, client: GeminiClient, model: str | None = None):
        self.client = client
        self.model = model or model_name("pro")

    def write(self, state: AgentState, feedback: list[str] | None = None, previous: str | None = None,
              round_no: int = 0) -> Generator[AgentEvent, None, str]:
        title = "撰寫報告" if round_no == 0 else f"依審查意見修正報告（第 {round_no} 次）"
        yield AgentEvent(stage="synthesizer", kind="thought", title=title,
                         content=f"整合 {len(state.evidence)} 筆證據" + (f"，處理 {len(feedback)} 項審查意見" if feedback else ""),
                         payload={"round": round_no})
        contents: list = [synthesizer_input(state, feedback, previous)]
        keyframes = state.runtime.get("keyframes", [])
        main, second = ("contact", "foot_plant") if state.focus == "batter" else ("release", "set")
        releases = [k for k in keyframes if k["phase"] == main]
        others = [k for k in keyframes if k["phase"] == second]
        for k in (releases + others)[:MAX_REPORT_IMAGES]:
            try:   # 圖片只是輔助，讀不到就略過，不能讓報告寫不出來
                contents += [f"關鍵幀：{k['caption']}", ImageInput.from_path(k["path"])]
            except (FileNotFoundError, ValueError):
                continue

        text = ""
        try:
            for chunk in self.client.stream(self.model, contents, system=load_prompt("synthesizer")):
                text += chunk
                yield AgentEvent(stage="synthesizer", kind="report_delta", title="報告", content=chunk)
        except GeminiError as e:
            text = fallback_report(state, str(e))
            yield AgentEvent(stage="synthesizer", kind="error", title="Gemini 無法撰寫報告，改用證據彙整", content=str(e))
            yield AgentEvent(stage="synthesizer", kind="report_delta", title="報告", content=text)
        return text
