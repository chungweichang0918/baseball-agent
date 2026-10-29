"""Executor：依計畫的相依順序執行工具，把結果寫進證據庫並串流事件。

工具在背景執行緒中執行，主執行緒持續把進度訊息（YOLO 進度、Gemini 重試）轉成事件，
UI 因此能在長時間的步驟中即時更新。
"""

from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Generator

from ..llm.gemini_client import GeminiClient
from ..schemas import AgentEvent, Plan, PlanStep, ToolResult
from .state import STATUS_ZH, AgentState, Evidence
from .tools import REGISTRY, MissingRuntime, ToolContext

POLL_S = 0.2


class Executor:
    def __init__(self, client: GeminiClient, notices: queue.Queue):
        self.client = client
        self.notices = notices     # 工具進度與 Gemini 重試訊息共用的佇列

    def drain(self, stage: str = "executor") -> Generator[AgentEvent, None, None]:
        while True:
            try:
                msg = self.notices.get_nowait()
            except queue.Empty:
                return
            yield AgentEvent(stage=stage, kind="progress", title="進度", content=msg)

    def run_step(self, state: AgentState, step: PlanStep) -> Generator[AgentEvent, None, Evidence]:
        spec = REGISTRY[step.tool]
        yield AgentEvent(stage="executor", kind="tool_call", title=f"{step.id} 呼叫 {step.tool}",
                         content=step.purpose or spec.description, payload={"step": step.id, "tool": step.tool,
                                                                            "args": step.args})
        missing = [r for r in spec.requires if not state.succeeded(r)]
        if state.focus not in spec.focus:
            result = ToolResult(status="failed", summary=f"略過：{step.tool} 不適用於這個分析分支",
                                error=f"{step.tool} 只適用於 {'、'.join(spec.focus)}")
        elif missing:
            result = ToolResult(status="failed", summary=f"略過：前置步驟 {'、'.join(missing)} 沒有成功",
                                error=f"缺少前置步驟：{'、'.join(missing)}")
        else:
            result = yield from self._run_threaded(state, step)
        ev = state.add_evidence(step.id, step.tool, step.args, result)
        yield AgentEvent(stage="executor", kind="tool_result",
                         title=f"{ev.id} {step.tool}：{STATUS_ZH[result.status]}",
                         content=result.summary + (f"\n錯誤：{result.error}" if result.error else ""),
                         payload={"evidence_id": ev.id, "tool": step.tool, "status": result.status,
                                  "artifacts": result.artifacts, "data": result.data})
        return ev

    def run_plan(self, state: AgentState, plan: Plan) -> Generator[AgentEvent, None, dict | None]:
        """依序執行。關鍵工具失敗時停止並回傳失敗資訊，交給 Planner 重新規劃。"""
        for step in plan.execution_order():
            ev = yield from self.run_step(state, step)
            if ev.result.status == "failed" and REGISTRY[step.tool].critical:
                return {"step": step.id, "tool": step.tool, "error": ev.result.error or ev.result.summary}
        return None

    def _run_threaded(self, state: AgentState, step: PlanStep) -> Generator[AgentEvent, None, ToolResult]:
        ctx = ToolContext(state, self.client, self.notices.put)
        box: dict = {}

        def work() -> None:
            try:
                box["result"] = REGISTRY[step.tool].run(ctx, step.args)
            except MissingRuntime as e:
                box["result"] = ToolResult(status="failed", summary="缺少前置步驟的結果", error=f"缺少 {e}")
            except Exception as e:  # noqa: BLE001  工具內部錯誤不能讓 Agent 停擺
                box["result"] = ToolResult(status="failed", summary="工具執行時發生程式錯誤",
                                           error=f"{type(e).__name__}: {e}",
                                           data={"traceback": traceback.format_exc(limit=5)})

        t = threading.Thread(target=work, daemon=True)
        t.start()
        while t.is_alive():
            t.join(POLL_S)
            yield from self.drain()
        yield from self.drain()
        return box["result"]
