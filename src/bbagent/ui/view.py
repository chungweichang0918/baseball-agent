"""把 AgentEvent 串流整理成 UI 要顯示的狀態。不依賴 Gradio，方便測試。

  messages   思考與工具呼叫紀錄（Gradio Chatbot 的 messages 格式，工具呼叫用可收合的區塊）
  plan_md    計畫表：每個步驟的工具、目的與執行狀態
  keyframes  關鍵幀圖庫 [(路徑, 說明)]
  curves     動作曲線圖路徑
  report_md  報告（撰寫中的草稿或最終版）
  review_md  每一輪審查的結果（展示 Reflection）
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import AgentEvent

STAGE_ICON = {"planner": "🧭", "executor": "🔧", "synthesizer": "✍️", "critic": "🔍", "system": "⚙️"}
STATUS_ICON = {"ok": "✅", "degraded": "⚠️", "failed": "❌"}
ROW_STATUS = {"waiting": "⏸️ 等待", "running": "⏳ 執行中", "ok": "✅ 成功", "degraded": "⚠️ 部分成功",
              "failed": "❌ 失敗", "skipped": "⏭️ 略過（重新規劃）"}
RECON = {"r1": "讀取影片資訊", "r2": "判斷視角與拍攝條件"}


def _md(text: str) -> str:
    """Markdown 中單一換行會被合併，改成強制換行。"""
    return text.replace("\n", "  \n")


def _cell(text: str, limit: int = 30) -> str:
    text = text.replace("|", "／").replace("\n", " ")
    return text if len(text) <= limit else text[:limit - 1] + "…"


@dataclass
class Row:
    key: str
    step: str
    tool: str
    purpose: str
    deps: list[str] = field(default_factory=list)
    status: str = "waiting"
    evidence: str = ""


@dataclass
class RunView:
    messages: list[dict] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    keyframes: list[tuple[str, str]] = field(default_factory=list)
    curves: str | None = None
    report: str = ""
    report_header: str = ""
    reviews: list[str] = field(default_factory=list)
    final: dict | None = None
    finished: bool = False
    status: str = "⏸️ 等待開始"
    evidence: dict[str, dict] = field(default_factory=dict)
    _plan_no: int = 0
    _current: dict = field(default_factory=dict)      # 執行中的步驟：{"row": Row, "msg": index, "progress": str}
    _round: int = 0

    # ---------- 事件 ----------

    def apply(self, ev: AgentEvent) -> None:
        if ev.kind not in ("report_delta", "progress"):
            self.status = f"{STAGE_ICON[ev.stage]} {ev.title}"
        handler = getattr(self, f"_on_{ev.kind}", None)
        if handler:
            handler(ev)

    def _say(self, ev: AgentEvent, title: str | None = None, content: str | None = None, status: str = "done") -> int:
        self.messages.append({"role": "assistant", "content": _md(ev.content if content is None else content),
                              "metadata": {"title": title or f"{STAGE_ICON[ev.stage]} {ev.title}", "status": status}})
        return len(self.messages) - 1

    def _on_thought(self, ev: AgentEvent) -> None:
        if "plan" in ev.payload:
            self._set_plan(ev.payload["plan"])
        if ev.stage == "synthesizer":
            self._round = ev.payload.get("round", 0)
            self.report = ""
            self.report_header = f"> ✍️ **{ev.title}**（第 {self._round + 1} 版，撰寫中…）"
        self._say(ev)

    def _on_error(self, ev: AgentEvent) -> None:
        self._say(ev, title=f"❌ {ev.title}")

    def _on_critique(self, ev: AgentEvent) -> None:
        self._say(ev, content=ev.content or ("審查通過" if ev.payload.get("passed") else ""))
        if ev.stage != "critic" or "passed" not in ev.payload:
            return
        n = ev.payload.get("round", 0) + 1
        issues = ev.payload.get("issues", [])
        lines = [f"### 第 {n} 版：{'✅ 通過' if ev.payload['passed'] else '❌ 退回'}"]
        for i in issues:
            source = "規則" if i.get("source") == "rule" else "Gemini"
            lines.append(f"- **[{i['severity']}｜{source}]** {i['location']}：{i['message']}")
        if not issues:
            lines.append("- 規則檢查與推論品質檢查都沒有意見")
        self.reviews.append("\n".join(lines))
        verdict = "✅ 審查通過" if ev.payload["passed"] else "❌ 審查未通過，退回修正"
        self.report_header = f"> 🔍 **第 {n} 版：{verdict}**"

    def _on_tool_call(self, ev: AgentEvent) -> None:
        step = ev.payload.get("step", "")
        row = self._row(step, ev.payload.get("tool", ""))
        row.status = "running"
        idx = self._say(ev, content=ev.content, status="pending")
        self._current = {"row": row, "msg": idx, "purpose": ev.content}

    def _on_progress(self, ev: AgentEvent) -> None:
        if self._current and ev.stage == "executor":
            msg = self.messages[self._current["msg"]]
            msg["content"] = _md(f"{self._current['purpose']}\n⏳ {ev.content}")
        else:
            self._say(ev, title=f"⏳ {ev.content}", content="")

    def _on_tool_result(self, ev: AgentEvent) -> None:
        status = ev.payload.get("status", "ok")
        if self._current:
            row = self._current["row"]
            row.status, row.evidence = status, ev.payload.get("evidence_id", "")
            msg = self.messages[self._current["msg"]]
            msg["content"] = _md(ev.content)
            msg["metadata"] = {"title": f"{STATUS_ICON[status]} {ev.title}", "status": "done"}
            self._current = {}
        else:
            self._say(ev, title=f"{STATUS_ICON[status]} {ev.title}")
        tool, data = ev.payload.get("tool"), ev.payload.get("data") or {}
        if tool:
            self.evidence[tool] = dict(ev.payload)
        if tool == "render_keyframes":
            balls = [k for k in self.keyframes if "球軌跡" in k[1]]
            self.keyframes = [(k["path"], k["caption"]) for k in data.get("keyframes", [])] + balls
        elif tool == "track_ball":
            found = [b for b in data.get("pitches", []) if b.get("found")]
            for path, b in zip(ev.payload.get("artifacts") or [], found):
                speed = f"，粗估 {b['speed_kmh_est']:.0f} 公里" if b.get("speed_kmh_est") else ""
                self.keyframes.append((path, f"第 {b['pitch']} 球｜球軌跡{speed}"))
        elif tool == "segment_motions" and ev.payload.get("artifacts"):
            self.curves = ev.payload["artifacts"][0]

    def _on_report_delta(self, ev: AgentEvent) -> None:
        self.report += ev.content

    def _on_final(self, ev: AgentEvent) -> None:
        self.finished = True
        self.final = ev.payload
        for row in self.rows:
            if row.status in ("waiting", "running"):
                row.status = "skipped"
        if ev.payload.get("aborted"):
            self._say(ev, title=f"❌ {ev.title}")
            self.report_header = f"> ❌ **{ev.title}**：{ev.content}"
            return
        self.report = ev.content
        passed = ev.payload.get("passed")
        self.report_header = "" if passed else "> ⚠️ **審查未完全通過**，請參考報告開頭列出的項目自行判讀。"
        self._say(ev, title=f"{'✅' if passed else '⚠️'} {ev.title}",
                  content=f"報告：`{ev.payload.get('report_path', '')}`")

    # ---------- 計畫表 ----------

    def _set_plan(self, plan: dict) -> None:
        self._plan_no += 1
        for row in self.rows:          # 重新規劃：上一個計畫還沒執行的步驟不會再執行
            if row.status == "waiting":
                row.status = "skipped"
        for s in plan["steps"]:
            self.rows.append(Row(key=f"{self._plan_no}:{s['id']}", step=s["id"], tool=s["tool"],
                                 purpose=s.get("purpose", ""), deps=s.get("depends_on", [])))

    def _row(self, step: str, tool: str) -> Row:
        key = f"{self._plan_no}:{step}" if step not in RECON else f"0:{step}"
        for row in self.rows:
            if row.key == key:
                return row
        row = Row(key=key, step=step, tool=tool, purpose=RECON.get(step, ""))
        insert_at = sum(r.key.startswith("0:") for r in self.rows) if key.startswith("0:") else len(self.rows)
        self.rows.insert(insert_at, row)
        return row

    # ---------- 輸出 ----------

    def plan_md(self) -> str:
        if not self.rows:
            return "_尚未開始_"
        lines = ["| 步驟 | 工具 | 目的 | 依賴 | 狀態 | 證據 |", "|---|---|---|---|---|---|"]
        plan_no = 0
        for row in self.rows:
            n = int(row.key.split(":")[0])
            if n > 1 and n != plan_no:
                lines.append(f"| **重新規劃** | | | | | |")
            plan_no = n
            lines.append(f"| {row.step} | `{row.tool}` | {_cell(row.purpose)} | {', '.join(row.deps) or '—'} "
                         f"| {ROW_STATUS[row.status]} | {row.evidence or '—'} |")
        return "\n".join(lines)

    def report_md(self) -> str:
        if not self.report and not self.report_header:
            return "_報告會在分析完成後顯示在這裡，撰寫過程會即時更新。_"
        return (self.report_header + "\n\n" if self.report_header else "") + self.report

    def review_md(self) -> str:
        return "\n\n".join(self.reviews) if self.reviews else "_審查結果會顯示在這裡。_"

    def files(self) -> list[str]:
        if not self.final or self.final.get("aborted"):
            return []
        return [p for p in (self.final.get("report_path"), self.final.get("trace_path")) if p]
