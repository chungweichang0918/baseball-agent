"""全專案共用的資料結構。

Agent 各階段、工具與 UI 之間只透過這些型別交換資料，
方便序列化（存執行紀錄）與驗證（LLM 的結構化輸出）。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Role = Literal["offense", "defense"]
Subject = Literal["pitcher", "batter"]
CameraView = Literal["side", "center_field", "other"]
Hand = Literal["R", "L"]

OUTS_ZH = {0: "無人出局", 1: "一出局", 2: "兩出局", 3: "三出局"}
BASE_ZH = ("一", "二", "三")


def bases_zh(first: bool, second: bool, third: bool) -> str:
    occupied = [z for z, on in zip(BASE_ZH, (first, second, third)) if on]
    if not occupied:
        return "壘上無人"
    if len(occupied) == 3:
        return "滿壘"
    return "".join(occupied) + "壘有人"


# ---------- 賽況 ----------

class GameState(BaseModel):
    """使用者在 UI 填入的結構化賽況。比分差以我方觀點計算（我方 − 對方）。"""

    inning: int = Field(ge=1, le=15)
    half: Literal["top", "bottom"]
    outs: int = Field(ge=0, le=2)
    on_first: bool = False
    on_second: bool = False
    on_third: bool = False
    balls: int = Field(default=0, ge=0, le=3)
    strikes: int = Field(default=0, ge=0, le=2)
    score_diff: int = 0
    our_role: Role
    analysis_target: Literal["auto", "pitcher", "batter"] = "auto"
    pitcher_hand: Hand | None = None
    batter_hand: Hand | None = None
    subject_track_id: int | None = Field(default=None, ge=0)
    notes: str = ""

    @property
    def bases_key(self) -> str:
        """RE24 表的鍵，例如一三壘有人為 "1_3"。"""
        return "".join(c if on else "_" for c, on in zip("123", (self.on_first, self.on_second, self.on_third)))

    def describe_zh(self) -> str:
        if self.score_diff > 0:
            score = f"我方領先 {self.score_diff} 分"
        elif self.score_diff < 0:
            score = f"我方落後 {-self.score_diff} 分"
        else:
            score = "平手"
        return "｜".join([
            f"{self.inning} 局{'上' if self.half == 'top' else '下'}",
            OUTS_ZH[self.outs],
            bases_zh(self.on_first, self.on_second, self.on_third),
            f"球數 {self.balls}-{self.strikes}",
            score,
            f"我方{'進攻' if self.our_role == 'offense' else '防守'}",
        ])


# ---------- 工具 ----------

class ToolResult(BaseModel):
    """所有工具的統一回傳格式。evidence_id 由 Executor 寫入證據庫時指定。"""

    status: Literal["ok", "degraded", "failed"]
    summary: str                                   # 給 LLM 與 UI 看的短文字
    data: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)  # 圖片或圖表路徑
    evidence_id: str | None = None
    error: str | None = None


# ---------- 計畫 ----------

class PlanStep(BaseModel):
    id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    purpose: str = ""
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """Planner 的輸出。這裡只做結構檢查；工具白名單與賽況相關的檢查在 Planner 中做。"""

    analysis_focus: Subject
    camera_view: CameraView
    rationale: str
    tactical_questions: list[str] = Field(default_factory=list)
    candidate_options: list[str] = Field(default_factory=list)
    steps: list[PlanStep]

    @model_validator(mode="after")
    def _check_steps(self) -> Plan:
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("步驟 id 重複")
        known = set(ids)
        for s in self.steps:
            missing = set(s.depends_on) - known
            if missing:
                raise ValueError(f"步驟 {s.id} 依賴不存在的步驟：{sorted(missing)}")
        self.execution_order()  # 有循環會拋出錯誤
        return self

    def execution_order(self) -> list[PlanStep]:
        """依相依關係排出執行順序（拓撲排序）；同一層維持原本順序。"""
        done: set[str] = set()
        order: list[PlanStep] = []
        pending = list(self.steps)
        while pending:
            ready = [s for s in pending if set(s.depends_on) <= done]
            if not ready:
                raise ValueError(f"步驟相依關係有循環：{[s.id for s in pending]}")
            for s in ready:
                order.append(s)
                done.add(s.id)
            pending = [s for s in pending if s.id not in done]
        return order


# ---------- 事件串流 ----------

class AgentEvent(BaseModel):
    """Orchestrator 每一步產生的事件。UI 與 CLI 都只消費這個型別。"""

    stage: Literal["planner", "executor", "synthesizer", "critic", "system"]
    kind: Literal[
        "thought", "tool_call", "tool_result", "progress", "artifact",
        "report_delta", "critique", "error", "final",
    ]
    title: str
    content: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
