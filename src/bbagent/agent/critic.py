"""Critic：先跑規則檢查，規則全部通過才請 Gemini 檢查推論品質（省額度）。

有 high 等級的問題就不通過，問題清單會交回 Synthesizer 修正。這就是 Reflection 迴圈。
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from ..llm.gemini_client import GeminiClient, GeminiError, model_name
from ..llm.prompts import load_prompt
from ..schemas import AgentEvent
from .critic_rules import check_rules
from .state import AgentState


class CriticIssue(BaseModel):
    severity: Literal["high", "medium", "low"]
    location: str
    message: str


class CriticReview(BaseModel):
    passed: bool
    issues: list[CriticIssue]


@dataclass
class Verdict:
    passed: bool
    issues: list[dict] = field(default_factory=list)
    llm_checked: bool = False

    @property
    def blocking(self) -> list[dict]:
        return [i for i in self.issues if i["severity"] == "high"]


def _fmt_issues(issues: list[dict]) -> str:
    return "\n".join(f"- [{i['severity']}] {i['location']}：{i['message']}" for i in issues)


class Critic:
    def __init__(self, client: GeminiClient, model: str | None = None):
        self.client = client
        self.model = model or model_name("fast")

    def review(self, state: AgentState, report: str, round_no: int = 0) -> Generator[AgentEvent, None, Verdict]:
        yield AgentEvent(stage="critic", kind="thought", title=f"審查第 {round_no + 1} 版報告",
                         content="先用規則檢查格式、證據引用、RE24 數字與戰術合法性。")
        issues = check_rules(state, report)
        yield AgentEvent(stage="critic", kind="thought",
                         title="規則檢查通過" if not issues else f"規則檢查發現 {len(issues)} 項問題",
                         content=_fmt_issues(issues) or "格式、引用、數字與戰術都符合規則。",
                         payload={"issues": issues})

        llm_checked = False
        if not issues:
            text = "\n".join([f"【賽況】{state.game.describe_zh()}", "【證據清單】",
                              *[ev.brief() for ev in state.evidence.values()], "【報告草稿】", report])
            try:
                review, _ = self.client.generate_structured(self.model, [text], CriticReview,
                                                            system=load_prompt("critic"))
                llm_checked = True
                issues = [{"source": "llm", **i.model_dump()} for i in review.issues]
                yield AgentEvent(stage="critic", kind="thought", title=f"推論品質檢查：{len(issues)} 項意見",
                                 content=_fmt_issues(issues) or "推論有依據、建議具體、限制有交代。",
                                 payload={"issues": issues})
            except GeminiError as e:
                yield AgentEvent(stage="critic", kind="error", title="略過推論品質檢查（Gemini 無法使用）",
                                 content=str(e))

        verdict = Verdict(passed=not any(i["severity"] == "high" for i in issues), issues=issues,
                          llm_checked=llm_checked)
        state.critiques.append({"round": round_no, "passed": verdict.passed, "issues": issues,
                                "llm_checked": llm_checked})
        title = "✅ 審查通過" if verdict.passed else f"❌ 審查未通過：{len(verdict.blocking)} 項必須修正"
        yield AgentEvent(stage="critic", kind="critique", title=title, content=_fmt_issues(verdict.blocking),
                         payload={"passed": verdict.passed, "issues": issues, "round": round_no})
        return verdict
