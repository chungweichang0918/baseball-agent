"""RE24 得分期望值查表與戰術評估。

RE24 是在 24 種「壘包 × 出局」狀態下，從當下到半局結束平均還能得幾分，
一律以進攻方的觀點計算。比較戰術時：

    ΔRE = 本次得分 + RE(戰術後的狀態) − RE(戰術前的狀態)

數值表與各戰術的限制條件放在 data/re24.json，狀態轉換寫在本模組。
Critic 也會用 check_option() 判斷報告中的戰術是否違反賽況。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import load_config, project_path
from ..schemas import OUTS_ZH, GameState, Role, ToolResult, bases_zh

BASE_ZH = {"1": "一壘", "2": "二壘", "3": "三壘"}
OUTCOME_ZH = {"success": "成功", "fail": "失敗", "result": "結果", "baseline": "基準"}
ROLE_ZH = {"offense": "進攻", "defense": "防守"}


@dataclass(frozen=True)
class BaseOutState:
    first: bool = False
    second: bool = False
    third: bool = False
    outs: int = 0

    @classmethod
    def from_key(cls, key: str, outs: int) -> BaseOutState:
        if len(key) != 3 or any(c not in (d, "_") for c, d in zip(key, "123")):
            raise ValueError(f"壘包代碼格式錯誤：{key!r}（應為 1_3 這類格式）")
        return cls(key[0] == "1", key[1] == "2", key[2] == "3", outs)

    @classmethod
    def from_game(cls, g: GameState) -> BaseOutState:
        return cls(g.on_first, g.on_second, g.on_third, g.outs)

    @property
    def key(self) -> str:
        return "".join(c if on else "_" for c, on in zip("123", (self.first, self.second, self.third)))

    def occupied(self, base: str) -> bool:
        return {"1": self.first, "2": self.second, "3": self.third}[base]

    def describe_zh(self) -> str:
        if self.outs >= 3:
            return "三出局，攻守交換"
        return f"{bases_zh(self.first, self.second, self.third)}，{OUTS_ZH[self.outs]}"


# ---------- 狀態轉換：回傳 {結果名稱: (轉換後狀態, 本次得分)} ----------

Outcomes = dict[str, tuple[BaseOutState, int]]


def _advance_all(s: BaseOutState) -> tuple[BaseOutState, int]:
    """所有跑者推進一個壘包（不含打者），三壘跑者得分。"""
    return replace(s, first=False, second=s.first, third=s.second), int(s.third)


def _walk(s: BaseOutState) -> tuple[BaseOutState, int]:
    """打者上一壘，只有被擠壓的跑者推進。"""
    if not s.first:
        return replace(s, first=True), 0
    if not s.second:
        return replace(s, second=True), 0
    if not s.third:
        return replace(s, third=True), 0
    return s, 1


def _sac_bunt(s: BaseOutState) -> Outcomes:
    after, runs = _advance_all(s)
    return {"success": (replace(after, outs=s.outs + 1), runs)}


TRANSITIONS: dict[str, Callable[[BaseOutState], Outcomes]] = {
    "steal_2b": lambda s: {
        "success": (replace(s, first=False, second=True), 0),
        "fail": (replace(s, first=False, outs=s.outs + 1), 0),
    },
    "steal_3b": lambda s: {
        "success": (replace(s, second=False, third=True), 0),
        "fail": (replace(s, second=False, outs=s.outs + 1), 0),
    },
    "sac_bunt": _sac_bunt,
    "swing_away": lambda s: {"baseline": (s, 0)},
    "ibb": lambda s: {"result": _walk(s)},
    "pitch_to_batter": lambda s: {"baseline": (s, 0)},
}


class RE24:
    def __init__(self, data: dict[str, Any]):
        table = data["table"]
        expected = {BaseOutState.from_key(k, 0).key for k in table}
        if len(expected) != 8 or any(len(v) != 3 for v in table.values()):
            raise ValueError("RE24 表必須有 8 種壘包狀態，每種 3 個出局數的值")
        self.table: dict[str, list[float]] = table
        self.options: dict[str, dict[str, Any]] = data["options"]
        self.meta: dict[str, Any] = data.get("meta", {})
        missing = set(self.options) - set(TRANSITIONS)
        if missing:
            raise ValueError(f"re24.json 中的戰術沒有對應的狀態轉換：{sorted(missing)}")

    @classmethod
    def load(cls, path: str | Path | None = None) -> RE24:
        path = Path(path) if path else project_path(load_config()["paths"]["re24"])
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def value(self, s: BaseOutState) -> float:
        if s.outs >= 3:
            return 0.0
        return self.table[s.key][s.outs]

    def label(self, option: str) -> str:
        return self.options.get(option, {}).get("label", option)

    def check_option(self, s: BaseOutState, option: str, our_role: Role | None = None) -> str | None:
        """戰術可行時回傳 None，不可行時回傳原因。"""
        spec = self.options.get(option)
        if spec is None:
            return f"未知的戰術：{option}"
        label = spec["label"]
        if our_role and spec["role"] != our_role:
            return f"{label}是{ROLE_ZH[spec['role']]}方的戰術，與我方角色（{ROLE_ZH[our_role]}）不符"
        for base in spec.get("requires", []):
            if not s.occupied(base):
                return f"{label}需要{BASE_ZH[base]}有跑者"
        for base in spec.get("requires_empty", []):
            if s.occupied(base):
                return f"{label}需要{BASE_ZH[base]}沒有跑者"
        if spec.get("requires_any_runner") and not (s.first or s.second or s.third):
            return f"壘上無人時不能{label}"
        if s.outs in spec.get("forbid_outs", []):
            return f"{OUTS_ZH[s.outs]}時不能{label}"
        return None

    def evaluate(self, s: BaseOutState, option: str, our_role: Role = "offense",
                 success_prob: float | None = None) -> dict[str, Any]:
        """評估單一戰術。delta 為進攻方觀點，delta_for_us 已換算成我方觀點。"""
        result: dict[str, Any] = {
            "option": option,
            "label": self.label(option),
            "state_before": s.describe_zh(),
            "re_before": self.value(s),
        }
        reason = self.check_option(s, option, our_role)
        if reason:
            return result | {"valid": False, "reason": reason}

        sign = 1 if our_role == "offense" else -1
        outcomes = []
        for name, (after, runs) in TRANSITIONS[option](s).items():
            delta = round(runs + self.value(after) - self.value(s), 3)
            outcomes.append({
                "name": name,
                "label": OUTCOME_ZH[name],
                "state_after": after.describe_zh(),
                "runs": runs,
                "re_after": self.value(after),
                "delta": delta,
                "delta_for_us": round(sign * delta, 3),
            })
        result |= {"valid": True, "reason": None, "outcomes": outcomes,
                   "break_even": None, "expected_delta": None}

        by_name = {o["name"]: o["delta"] for o in outcomes}
        if {"success", "fail"} <= by_name.keys():
            gain, loss = by_name["success"], by_name["fail"]
            if gain > 0 > loss:
                result["break_even"] = round(-loss / (gain - loss), 3)
            if success_prob is not None:
                if not 0 <= success_prob <= 1:
                    raise ValueError("success_prob 必須介於 0 和 1 之間")
                result["expected_delta"] = round(success_prob * gain + (1 - success_prob) * loss, 3)
        return result


@lru_cache(maxsize=1)
def get_re24() -> RE24:
    return RE24.load()


# ---------- 工具介面（Executor 呼叫） ----------

def re24_lookup(bases: str, outs: int) -> ToolResult:
    try:
        s = BaseOutState.from_key(bases, outs)
        value = get_re24().value(s)
    except (ValueError, KeyError) as e:
        return ToolResult(status="failed", summary="RE24 查表失敗", error=str(e))
    return ToolResult(
        status="ok",
        summary=f"{s.describe_zh()}：得分期望值 {value:.3f}",
        data={"bases": s.key, "outs": outs, "re": value},
    )


def _fmt_delta(x: float) -> str:
    # 保持 ASCII 正負號，Critic 比對報告數字時才不用處理全形符號
    return f"{x:+.3f}"


def re24_evaluate(game: GameState, options: list[str],
                  success_probs: dict[str, float] | None = None) -> ToolResult:
    """評估多個候選戰術。success_probs 可提供各戰術的預估成功率，用來算期望 ΔRE。"""
    re24 = get_re24()
    s = BaseOutState.from_game(game)
    probs = success_probs or {}
    evals = [re24.evaluate(s, opt, game.our_role, probs.get(opt)) for opt in options]

    lines = [f"目前狀態：{s.describe_zh()}，得分期望值 {re24.value(s):.3f}（ΔRE 以進攻方計）"]
    for e in evals:
        if not e["valid"]:
            lines.append(f"- {e['label']}：不適用（{e['reason']}）")
            continue
        parts = [f"{o['label']} ΔRE {_fmt_delta(o['delta'])}" for o in e["outcomes"]]
        if game.our_role == "defense":  # 防守方看到正的 ΔRE 容易誤讀成好事
            parts = [f"{p}（對我方 {_fmt_delta(o['delta_for_us'])}）" for p, o in zip(parts, e["outcomes"])]
        line = f"- {e['label']}：" + "／".join(parts)
        if e["break_even"] is not None:
            line += f"；損益平衡成功率 {e['break_even']:.1%}"
        if e["expected_delta"] is not None:
            line += f"；以成功率 {probs[e['option']]:.0%} 估計，期望 ΔRE {_fmt_delta(e['expected_delta'])}"
        lines.append(line)

    return ToolResult(
        status="ok",
        summary="\n".join(lines),
        data={
            "bases": s.key,
            "outs": s.outs,
            "re_before": re24.value(s),
            "our_role": game.our_role,
            "options": evals,
            "source": re24.meta,
        },
    )
