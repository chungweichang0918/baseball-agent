import json
from pathlib import Path

from fake_gemini import FakeSDK, api_error

from bbagent.agent.planner import (PlanDraft, Planner, default_options, default_plan, validate_draft,
                                   valid_options)
from bbagent.agent.state import AgentState
from bbagent.llm.gemini_client import GeminiClient
from bbagent.schemas import GameState, ToolResult


def state_for(tmp_path, **game) -> AgentState:
    g = GameState(**({"inning": 7, "half": "bottom", "outs": 1, "on_first": True, "our_role": "offense"} | game))
    s = AgentState(video=Path("x.mp4"), game=g, run_dir=tmp_path)
    s.add_evidence("r1", "probe_video", {}, ToolResult(status="ok", summary="影片資訊"))
    s.add_evidence("r2", "scout_overview", {}, ToolResult(status="ok", summary="正側面",
                                                          data={"plan_view": "side", "camera_moving": False}))
    return s


def step(i, tool, deps=(), focus=None):
    return {"id": f"s{i}", "tool": tool, "purpose": "p", "depends_on": list(deps), "inspect_focus": focus}


GOOD_STEPS = [step(1, "coarse_pose_scan"), step(2, "segment_motions", ["s1"]), step(3, "pitching_metrics", ["s2"]),
              step(4, "render_keyframes", ["s2"]), step(5, "gemini_inspect", ["s4"], "固定姿勢時的視線"),
              step(6, "re24_evaluate")]


def draft(steps=GOOD_STEPS, options=("steal_2b", "sac_bunt"), focus="pitcher") -> PlanDraft:
    return PlanDraft(analysis_focus=focus, rationale="一壘有人，評估盜壘", tactical_questions=["要不要盜二壘"],
                     candidate_options=list(options), steps=steps)


def test_valid_draft(tmp_path):
    plan, errors, warnings = validate_draft(draft(options=["steal_2b", "ibb"]), state_for(tmp_path))
    assert errors == [] and plan is not None
    assert plan.candidate_options == ["steal_2b", "swing_away"]          # ibb 不適用，補上比較基準
    assert any("ibb" in w for w in warnings)
    by_tool = {s.tool: s for s in plan.steps}
    assert by_tool["re24_evaluate"].args == {"options": ["steal_2b", "swing_away"]}
    assert by_tool["gemini_inspect"].args == {"focus": "固定姿勢時的視線"}
    assert plan.camera_view == "side"


def test_invalid_drafts(tmp_path):
    s = state_for(tmp_path)
    _, errors, _ = validate_draft(draft(steps=GOOD_STEPS[:2] + GOOD_STEPS[5:]), s)
    assert any("pitching_metrics" in e for e in errors)
    _, errors, _ = validate_draft(draft(steps=[step(1, "segment_motions")] + GOOD_STEPS), s)
    assert any("需要先執行 coarse_pose_scan" in e for e in errors)
    _, errors, _ = validate_draft(draft(steps=GOOD_STEPS + [step(9, "measure_bat_speed")]), s)
    assert any("不存在的工具 measure_bat_speed" in e for e in errors)
    _, errors, _ = validate_draft(draft(focus="batter"), s)            # 打者計畫裡放了投手專用的工具
    assert any("pitching_metrics 不適用於打者分析" in e for e in errors)
    assert any("缺少必要步驟 batting_metrics" in e for e in errors)


def test_lenient_cases(tmp_path):
    s = state_for(tmp_path)
    steps = [step(0, "probe_video"), step(1, "coarse_pose_scan"), step(2, "segment_motions", ["coarse_pose_scan"]),
             *GOOD_STEPS[2:], step(7, "gemini_inspect", ["s4"], "a"), step(8, "gemini_inspect", ["s4"], "b")]
    plan, errors, warnings = validate_draft(draft(steps=steps), s)
    assert errors == []
    assert [x.depends_on for x in plan.steps if x.tool == "segment_motions"] == [["s1"]]   # 工具名稱轉成步驟 id
    assert sum(x.tool == "gemini_inspect" for x in plan.steps) == 2
    assert any("probe_video" in w for w in warnings) and any("超過" in w for w in warnings)


def test_default_plan_and_options(tmp_path):
    s = state_for(tmp_path)
    plan = default_plan(s)
    tools = [x.tool for x in plan.execution_order()]
    assert tools[0] == "coarse_pose_scan" and "re24_evaluate" in tools and len(tools) == 9
    assert tools.index("pitching_metrics") < tools.index("track_ball")
    assert tools.index("segment_motions") < tools.index("pitching_metrics") < tools.index("tipping_compare")
    assert default_options(s)[-1] == "swing_away"
    reduced = default_plan(s, excluded={"segment_motions", "pitching_metrics", "tipping_compare", "render_keyframes",
                                        "gemini_inspect"}, done={"coarse_pose_scan"})
    assert [x.tool for x in reduced.steps] == ["camera_motion", "re24_evaluate"]


def test_defense_options(tmp_path):
    s = state_for(tmp_path, our_role="defense", on_first=False, on_second=True)
    ok, bad = valid_options(s)
    assert "ibb" in ok and "steal_2b" in bad
    assert default_options(s)[-1] == "pitch_to_batter"


def client(tmp_path, replies):
    return GeminiClient(cache_dir=tmp_path, cache_mode="off", sdk_client=FakeSDK(replies), sleep=lambda s: None)


def run(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return stop.value, events


def test_planner_retries_with_feedback(tmp_path):
    bad = draft(steps=GOOD_STEPS[:1]).model_dump_json()
    c = client(tmp_path, [bad, draft().model_dump_json()])
    s = state_for(tmp_path)
    plan, events = run(Planner(c, "m").plan(s))
    assert s.plan_source == "llm" and len(plan.steps) == 6
    assert [e.kind for e in events].count("critique") == 1
    second_prompt = c._sdk.models.calls[1]["contents"][0]
    assert "上一份計畫沒有通過檢查" in second_prompt and "缺少必要步驟" in second_prompt


def test_planner_falls_back_when_gemini_fails(tmp_path):
    s = state_for(tmp_path)
    plan, events = run(Planner(client(tmp_path, [api_error(400, "bad")]), "m").plan(s))
    assert s.plan_source == "default" and len(plan.steps) == 9
    assert any(e.kind == "error" for e in events)


def test_replan_excludes_failed_tool_and_dependents(tmp_path):
    s = state_for(tmp_path)
    s.add_evidence("s1", "coarse_pose_scan", {}, ToolResult(status="ok", summary="ok"))
    s.add_evidence("s2", "segment_motions", {}, ToolResult(status="failed", summary="no", error="找不到投球"))
    reply = draft(steps=[step(1, "re24_evaluate")]).model_dump_json()
    c = client(tmp_path, [reply])
    plan, events = run(Planner(c, "m").replan(s, {"step": "s2", "tool": "segment_motions", "error": "找不到投球"}))
    assert [x.tool for x in plan.steps] == ["re24_evaluate"] and s.counters["replans"] == 1
    prompt = c._sdk.models.calls[0]["contents"][0]
    assert "不可使用的工具" in prompt and "gemini_inspect" in prompt
    assert json.loads(reply)["steps"][0]["tool"] == "re24_evaluate"


BATTER_STEPS = [step(1, "coarse_pose_scan"), step(2, "segment_motions", ["s1"]), step(3, "batting_metrics", ["s2"]),
                step(4, "swing_compare", ["s3"]), step(5, "re24_evaluate")]


def test_batter_branch(tmp_path):
    s = state_for(tmp_path, our_role="defense")
    plan, errors, _ = validate_draft(draft(steps=BATTER_STEPS, options=["ibb"], focus="batter"), s)
    assert errors == [] and plan.analysis_focus == "batter"
    assert plan.candidate_options == ["ibb", "pitch_to_batter"]

    _, errors, _ = validate_draft(draft(steps=BATTER_STEPS + [step(6, "tipping_compare", ["s3"])], focus="batter"), s)
    assert any("tipping_compare 不適用於打者分析" in e for e in errors)

    target = state_for(tmp_path, analysis_target="batter")
    _, errors, _ = validate_draft(draft(), target)                   # 使用者指定打者，Gemini 卻規劃投手
    assert any("使用者指定分析打者" in e for e in errors)


def test_focus_and_default_plan_follow_overview(tmp_path):
    s = state_for(tmp_path)
    s.evidence["E02"].result.data["subject"] = "batter"
    assert s.focus == "batter"
    tools = [x.tool for x in default_plan(s).steps]
    assert "batting_metrics" in tools and "swing_compare" in tools and "pitching_metrics" not in tools
    assert default_plan(s).analysis_focus == "batter"
    pinned = state_for(tmp_path, analysis_target="pitcher")
    pinned.evidence["E02"].result.data["subject"] = "batter"
    assert pinned.focus == "pitcher"                                 # 使用者指定優先於影片概況
