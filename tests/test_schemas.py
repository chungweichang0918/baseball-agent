import pytest
from pydantic import ValidationError

from bbagent.schemas import AgentEvent, GameState, Plan, ToolResult


def test_game_state_describe_and_key():
    g = GameState(inning=7, half="bottom", outs=1, on_first=True, on_third=True,
                  balls=1, strikes=1, score_diff=0, our_role="offense")
    assert g.bases_key == "1_3"
    assert g.describe_zh() == "7 局下｜一出局｜一三壘有人｜球數 1-1｜平手｜我方進攻"
    g2 = g.model_copy(update={"score_diff": -2, "our_role": "defense", "half": "top",
                              "on_second": True})
    assert g2.bases_key == "123"
    assert "滿壘" in g2.describe_zh() and "我方落後 2 分" in g2.describe_zh()


@pytest.mark.parametrize("field,value", [("outs", 3), ("balls", 4), ("strikes", 3), ("inning", 0)])
def test_game_state_rejects_out_of_range(field, value):
    kwargs = dict(inning=1, half="top", outs=0, our_role="offense") | {field: value}
    with pytest.raises(ValidationError):
        GameState(**kwargs)


def _plan(steps):
    return {"analysis_focus": "pitcher", "camera_view": "side", "rationale": "測試", "steps": steps}


def test_plan_execution_order():
    p = Plan.model_validate(_plan([
        {"id": "s3", "tool": "c", "depends_on": ["s1", "s2"]},
        {"id": "s1", "tool": "a"},
        {"id": "s2", "tool": "b", "depends_on": ["s1"]},
    ]))
    assert [s.id for s in p.execution_order()] == ["s1", "s2", "s3"]


@pytest.mark.parametrize("steps,msg", [
    ([{"id": "s1", "tool": "a"}, {"id": "s1", "tool": "b"}], "重複"),
    ([{"id": "s1", "tool": "a", "depends_on": ["s9"]}], "不存在"),
    ([{"id": "s1", "tool": "a", "depends_on": ["s2"]},
      {"id": "s2", "tool": "b", "depends_on": ["s1"]}], "循環"),
])
def test_plan_rejects_bad_steps(steps, msg):
    with pytest.raises(ValidationError, match=msg):
        Plan.model_validate(_plan(steps))


def test_tool_result_and_event_roundtrip():
    r = ToolResult(status="degraded", summary="球軌跡品質不足", data={"inliers": 3})
    assert ToolResult.model_validate_json(r.model_dump_json()) == r
    e = AgentEvent(stage="executor", kind="tool_result", title="track_ball", payload=r.model_dump())
    assert AgentEvent.model_validate_json(e.model_dump_json()).payload["data"]["inliers"] == 3
