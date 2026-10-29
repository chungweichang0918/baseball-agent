import pytest

from bbagent.schemas import GameState
from bbagent.tools.re24 import RE24, BaseOutState, get_re24, re24_evaluate, re24_lookup


@pytest.fixture(scope="module")
def re24() -> RE24:
    return get_re24()


def state(key: str, outs: int) -> BaseOutState:
    return BaseOutState.from_key(key, outs)


def game(**kw) -> GameState:
    base = dict(inning=7, half="bottom", outs=0, our_role="offense")
    return GameState(**(base | kw))


# ---------- 數值表 ----------

def test_table_shape_and_sanity(re24):
    assert len(re24.table) == 8
    for key, values in re24.table.items():
        assert len(values) == 3
        assert values[0] > values[1] > values[2] > 0, f"{key} 的期望值應隨出局數遞減"
    # 跑者越多、越前進，期望值越高（以無人出局為例）
    assert re24.value(state("___", 0)) < re24.value(state("1__", 0)) < re24.value(state("_2_", 0))
    assert re24.value(state("123", 0)) == max(v[0] for v in re24.table.values())


def test_matches_tango_2010_2015(re24):
    """與 https://tangotiger.net/re24.html 的 2010–2015 表格一致。"""
    assert re24.table == {
        "___": [0.481, 0.254, 0.098],
        "1__": [0.859, 0.509, 0.224],
        "_2_": [1.100, 0.664, 0.319],
        "12_": [1.437, 0.884, 0.429],
        "__3": [1.350, 0.950, 0.353],
        "1_3": [1.784, 1.130, 0.478],
        "_23": [1.964, 1.376, 0.580],
        "123": [2.292, 1.541, 0.752],
    }
    assert re24.meta["seasons"] == "2010–2015" and "tangotiger.net" in re24.meta["source"]


def test_three_outs_is_zero(re24):
    assert re24.value(BaseOutState(True, True, True, outs=3)) == 0.0


@pytest.mark.parametrize("key", ["123", "12", "1-3", "213", "abc"])
def test_bad_key(key):
    if key == "123":
        assert BaseOutState.from_key(key, 0).key == "123"
    else:
        with pytest.raises(ValueError):
            BaseOutState.from_key(key, 0)


# ---------- 規劃書 6.4 的範例 ----------

def test_steal_2b_example(re24):
    e = re24.evaluate(state("1__", 0), "steal_2b")
    d = {o["name"]: o["delta"] for o in e["outcomes"]}
    assert d == {"success": 0.241, "fail": -0.605}
    assert e["break_even"] == 0.715


def test_sac_bunt_example(re24):
    e = re24.evaluate(state("1__", 0), "sac_bunt")
    assert [o["delta"] for o in e["outcomes"]] == [-0.195]
    assert e["outcomes"][0]["state_after"] == "二壘有人，一出局"
    assert e["break_even"] is None


def test_ibb_example_defense(re24):
    e = re24.evaluate(state("_2_", 1), "ibb", our_role="defense")
    o = e["outcomes"][0]
    assert o["delta"] == 0.220            # 對手期望得分增加
    assert o["delta_for_us"] == -0.220    # 對我方（防守）是損失
    assert o["state_after"] == "一二壘有人，一出局"


# ---------- 狀態轉換 ----------

def test_ibb_bases_loaded_forces_run(re24):
    o = re24.evaluate(state("123", 1), "ibb", our_role="defense")["outcomes"][0]
    assert o["runs"] == 1 and o["delta"] == 1.0


def test_ibb_only_forced_runners_move(re24):
    o = re24.evaluate(state("__3", 0), "ibb", our_role="defense")["outcomes"][0]
    assert o["state_after"] == "一三壘有人，無人出局" and o["runs"] == 0


def test_squeeze_scores_runner_from_third(re24):
    o = re24.evaluate(state("__3", 0), "sac_bunt")["outcomes"][0]
    assert o["runs"] == 1
    assert o["delta"] == round(1 + 0.254 - 1.350, 3)


def test_steal_with_two_outs_fail_ends_inning(re24):
    e = re24.evaluate(state("1__", 2), "steal_2b")
    fail = next(o for o in e["outcomes"] if o["name"] == "fail")
    assert fail["state_after"] == "三出局，攻守交換"
    assert fail["re_after"] == 0.0
    assert e["break_even"] == round(0.224 / 0.319, 3)


def test_expected_delta_with_success_prob(re24):
    e = re24.evaluate(state("1__", 0), "steal_2b", success_prob=0.8)
    assert e["expected_delta"] == round(0.8 * 0.241 + 0.2 * -0.605, 3)
    with pytest.raises(ValueError):
        re24.evaluate(state("1__", 0), "steal_2b", success_prob=1.5)


def test_baseline_options_are_zero(re24):
    assert re24.evaluate(state("12_", 1), "swing_away")["outcomes"][0]["delta"] == 0.0
    assert re24.evaluate(state("12_", 1), "pitch_to_batter", our_role="defense")["outcomes"][0]["delta"] == 0.0


# ---------- 防呆（Critic 也會用） ----------

@pytest.mark.parametrize("key,outs,option,role,expect", [
    ("___", 0, "steal_2b", "offense", "需要一壘有跑者"),
    ("12_", 0, "steal_2b", "offense", "需要二壘沒有跑者"),
    ("1_3", 0, "steal_3b", "offense", "需要二壘有跑者"),
    ("1__", 2, "sac_bunt", "offense", "兩出局時不能犧牲觸擊"),
    ("___", 0, "sac_bunt", "offense", "壘上無人時不能犧牲觸擊"),
    ("_2_", 1, "ibb", "offense", "與我方角色（進攻）不符"),
    ("1__", 0, "steal_2b", "defense", "與我方角色（防守）不符"),
    ("1__", 0, "hit_and_run", "offense", "未知的戰術"),
])
def test_invalid_options(re24, key, outs, option, role, expect):
    reason = re24.check_option(state(key, outs), option, role)
    assert reason is not None and expect in reason
    e = re24.evaluate(state(key, outs), option, role)
    assert e["valid"] is False and expect in e["reason"]


def test_valid_options_pass(re24):
    assert re24.check_option(state("1_3", 0), "steal_2b", "offense") is None
    assert re24.check_option(state("_2_", 1), "ibb", "defense") is None
    assert re24.check_option(state("1__", 1), "sac_bunt") is None  # 不指定角色時不檢查角色


# ---------- 工具介面 ----------

def test_lookup_tool():
    r = re24_lookup("1_3", 1)
    assert r.status == "ok" and r.data["re"] == 1.130
    assert "一三壘有人，一出局" in r.summary
    assert re24_lookup("x", 0).status == "failed"
    assert re24_lookup("123", 3).data["re"] == 0.0


def test_evaluate_tool_summary():
    r = re24_evaluate(game(on_first=True), ["steal_2b", "sac_bunt", "swing_away", "ibb"],
                      success_probs={"steal_2b": 0.8})
    assert r.status == "ok"
    assert "損益平衡成功率 71.5%" in r.summary
    assert "成功 ΔRE +0.241" in r.summary and "失敗 ΔRE -0.605" in r.summary
    assert "期望 ΔRE +0.072" in r.summary
    assert "故意四壞：不適用" in r.summary
    assert [o["valid"] for o in r.data["options"]] == [True, True, True, False]
    assert "對我方" not in r.summary


def test_evaluate_tool_summary_defense_perspective():
    g = game(half="top", outs=1, on_second=True, our_role="defense")
    r = re24_evaluate(g, ["ibb", "pitch_to_batter"])
    assert "故意四壞：結果 ΔRE +0.220（對我方 -0.220）" in r.summary
