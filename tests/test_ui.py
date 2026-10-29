from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
from synthetic import make_pitcher
from test_agent_e2e import patch_scan, run, video  # noqa: F401  video 是 fixture

from bbagent.schemas import AgentEvent, GameState
from bbagent.ui.diagram import RUNNER, situation_svg
from bbagent.ui.view import RunView
from bbagent.vision.pose_scan import PoseTrack


def game(**kw) -> GameState:
    return GameState(**({"inning": 7, "half": "bottom", "outs": 1, "on_first": True, "our_role": "offense"} | kw))


def test_situation_svg():
    svg = situation_svg(game(on_third=True, balls=2, strikes=1, score_diff=-2, pitcher_hand="L"))
    root = ElementTree.fromstring(svg)                        # 是合法的 XML
    ns = "{http://www.w3.org/2000/svg}"
    bases = root.findall(f"{ns}polygon")[2:]                  # 內野、本壘之後是一、二、三壘
    assert [b.get("fill") == RUNNER for b in bases] == [True, False, True]
    assert "我方落後 2 分" in svg and "左投" in svg and "7 局 ▼ 下" in svg
    filled = [c for c in root.findall(f"{ns}circle") if c.get("fill") != "currentColor"]
    assert len(filled) == 2 + 1 + 1                           # 2 壞、1 好、1 出局


def ev(stage, kind, title, content="", **payload) -> AgentEvent:
    return AgentEvent(stage=stage, kind=kind, title=title, content=content, payload=payload)


def test_view_tracks_steps_and_replan():
    v = RunView()
    v.apply(ev("executor", "tool_call", "r1 呼叫 probe_video", "讀取影片資訊", step="r1", tool="probe_video"))
    v.apply(ev("executor", "tool_result", "E01 probe_video：成功", "720x1280", status="ok", evidence_id="E01",
               tool="probe_video"))
    plan = {"steps": [{"id": "s1", "tool": "coarse_pose_scan", "purpose": "找投手", "depends_on": []},
                      {"id": "s2", "tool": "segment_motions", "purpose": "分段", "depends_on": ["s1"]}]}
    v.apply(ev("planner", "thought", "計畫完成（2 個步驟）", "…", plan=plan))
    v.apply(ev("executor", "tool_call", "s1 呼叫 coarse_pose_scan", "找投手", step="s1", tool="coarse_pose_scan"))
    v.apply(ev("executor", "progress", "進度", "姿勢模型：10/100 幀（10%）"))
    assert v.messages[-1]["metadata"]["status"] == "pending" and "10%" in v.messages[-1]["content"]
    v.apply(ev("executor", "tool_result", "E02 coarse_pose_scan：失敗", "找不到", status="failed",
               evidence_id="E02", tool="coarse_pose_scan"))
    assert v.messages[-1]["metadata"] == {"title": "❌ E02 coarse_pose_scan：失敗", "status": "done"}

    v.apply(ev("planner", "thought", "改用預設計畫（1 個步驟）", "…",
               plan={"steps": [{"id": "s1", "tool": "re24_evaluate", "purpose": "RE24", "depends_on": []}]}))
    v.apply(ev("executor", "tool_call", "s1 呼叫 re24_evaluate", "RE24", step="s1", tool="re24_evaluate"))
    statuses = [(r.step, r.tool, r.status) for r in v.rows]
    assert statuses == [("r1", "probe_video", "ok"), ("s1", "coarse_pose_scan", "failed"),
                        ("s2", "segment_motions", "skipped"), ("s1", "re24_evaluate", "running")]
    assert "**重新規劃**" in v.plan_md()


def test_view_report_and_reviews():
    v = RunView()
    v.apply(ev("synthesizer", "thought", "撰寫報告", "整合 9 筆證據", round=0))
    v.apply(ev("synthesizer", "report_delta", "報告", "# 草稿"))
    assert "撰寫中" in v.report_md() and "# 草稿" in v.report_md()
    issue = {"source": "rule", "severity": "high", "location": "RE24", "message": "ΔRE +0.999 不對"}
    v.apply(ev("critic", "critique", "❌ 審查未通過", "…", passed=False, issues=[issue], round=0))
    v.apply(ev("synthesizer", "thought", "依審查意見修正報告（第 1 次）", "…", round=1))
    assert v.report == ""                                     # 新的一版從頭串流
    v.apply(ev("critic", "critique", "✅ 審查通過", "", passed=True, issues=[], round=1))
    assert "第 1 版：❌ 退回" in v.review_md() and "+0.999" in v.review_md() and "第 2 版：✅ 通過" in v.review_md()
    v.apply(ev("system", "final", "分析完成", "# 最終報告", passed=True, report_path="r.md", trace_path="t.json"))
    assert v.report_md() == "# 最終報告" and v.files() == ["r.md", "t.json"] and v.finished


def test_view_aborted_run():
    v = RunView()
    v.apply(ev("system", "final", "分析中止", "讀不到影片", aborted=True))
    assert v.files() == [] and "讀不到影片" in v.report_md()


def test_full_run_through_view(video, tmp_path, monkeypatch):  # noqa: F811
    track, t, frames, _ = make_pitcher(n_pitches=2, stride=2)
    patch_scan(monkeypatch, [track], frames)
    events, _ = run(video, tmp_path, demo_fault=True)
    v = RunView()
    for e in events:
        v.apply(e)
    assert v.finished and v.final["passed"]
    assert all(r.status in ("ok", "degraded") for r in v.rows), [(r.tool, r.status) for r in v.rows]
    assert [r.step for r in v.rows][:2] == ["r1", "r2"] and len(v.rows) == 10
    assert v.keyframes and all(Path(p).exists() for p, _ in v.keyframes)
    assert v.curves and Path(v.curves).exists()               # segment_motions 畫的動作曲線
    assert len(v.reviews) == 2 and "❌ 退回" in v.reviews[0]
    assert not any(m["metadata"]["status"] == "pending" for m in v.messages)


def test_curves_drawn_even_without_pitch(video, tmp_path, monkeypatch):  # noqa: F811
    n = 315
    kpts = np.tile(np.linspace([400, 300], [420, 700], 17), (n, 1, 1))
    still = PoseTrack(1, np.tile([380.0, 280, 440, 720], (n, 1)), np.full(n, 0.9), kpts, np.full((n, 17), 0.9))
    patch_scan(monkeypatch, [still], np.arange(0, 2 * n, 2))
    events, _ = run(video, tmp_path)
    seg = next(e for e in events if e.kind == "tool_result" and e.payload["tool"] == "segment_motions")
    assert seg.payload["status"] == "failed" and Path(seg.payload["artifacts"][0]).exists()


def test_app_builds():
    pytest.importorskip("gradio")
    from bbagent.ui.app import build_app, make_game, preview_situation

    assert build_app() is not None
    g = make_game(9, "top", 2, ["1", "3"], 3, 2, -1.0, "defense", "batter", "", "L", "")
    assert g.on_first and g.on_third and not g.on_second and g.pitcher_hand is None and g.score_diff == -1
    assert g.analysis_target == "batter" and g.batter_hand == "L"
    assert "分析打者" in preview_situation(9, "top", 2, [], 0, 0, 0, "defense", "batter", "R", "L", "")
