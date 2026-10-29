import json

import cv2
import numpy as np
import pytest
from fake_gemini import FakeSDK, api_error
from google.genai import types

from bbagent.llm.gemini_client import GeminiClient
from bbagent.tools.gemini_inspect import MAX_IMAGES, gemini_inspect
from bbagent.tools.scout_overview import VideoOverview, scout_overview

OVERVIEW = {
    "camera_view": "front", "view_reason": "投手正面朝向鏡頭", "subject": "pitcher",
    "camera_moving": True, "full_body_visible": True, "is_live_action": True,
    "scoreboard": {"visible": True, "pitch_speed_kmh": 92, "pitch_type": None, "balls": None,
                   "strikes": None, "outs": None, "inning": None},
    "concerns": ["鏡頭從斜前方拍攝，出手時間會不準"],
}
INSPECT = {
    "answer": "手臂在出手時接近垂直。",
    "observations": [
        {"aspect": "手臂角度", "finding": "高壓投法", "confidence": "medium", "images": [2, 3, 99]},
        {"aspect": "手套", "finding": "看不清楚", "confidence": "low", "images": []},
    ],
    "limitations": ["只有一球"],
}


def client_with(tmp_path, replies):
    return GeminiClient(cache_dir=tmp_path, cache_mode="off", sdk_client=FakeSDK(replies), sleep=lambda s: None)


def image_parts(call) -> int:
    return sum(isinstance(c, types.Part) for c in call["contents"])


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(60):
        writer.write(np.full((48, 64, 3), i * 4, np.uint8))
    writer.release()
    return path


def test_scout_overview(tmp_path, video):
    client = client_with(tmp_path, [json.dumps(OVERVIEW)])
    r = scout_overview(video, client=client, model="m")
    assert r.status == "ok"
    assert "前方或斜前方" in r.summary and "鏡頭移動：是" in r.summary
    assert "球速 92 km/h" in r.summary and "⚠️ 鏡頭從斜前方拍攝" in r.summary
    assert r.data["plan_view"] == "other"
    assert image_parts(client._sdk.models.calls[0]) == 4


def test_scout_overview_failure(tmp_path, video):
    r = scout_overview(video, client=client_with(tmp_path, [api_error(400, "bad")]), model="m")
    assert r.status == "failed" and "400" in r.error


def test_scout_overview_missing_video(tmp_path):
    client = client_with(tmp_path, [])
    r = scout_overview(tmp_path / "nope.mp4", client=client, model="m")
    assert r.status == "failed" and "無法開啟影片" in r.error
    assert client._sdk.models.calls == []


@pytest.mark.parametrize("view,expected", [("side", "side"), ("center_field", "center_field"),
                                           ("behind", "other"), ("front", "other")])
def test_plan_view_mapping(view, expected):
    assert VideoOverview(**(OVERVIEW | {"camera_view": view})).plan_view == expected


def test_gemini_inspect_subsamples_and_cleans_refs(tmp_path):
    client = client_with(tmp_path, [json.dumps(INSPECT)])
    imgs = [np.full((100, 80, 3), i * 20, np.uint8) for i in range(10)]
    r = gemini_inspect(imgs, "出手時的手臂角度", [f"幀 {i}" for i in range(10)], context="右投", client=client, model="m")
    assert r.status == "ok"
    call = client._sdk.models.calls[0]
    assert image_parts(call) == MAX_IMAGES
    assert call["contents"][0] == "觀察重點：出手時的手臂角度" and call["contents"][1].startswith("背景資訊")
    assert r.data["observations"][0]["images"] == [2, 3]          # 99 超出範圍被剔除
    assert len(r.data["captions"]) == MAX_IMAGES and r.data["captions"][0] == "幀 0"
    assert "信心中" in r.summary and "只送出均勻取樣的 8 張" in r.summary


def test_gemini_inspect_input_checks(tmp_path):
    assert gemini_inspect([], "x", client=client_with(tmp_path, [])).status == "failed"
    with pytest.raises(ValueError):
        gemini_inspect([np.zeros((10, 10, 3), np.uint8)], "x", ["a", "b"], client=client_with(tmp_path, []))
