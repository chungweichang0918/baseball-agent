import json

import numpy as np
import pytest
from fake_gemini import FakeSDK, api_error
from pydantic import BaseModel

from bbagent.llm import gemini_client as gc
from bbagent.llm.gemini_client import GeminiClient, GeminiError, ImageInput, QuotaUnavailableError


class Answer(BaseModel):
    view: str
    score: int


GOOD = json.dumps({"view": "side", "score": 3})


def make(tmp_path, replies, **kw):
    sleeps = []
    client = GeminiClient(cache_dir=tmp_path, sdk_client=FakeSDK(replies), sleep=sleeps.append, **kw)
    return client, client._sdk.models, sleeps


def test_structured_output_and_cache(tmp_path):
    client, models, _ = make(tmp_path, [GOOD])
    ans, resp = client.generate_structured("m", ["問題"], Answer)
    assert ans == Answer(view="side", score=3) and not resp.cached
    assert resp.usage == {"prompt_tokens": 100, "output_tokens": len(GOOD)}
    assert models.calls[0]["config"].response_mime_type == "application/json"

    again, resp2 = client.generate_structured("m", ["問題"], Answer)   # 沒有剩餘回覆，只能來自快取
    assert again == ans and resp2.cached
    assert len(models.calls) == 1 and len(list(tmp_path.glob("*.json"))) == 1


def test_cache_key_depends_on_images_and_model(tmp_path):
    client, models, _ = make(tmp_path, [GOOD, GOOD, GOOD])
    img_a = ImageInput(b"aaa")
    img_b = ImageInput(b"bbb")
    client.generate("m", ["看圖", img_a])
    client.generate("m", ["看圖", img_b])
    client.generate("other", ["看圖", img_a])
    client.generate("m", ["看圖", img_a])            # 與第一次相同 → 快取
    assert len(models.calls) == 3


def test_thinking_level_is_sent_and_part_of_cache_key(tmp_path):
    client, models, _ = make(tmp_path, [GOOD, GOOD], thinking_level="low")
    client.generate("m", ["q"])
    assert models.calls[0]["config"].thinking_config.thinking_level.value.lower() == "low"
    client.thinking_level = None
    client.generate("m", ["q"])                      # 思考程度不同 → 不共用快取
    assert models.calls[1]["config"].thinking_config is None
    assert len(models.calls) == 2


def test_stream_reports_first_chunk_time(tmp_path):
    client, _, _ = make(tmp_path, ["hello world"])
    list(client.stream("m", ["q"]))
    r = client.last_response
    assert r.first_chunk_s is not None and r.first_chunk_s <= r.latency_s


def test_off_mode_always_calls(tmp_path):
    client, models, _ = make(tmp_path, [GOOD, GOOD], cache_mode="off")
    client.generate("m", ["q"])
    client.generate("m", ["q"])
    assert len(models.calls) == 2 and not list(tmp_path.glob("*.json"))


def test_replay_mode_requires_cache(tmp_path):
    client, models, _ = make(tmp_path, [GOOD], cache_mode="replay")
    with pytest.raises(GeminiError, match="重播模式"):
        client.generate("m", ["q"])
    assert models.calls == []


def test_retry_then_success(tmp_path):
    client, models, sleeps = make(tmp_path, [api_error(503), api_error(429, "rate"), GOOD], max_retries=3)
    assert client.generate("m", ["q"]).text == GOOD
    assert sleeps == [2.0, 4.0]


def test_timeout_is_retried_and_reported(tmp_path):
    import httpx
    events = []
    client, _, sleeps = make(tmp_path, [httpx.ReadTimeout("slow"), GOOD], timeout_s=30,
                             on_retry=lambda n, wait, reason: events.append((n, wait, reason)))
    assert client.generate("m", ["q"]).text == GOOD
    assert events == [(1, 2.0, "逾時（超過 30 秒）")] and sleeps == [2.0]


def test_repeated_timeouts_give_clear_error(tmp_path):
    import httpx
    client, _, _ = make(tmp_path, [httpx.ReadTimeout("slow")] * 2, max_retries=1, timeout_s=30)
    with pytest.raises(GeminiError, match="連續 2 次逾時"):
        client.generate("m", ["q"])


def test_retry_exhausted(tmp_path):
    client, _, sleeps = make(tmp_path, [api_error(503)] * 3, max_retries=2)
    with pytest.raises(GeminiError, match="503"):
        client.generate("m", ["q"])
    assert len(sleeps) == 2


def test_free_tier_zero_quota_is_not_retried(tmp_path):
    client, models, sleeps = make(tmp_path, [api_error(429, "Quota exceeded, limit: 0, model: x")])
    with pytest.raises(QuotaUnavailableError):
        client.generate("m", ["q"])
    assert sleeps == [] and len(models.calls) == 1


def test_client_errors_are_not_retried(tmp_path):
    client, _, sleeps = make(tmp_path, [api_error(400, "bad request")])
    with pytest.raises(GeminiError, match="400"):
        client.generate("m", ["q"])
    assert sleeps == []


def test_invalid_json_raises(tmp_path):
    client, _, _ = make(tmp_path, ['{"view": "side"}'])
    with pytest.raises(GeminiError, match="Answer"):
        client.generate_structured("m", ["q"], Answer)


def test_stream_records_and_replays(tmp_path):
    text = "得分期望值是指在某個壘包出局狀態下，到半局結束平均還能得幾分。"
    client, models, _ = make(tmp_path, [text])
    chunks = list(client.stream("m", ["說明 RE24"]))
    assert len(chunks) > 1 and "".join(chunks) == text
    assert not client.last_response.cached

    replay = list(client.stream("m", ["說明 RE24"]))
    assert "".join(replay) == text and client.last_response.cached
    assert len(models.calls) == 1


def test_stream_retries_before_first_chunk(tmp_path):
    client, _, sleeps = make(tmp_path, [api_error(503), "ok text"])
    assert "".join(client.stream("m", ["q"])) == "ok text"
    assert sleeps == [2.0]


def test_missing_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "gemini_api_key", lambda: None)
    client = GeminiClient(cache_dir=tmp_path, cache_mode="off")
    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        client.generate("m", ["q"])


def test_image_input_is_resized_and_jpeg():
    img = ImageInput.from_array(np.zeros((2000, 1000, 3), np.uint8), max_side=500)
    assert img.data[:2] == b"\xff\xd8" and img.mime_type == "image/jpeg"
    import cv2
    decoded = cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (500, 250)
