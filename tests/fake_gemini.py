"""假的 google-genai 客戶端：依序回傳預先準備的回覆（或丟出錯誤），並記錄每次呼叫。"""

from __future__ import annotations

from types import SimpleNamespace

from google.genai import errors


def api_error(code: int, message: str = "error", status: str = "ERROR") -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": message, "status": status}})


class FakeResponse:
    def __init__(self, text: str):
        self.text = text
        self.usage_metadata = SimpleNamespace(prompt_token_count=100, candidates_token_count=len(text),
                                              thoughts_token_count=None)


class FakeModels:
    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def _next(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def generate_content(self, *, model, contents, config):
        return FakeResponse(self._next(model=model, contents=contents, config=config))

    def generate_content_stream(self, *, model, contents, config):
        text = self._next(model=model, contents=contents, config=config)   # 錯誤在第一次 next() 時丟出
        for i in range(0, len(text), 5):
            yield FakeResponse(text[i:i + 5])


class FakeSDK:
    def __init__(self, replies: list):
        self.models = FakeModels(replies)
