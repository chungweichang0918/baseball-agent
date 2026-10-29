"""Gemini API 的薄包裝：重試、結構化輸出、串流、回應快取（錄製／重播）。

快取模式（config.llm.cache_mode）：
  auto    有快取就重播，沒有才呼叫 API 並存檔。同樣的輸入不重複消耗額度（預設）
  off     一律呼叫 API，不讀也不寫快取
  replay  只能重播，沒有快取就報錯。測試或離線展示用，保證不連網
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import cv2
import numpy as np
from pydantic import BaseModel, ValidationError

from ..config import gemini_api_key, load_config, project_path
from ..vision.imageio import imread

T = TypeVar("T", bound=BaseModel)

RETRYABLE_CODES = {429, 500, 502, 503, 504}
BACKOFF_BASE_S = 2.0
CACHE_MODES = ("auto", "off", "replay")
REPLAY_CHUNK_CHARS = 24     # 重播串流時每次吐出的字數，讓 UI 仍有逐字出現的效果


class GeminiError(RuntimeError):
    pass


class QuotaUnavailableError(GeminiError):
    """免費層級不提供此型號（額度為 0），重試也沒用。"""


@dataclass(frozen=True)
class ImageInput:
    data: bytes
    mime_type: str = "image/jpeg"

    @classmethod
    def from_array(cls, img: np.ndarray, max_side: int = 1024, quality: int = 85) -> ImageInput:
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("圖片編碼失敗")
        return cls(buf.tobytes())

    @classmethod
    def from_path(cls, path: str | Path, max_side: int = 1024) -> ImageInput:
        img = imread(path)
        if img is None:
            raise FileNotFoundError(f"讀不到圖片：{path}")
        return cls.from_array(img, max_side)


Content = str | ImageInput


@dataclass
class LLMResponse:
    text: str
    model: str
    cached: bool
    latency_s: float
    usage: dict[str, int] = field(default_factory=dict)
    first_chunk_s: float | None = None    # 串流時第一段文字出現的時間，用來分辨「排隊久」還是「生成慢」


class GeminiClient:
    def __init__(self, api_key: str | None = None, cache_mode: str | None = None,
                 cache_dir: str | Path | None = None, max_retries: int | None = None,
                 timeout_s: float | None = None, thinking_level: str | None = None, sdk_client: Any = None,
                 sleep: Callable[[float], None] = time.sleep,
                 on_retry: Callable[[int, float, str], None] | None = None):
        """on_retry(第幾次重試, 等待秒數, 原因)：每次重試前呼叫，讓 UI 或 CLI 顯示進度。

        thinking_level：minimal / low / medium / high，只適用 Gemini 3 系列；None 表示用模型預設。
        """
        cfg = load_config()
        llm = cfg["llm"]
        self.cache_mode = cache_mode or llm.get("cache_mode", "auto")
        if self.cache_mode not in CACHE_MODES:
            raise ValueError(f"cache_mode 必須是 {CACHE_MODES} 之一")
        self.cache_dir = Path(cache_dir) if cache_dir else project_path(cfg["paths"]["cache"]) / "llm"
        self.max_retries = llm["max_retries"] if max_retries is None else max_retries
        self.timeout_s = timeout_s if timeout_s is not None else llm.get("timeout_s", 90)
        self.thinking_level = thinking_level or llm.get("thinking_level")
        self._api_key = api_key
        self._sdk = sdk_client
        self._sleep = sleep
        self.on_retry = on_retry
        self.last_response: LLMResponse | None = None
        self.stats = {"api_calls": 0, "cache_hits": 0}   # 給執行摘要統計用

    # ---------- 公開介面 ----------

    def generate(self, model: str, contents: list[Content], *, system: str | None = None,
                 schema: type[BaseModel] | None = None, temperature: float = 0.2) -> LLMResponse:
        key = self._key(model, contents, system, schema, temperature)
        cached = self._read_cache(key)
        if cached:
            self.stats["cache_hits"] += 1
            self.last_response = cached
            return cached
        t0 = time.perf_counter()
        resp = self._with_retry(lambda: self._sdk_client().models.generate_content(
            model=model, contents=self._to_sdk(contents), config=self._config(system, schema, temperature)))
        self.stats["api_calls"] += 1
        result = LLMResponse(resp.text or "", model, False, round(time.perf_counter() - t0, 2), _usage(resp))
        self._write_cache(key, result)
        self.last_response = result
        return result

    def generate_structured(self, model: str, contents: list[Content], schema: type[T], *,
                            system: str | None = None, temperature: float = 0.2) -> tuple[T, LLMResponse]:
        resp = self.generate(model, contents, system=system, schema=schema, temperature=temperature)
        try:
            return schema.model_validate_json(resp.text), resp
        except ValidationError as e:
            raise GeminiError(f"模型回傳的 JSON 不符合 {schema.__name__} 格式：{e.errors()[:3]}\n原文：{resp.text[:500]}") from e

    def stream(self, model: str, contents: list[Content], *, system: str | None = None,
               temperature: float = 0.4) -> Iterator[str]:
        """逐段產生文字。結束後 self.last_response 會有完整結果與用量。"""
        key = self._key(model, contents, system, None, temperature)
        cached = self._read_cache(key)
        if cached:
            self.stats["cache_hits"] += 1
            for i in range(0, len(cached.text), REPLAY_CHUNK_CHARS):
                yield cached.text[i:i + REPLAY_CHUNK_CHARS]
            self.last_response = cached
            return

        t0 = time.perf_counter()
        chunks = self._with_retry(lambda: self._first_chunk(model, contents, system, temperature))
        self.stats["api_calls"] += 1
        first_chunk_s = round(time.perf_counter() - t0, 2)
        parts: list[str] = []
        last = None
        for chunk in chunks:
            last = chunk
            if chunk.text:
                parts.append(chunk.text)
                yield chunk.text
        result = LLMResponse("".join(parts), model, False, round(time.perf_counter() - t0, 2), _usage(last),
                             first_chunk_s)
        self._write_cache(key, result)
        self.last_response = result

    # ---------- 內部 ----------

    def _sdk_client(self):
        if self._sdk is None:
            key = self._api_key or gemini_api_key()
            if not key:
                raise GeminiError("找不到 GEMINI_API_KEY，請在 .env 中設定")
            from google import genai
            from google.genai import types
            self._sdk = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)))
        return self._sdk

    def _first_chunk(self, model, contents, system, temperature) -> Iterator:
        """先取出第一段再回傳迭代器，讓連線錯誤能在重試範圍內被抓到。"""
        it = iter(self._sdk_client().models.generate_content_stream(
            model=model, contents=self._to_sdk(contents), config=self._config(system, None, temperature)))
        first = next(it, None)

        def chained():
            if first is not None:
                yield first
            yield from it
        return chained()

    @staticmethod
    def _to_sdk(contents: list[Content]) -> list:
        from google.genai import types
        return [types.Part.from_bytes(data=c.data, mime_type=c.mime_type) if isinstance(c, ImageInput) else c
                for c in contents]

    def _config(self, system: str | None, schema: type[BaseModel] | None, temperature: float):
        from google.genai import types
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if schema else None,
            response_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level) if self.thinking_level else None,
        )

    def _with_retry(self, fn: Callable[[], Any]) -> Any:
        import httpx
        from google.genai import errors
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except errors.APIError as e:
                if e.code == 429 and "limit: 0" in str(e):
                    raise QuotaUnavailableError(
                        "免費層級不提供此型號（額度為 0），請在 config.yaml 改用其他型號或升級付費層級") from e
                if e.code not in RETRYABLE_CODES or attempt == self.max_retries:
                    raise GeminiError(f"Gemini API 錯誤 {e.code} {e.status}：{e.message}") from e
                reason = f"{e.code} {e.status}"
            except httpx.TimeoutException as e:
                if attempt == self.max_retries:
                    raise GeminiError(f"Gemini API 連續 {attempt + 1} 次逾時（每次 {self.timeout_s:g} 秒）") from e
                reason = f"逾時（超過 {self.timeout_s:g} 秒）"
            wait = BACKOFF_BASE_S * 2 ** attempt
            if self.on_retry:
                self.on_retry(attempt + 1, wait, reason)
            self._sleep(wait)
        raise AssertionError("unreachable")

    def _key(self, model: str, contents: list[Content], system: str | None,
             schema: type[BaseModel] | None, temperature: float) -> str:
        parts = [{"image": hashlib.sha256(c.data).hexdigest()} if isinstance(c, ImageInput) else {"text": c}
                 for c in contents]
        payload = {
            "model": model, "system": system, "temperature": temperature, "parts": parts,
            "thinking_level": self.thinking_level,
            "schema": schema.model_json_schema() if schema else None,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]

    def _read_cache(self, key: str) -> LLMResponse | None:
        if self.cache_mode == "off":
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            if self.cache_mode == "replay":
                raise GeminiError(f"重播模式下找不到快取的回應（{key}），請先用 auto 模式錄製")
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse(d["text"], d["model"], True, 0.0, d.get("usage", {}))

    def _write_cache(self, key: str, resp: LLMResponse) -> None:
        if self.cache_mode != "auto":
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        record = {"model": resp.model, "text": resp.text, "usage": resp.usage,
                  "latency_s": resp.latency_s, "created_at": datetime.now().isoformat(timespec="seconds")}
        (self.cache_dir / f"{key}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def _usage(resp: Any) -> dict[str, int]:
    meta = getattr(resp, "usage_metadata", None)
    if meta is None:
        return {}
    fields = {"prompt_tokens": "prompt_token_count", "output_tokens": "candidates_token_count",
              "thinking_tokens": "thoughts_token_count"}
    return {k: int(getattr(meta, v)) for k, v in fields.items() if getattr(meta, v, None) is not None}


@lru_cache(maxsize=1)
def get_client() -> GeminiClient:
    return GeminiClient()


def model_name(role: str) -> str:
    """role 為 "fast" 或 "pro"，對應 config.llm 中的型號。"""
    return load_config()["llm"][f"{role}_model"]
