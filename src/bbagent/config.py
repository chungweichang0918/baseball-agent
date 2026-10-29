"""載入 config.yaml 與 .env，提供全專案共用的設定與路徑。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(rel: str | os.PathLike) -> Path:
    """把 config 中的相對路徑轉成以專案根目錄為基準的絕對路徑。"""
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


def model_path(kind: str) -> Path:
    """kind 為 config.models 中的鍵，例如 "detector" 或 "pose"。"""
    models = load_config()["models"]
    return project_path(models["dir"]) / models[kind]


def gemini_api_key() -> str | None:
    load_config()  # 確保 .env 已載入
    return os.getenv("GEMINI_API_KEY") or None
