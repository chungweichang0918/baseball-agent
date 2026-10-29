"""讀取 src/bbagent/prompts/ 下的提示詞。提示詞放在 .md 檔，方便調整，也方便在報告中展示。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"找不到提示詞：{path}")
    return path.read_text(encoding="utf-8").strip()
