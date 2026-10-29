"""YOLO 結果的磁碟快取。鍵 = 影片內容雜湊 + 參數，值存成 npz。

調整分段門檻、重跑 Agent 或 Critic 退回修正時，都不必重跑 YOLO。
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import load_config, project_path


@lru_cache(maxsize=32)
def _sha1(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha1(path: str | Path) -> str:
    """影片內容的雜湊。以檔案大小與修改時間當快取鍵，同一支影片不重複計算。"""
    st = Path(path).stat()
    return _sha1(str(Path(path).resolve()), st.st_size, st.st_mtime_ns)


def cache_path(kind: str, video: str | Path, params: dict[str, Any]) -> Path:
    key_src = file_sha1(video) + json.dumps(params, sort_keys=True)
    key = hashlib.sha1(key_src.encode()).hexdigest()[:16]
    folder = project_path(load_config()["paths"]["cache"]) / kind
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{Path(video).stem}_{key}.npz"
