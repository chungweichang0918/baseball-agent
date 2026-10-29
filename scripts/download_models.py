"""下載 config.yaml 指定的 YOLO 權重到 models/。

用法：
    python scripts/download_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bbagent.config import model_path  # noqa: E402
from bbagent.vision.yolo import load_yolo  # noqa: E402


def main() -> int:
    for kind in ("detector", "pose"):
        path = model_path(kind)
        status = "已存在" if path.exists() else "下載中"
        print(f"[{kind}] {path.name}：{status}")
        model = load_yolo(kind)
        if not path.exists():
            print(f"  ❌ 載入成功但權重不在 {path}，請檢查 Ultralytics 的下載位置")
            return 1
        print(f"  ✅ {path}（{path.stat().st_size / 1e6:.1f} MB，task={model.task}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
