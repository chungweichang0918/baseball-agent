"""啟動 Gradio 介面。

用法：
    python scripts/app.py                  # 本機開啟 http://127.0.0.1:7860
    python scripts/app.py --share          # 產生公開連結（Colab 也用這個）
    python scripts/app.py --host 0.0.0.0   # 讓同一個區網的其他裝置連線
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bbagent.vision import device  # noqa: E402,F401  必須在 torch 之前 import

from bbagent.ui.app import launch  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="產生 gradio.live 公開連結")
    p.add_argument("--root-path", default=None, help="反向代理的路徑前綴，例如 /baseball（子網域部署可省略）")
    p.add_argument("--max-file-size", default="200mb", help="單支上傳影片上限，預設 200mb")
    args = p.parse_args()
    launch(args.host, args.port, args.share, root_path=args.root_path, max_file_size=args.max_file_size)


if __name__ == "__main__":
    main()
