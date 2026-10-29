"""Run the analysis on this computer and optionally update the GitHub Pages demo URL."""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def config_text(url: str) -> str:
    return json.dumps({"analysis_url": url}, ensure_ascii=False, indent=2) + "\n"


def write_config(root: Path, url: str) -> None:
    path = root / "docs/baseball/config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(config_text(url), encoding="utf-8")
    temporary.replace(path)


def publish_url(repo: str, url: str, *, expected: str | None = None) -> bool:
    """Update only this site's config; avoid clearing a newer demo on shutdown."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("GitHub 專案需為 owner/repository")
    endpoint = f"repos/{repo}/contents/docs/baseball/config.json"
    read = subprocess.run(["gh", "api", endpoint], capture_output=True, text=True, encoding="utf-8")
    if read.returncode:
        raise RuntimeError("無法讀取遠端網站設定；請先登入 GitHub 並發布網站來源")
    current = json.loads(read.stdout)
    if expected is not None:
        value = json.loads(base64.b64decode(current["content"]).decode("utf-8"))
        if value.get("analysis_url") != expected:
            return False
    payload = {"message": "Update local baseball demo connection", "sha": current["sha"],
               "content": base64.b64encode(config_text(url).encode("utf-8")).decode("ascii")}
    update = subprocess.run(["gh", "api", "--method", "PUT", endpoint, "--input", "-"],
                            input=json.dumps(payload), capture_output=True, text=True, encoding="utf-8")
    if update.returncode:
        raise RuntimeError("網站設定更新失敗；本機分析仍持續執行，請重新發布 config.json")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="透過已登入的 gh 更新網站連線設定")
    parser.add_argument("--repo", default="chungweichang0918/baseball-agent")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    from bbagent.ui.app import build_app, launch_options

    demo = build_app()
    demo.queue(default_concurrency_limit=1)
    url, published = None, False
    try:
        _, local_url, share_url = demo.launch(server_name="127.0.0.1", server_port=args.port,
            share=True, prevent_thread_lock=True, max_file_size="200mb", **launch_options(demo))
        if not share_url:
            print(f"無法取得公開展示連結。本機仍可使用：{local_url}", flush=True)
        else:
            url = share_url
            write_config(ROOT, url)
            print(f"\n展示連結：{url}\n請保持此視窗、電腦與網路開啟；Ctrl+C 結束展示。", flush=True)
            if args.publish:
                try:
                    published = publish_url(args.repo, url)
                    print("網站連線設定已更新；請等待 GitHub Pages 重新部署。", flush=True)
                except (OSError, RuntimeError, ValueError) as e:
                    print(f"尚未更新線上網站：{e}", flush=True)
            print("本機已更新 docs/baseball/config.json。臨時連結會過期，重新啟動需更新網站設定。", flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n結束展示。", flush=True)
        return 0
    finally:
        if url:
            write_config(ROOT, "")
            if published:
                try:
                    publish_url(args.repo, "", expected=url)
                except (OSError, RuntimeError, ValueError) as e:
                    print(f"請手動清除線上舊連結：{e}", flush=True)
        demo.close()


if __name__ == "__main__":
    raise SystemExit(main())
