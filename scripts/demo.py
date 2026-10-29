"""Run the analysis on this computer and optionally update the GitHub Pages demo URL."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
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


def github_api(endpoint: str, payload: dict | None = None) -> dict:
    """Prefer gh; fall back to the same Git credential helper used by git push."""
    command = ["gh", "api", endpoint]
    if payload is not None:
        command += ["--method", "PUT", "--input", "-"]
    try:
        result = subprocess.run(command, input=None if payload is None else json.dumps(payload),
                                capture_output=True, text=True, encoding="utf-8")
        if result.returncode == 0:
            return json.loads(result.stdout) if result.stdout.strip() else {}
    except FileNotFoundError:
        pass
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
    stored = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                             capture_output=True, text=True, encoding="utf-8", env=environment)
    fields = dict(line.split("=", 1) for line in stored.stdout.splitlines() if "=" in line)
    if stored.returncode or not fields.get("password") or not fields.get("username"):
        raise RuntimeError("請先登入 GitHub CLI 或 Git Credential Manager")
    authorization = base64.b64encode((fields["username"] + ":" + fields["password"]).encode()).decode()
    request = urllib.request.Request("https://api.github.com/" + endpoint,
        method="GET" if payload is None else "PUT",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Authorization": "Basic " + authorization, "Accept": "application/vnd.github+json",
                 "User-Agent": "Diamond-Studio-Deployment", "X-GitHub-Api-Version": "2022-11-28"})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"GitHub 更新失敗（HTTP {error.code}）") from None


def publish_url(repo: str, url: str, *, expected: str | None = None) -> bool:
    """Update only this site's config; avoid clearing a newer demo on shutdown."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("GitHub 專案需為 owner/repository")
    endpoint = f"repos/{repo}/contents/docs/baseball/config.json"
    current = github_api(endpoint)
    if expected is not None:
        value = json.loads(base64.b64decode(current["content"]).decode("utf-8"))
        if value.get("analysis_url") != expected:
            return False
    payload = {"message": "Update local baseball demo connection", "sha": current["sha"],
               "content": base64.b64encode(config_text(url).encode("utf-8")).decode("ascii")}
    github_api(endpoint, payload)
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
