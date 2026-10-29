"""檢查開發環境是否就緒。

用法：
    python scripts/check_env.py                # 套件、裝置、設定、權重、金鑰
    python scripts/check_env.py --list-models  # 另外列出 Gemini 可用型號
    python scripts/check_env.py --ping-gemini  # 另外對 fast_model 與 pro_model 各實際呼叫一次 API
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bbagent.vision import device  # noqa: E402  必須在 torch 之前 import
from bbagent.config import gemini_api_key, load_config, model_path  # noqa: E402

PACKAGES = [
    "torch", "ultralytics", "opencv-python", "numpy", "pandas", "scipy",
    "matplotlib", "gradio", "google-genai", "pydantic", "python-dotenv", "pyyaml",
]

failures: list[str] = []


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def bad(msg: str) -> None:
    print(f"  ❌ {msg}")
    failures.append(msg)


def warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")


def check_python() -> None:
    print("[Python]")
    print(f"  {sys.version.split()[0]}｜{platform.system()} {platform.machine()}")
    if sys.version_info < (3, 11):
        bad("需要 Python 3.11 以上")
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        warn("目前的 Python 不是 arm64 版本（可能在 Rosetta 下執行），將無法使用 MPS")


def check_packages() -> None:
    print("[套件]")
    for name in PACKAGES:
        try:
            ok(f"{name} {md.version(name)}")
        except md.PackageNotFoundError:
            bad(f"{name} 未安裝")


def check_torch() -> None:
    print("[PyTorch 裝置]")
    try:
        import torch
    except ImportError:
        bad("無法 import torch")
        return
    print(f"  MPS built={torch.backends.mps.is_built()} available={torch.backends.mps.is_available()}")
    print(f"  CUDA available={torch.cuda.is_available()}")
    cfg_device = load_config()["device"]
    try:
        dev = device.resolve_device(cfg_device)
    except RuntimeError as e:
        bad(str(e))
        return
    if platform.system() == "Darwin" and dev != "mps":
        warn(f"在 macOS 上沒有選到 mps（目前為 {dev}），YOLO 會很慢")
    # 用一次矩陣乘法確認裝置真的能運算
    x = torch.randn(1024, 1024, device=dev)
    t0 = time.perf_counter()
    for _ in range(10):
        y = x @ x
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()
    ok(f"在 {dev} 上完成矩陣乘法測試（{(time.perf_counter() - t0) * 1000:.1f} ms，y.sum={y.sum().item():.1f}）")


def check_config_and_weights() -> None:
    print("[設定與權重]")
    cfg = load_config()
    ok(f"config.yaml 載入成功（device={cfg['device']}）")
    for kind in ("detector", "pose"):
        p = model_path(kind)
        if p.exists():
            ok(f"{kind}: {p.name}（{p.stat().st_size / 1e6:.1f} MB）")
        else:
            warn(f"{kind}: {p.name} 尚未下載，請執行 python scripts/download_models.py")


def describe_api_error(e: Exception) -> str:
    """把冗長的 API 錯誤濃縮成一行；額度錯誤另外說明原因。"""
    msg = str(e)
    if "RESOURCE_EXHAUSTED" in msg:
        if "free_tier" in msg and "limit: 0" in msg:
            return "429：免費層級不提供此型號（額度為 0），請改用其他型號或升級付費層級"
        return "429：額度已用完，請稍後再試，或到 https://ai.dev/rate-limit 查看用量"
    return msg.splitlines()[0][:300]


def check_gemini(list_models: bool, ping: bool) -> None:
    print("[Gemini]")
    key = gemini_api_key()
    if not key:
        bad("找不到 GEMINI_API_KEY，請複製 .env.example 為 .env 並填入金鑰")
        return
    ok("GEMINI_API_KEY 已設定")
    if not (list_models or ping):
        return
    from google import genai

    client = genai.Client(api_key=key)
    llm_cfg = load_config()["llm"]
    if list_models:
        names = sorted(
            m.name.removeprefix("models/")
            for m in client.models.list()
            if "gemini" in m.name and "generateContent" in (m.supported_actions or [])
        )
        print("  可用的 Gemini 型號：")
        for n in names:
            mark = "  ← config" if n in (llm_cfg["fast_model"], llm_cfg["pro_model"]) else ""
            print(f"    - {n}{mark}")
        for role in ("fast_model", "pro_model"):
            if llm_cfg[role] not in names:
                warn(f"config 中的 {role}={llm_cfg[role]} 不在可用清單內，請更新 config.yaml")
    if ping:
        for model in dict.fromkeys((llm_cfg["fast_model"], llm_cfg["pro_model"])):
            try:
                resp = client.models.generate_content(model=model, contents="只回覆兩個字：收到")
                ok(f"{model} 回應：{resp.text.strip()}")
            except Exception as e:  # noqa: BLE001  任何 API 錯誤都只需回報
                bad(f"呼叫 {model} 失敗：{describe_api_error(e)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-models", action="store_true", help="列出 Gemini 可用型號")
    parser.add_argument("--ping-gemini", action="store_true", help="實際呼叫 fast_model 與 pro_model 各一次")
    args = parser.parse_args()

    check_python()
    check_packages()
    check_torch()
    check_config_and_weights()
    check_gemini(args.list_models, args.ping_gemini)

    print()
    if failures:
        print(f"共 {len(failures)} 項未通過。")
        return 1
    print("環境檢查全部通過。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
