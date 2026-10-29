import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("demo_hosting", Path(__file__).parents[1] / "scripts/demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_connection_file_can_be_cleared(tmp_path):
    demo.write_config(tmp_path, "https://test.gradio.live")
    path = tmp_path / "docs/baseball/config.json"
    assert json.loads(path.read_text())["analysis_url"] == "https://test.gradio.live"
    demo.write_config(tmp_path, "")
    assert json.loads(path.read_text())["analysis_url"] == ""


def test_shutdown_preserves_a_newer_session(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        data = {"sha": "revision", "content": base64.b64encode(demo.config_text("https://new.gradio.live").encode()).decode()}
        return SimpleNamespace(returncode=0, stdout=json.dumps(data))
    monkeypatch.setattr(demo.subprocess, "run", run)
    assert not demo.publish_url("owner/repo", "", expected="https://old.gradio.live")
    assert len(calls) == 1


def test_publication_uses_revision_and_structured_body(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"sha": "revision"}))
    monkeypatch.setattr(demo.subprocess, "run", run)
    assert demo.publish_url("owner/repo", "https://demo.gradio.live")
    payload = json.loads(calls[1][1]["input"])
    assert payload["sha"] == "revision"
    assert json.loads(base64.b64decode(payload["content"]))["analysis_url"] == "https://demo.gradio.live"
