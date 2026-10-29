"""Deployment parameters must reach Gradio without losing existing launch options."""
from bbagent.ui import app


def test_proxy_prefix_upload_limit_and_existing_options(monkeypatch):
    class Demo:
        def queue(self, **kwargs):
            self.queue_options = kwargs

        def launch(self, **kwargs):
            self.options = kwargs

    demo = Demo()
    monkeypatch.setattr(app, "build_app", lambda: demo)
    monkeypatch.setattr(app, "launch_options", lambda _: {"allowed_paths": ["/private/runs"], "css": "studio-css"})
    app.launch(root_path="/baseball", max_file_size="100mb")
    assert demo.options == {"server_name": "127.0.0.1", "server_port": 7860, "share": False,
                            "root_path": "/baseball", "max_file_size": "100mb",
                            "allowed_paths": ["/private/runs"], "css": "studio-css"}
    assert demo.queue_options == {"default_concurrency_limit": 1}
