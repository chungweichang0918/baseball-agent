"""scout_overview：讓 Gemini 看幾張均勻取樣的畫面，判斷鏡頭視角、主體與拍攝條件。

這是 Agent 的第一步。Planner 依它的結果決定分析分支、挑選適用的指標，
並把拍攝條件的問題（例如鏡頭移動、不是側拍）寫進報告的限制。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..llm.gemini_client import GeminiClient, GeminiError, ImageInput, get_client, model_name
from ..llm.prompts import load_prompt
from ..schemas import CameraView, ToolResult
from ..vision.render import read_frames
from ..vision.video import VideoMeta, probe_video

THUMB_POSITIONS = (0.1, 0.37, 0.63, 0.9)   # 取樣位置（影片長度的比例），避開頭尾
THUMB_MAX_SIDE = 768

VIEW_ZH = {"side": "正側面", "center_field": "中外野轉播視角", "behind": "後方",
           "front": "前方或斜前方", "other": "其他"}
SUBJECT_ZH = {"pitcher": "投手", "batter": "打者", "both": "投手與打者", "unknown": "無法判斷"}


class ScoreboardInfo(BaseModel):
    visible: bool
    pitch_speed_kmh: float | None = None
    pitch_type: str | None = None
    balls: int | None = None
    strikes: int | None = None
    outs: int | None = None
    inning: str | None = None


class VideoOverview(BaseModel):
    camera_view: Literal["side", "center_field", "behind", "front", "other"]
    view_reason: str
    subject: Literal["pitcher", "batter", "both", "unknown"]
    camera_moving: bool
    full_body_visible: bool
    is_live_action: bool
    scoreboard: ScoreboardInfo
    concerns: list[str]

    @property
    def plan_view(self) -> CameraView:
        """對應到分析流程使用的視角分類。"""
        return self.camera_view if self.camera_view in ("side", "center_field") else "other"


def sample_thumbnails(video: str | Path, meta: VideoMeta) -> list[tuple[float, ImageInput]]:
    indices = [min(meta.frame_count - 1, round(p * meta.frame_count)) for p in THUMB_POSITIONS]
    frames = read_frames(video, indices)
    return [(i / meta.fps, ImageInput.from_array(frames[i], THUMB_MAX_SIDE)) for i in indices if i in frames]


def scout_overview(video: str | Path, meta: VideoMeta | None = None, client: GeminiClient | None = None,
                   model: str | None = None) -> ToolResult:
    if meta is None:
        try:
            meta = probe_video(video)
        except (FileNotFoundError, ValueError) as e:  # 工具不丟例外，回傳失敗讓 Planner 處理
            return ToolResult(status="failed", summary="讀不到影片", error=str(e))
    client = client or get_client()
    model = model or model_name("fast")
    thumbs = sample_thumbnails(video, meta)
    if not thumbs:
        return ToolResult(status="failed", summary="讀不到影片畫面", error=f"無法從 {video} 取樣畫面")

    contents: list = [f"影片長度 {meta.duration_s:.1f} 秒，{meta.fps:.0f} fps，{meta.width}x{meta.height}。"
                      f"以下是依時間順序取出的 {len(thumbs)} 張畫面。"]
    for i, (t, img) in enumerate(thumbs, start=1):
        contents += [f"畫面 {i}（t={t:.1f}s）", img]
    try:
        ov, resp = client.generate_structured(model, contents, VideoOverview, system=load_prompt("scout_overview"))
    except GeminiError as e:
        return ToolResult(status="failed", summary="影片概況判斷失敗", error=str(e))

    lines = [
        f"視角：{VIEW_ZH[ov.camera_view]}（{ov.view_reason}）",
        f"主體：{SUBJECT_ZH[ov.subject]}｜全身入鏡：{'是' if ov.full_body_visible else '否'}"
        f"｜鏡頭移動：{'是' if ov.camera_moving else '否'}｜現場即時畫面：{'是' if ov.is_live_action else '否'}",
    ]
    sb = ov.scoreboard
    if sb.visible:
        info = [f"球速 {sb.pitch_speed_kmh:g} km/h" if sb.pitch_speed_kmh is not None else None,
                f"球種 {sb.pitch_type}" if sb.pitch_type else None,
                f"球數 {sb.balls}-{sb.strikes}" if sb.balls is not None and sb.strikes is not None else None,
                f"{sb.outs} 出局" if sb.outs is not None else None,
                sb.inning]
        lines.append("記分板：" + ("、".join(x for x in info if x) or "可見但讀不出資訊"))
    lines += [f"⚠️ {c}" for c in ov.concerns]

    return ToolResult(
        status="ok",
        summary="\n".join(lines),
        data=ov.model_dump() | {
            "plan_view": ov.plan_view,
            "thumbnail_times_s": [round(t, 2) for t, _ in thumbs],
            "llm": {"model": resp.model, "cached": resp.cached, "latency_s": resp.latency_s, "usage": resp.usage},
        },
    )
