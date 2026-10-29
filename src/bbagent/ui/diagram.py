"""局面圖：壘包、球數、出局數、局數與比分，輸出成內嵌 SVG（用 currentColor，亮暗主題都看得清楚）。"""

from __future__ import annotations

from html import escape

from ..schemas import GameState

RUNNER = "#3b82f6"
BALL = "#22c55e"
STRIKE = "#eab308"
OUT = "#ef4444"
EMPTY_OPACITY = 0.3


def _base(cx: float, cy: float, on: bool, label: str) -> str:
    r = 17
    pts = f"{cx},{cy - r} {cx + r},{cy} {cx},{cy + r} {cx - r},{cy}"
    fill = f'fill="{RUNNER}" stroke="{RUNNER}"' if on else 'fill="none" stroke="currentColor" stroke-opacity="0.5"'
    return (f'<polygon points="{pts}" {fill} stroke-width="2"><title>{label}{"有跑者" if on else "無人"}</title>'
            f'</polygon>')


def _dots(x: float, y: float, label: str, n: int, total: int, color: str) -> str:
    parts = [f'<text x="{x}" y="{y + 5}" font-size="15" font-weight="600" fill="currentColor">{label}</text>']
    for i in range(total):
        on = i < n
        style = f'fill="{color}"' if on else f'fill="currentColor" fill-opacity="{EMPTY_OPACITY}"'
        parts.append(f'<circle cx="{x + 30 + i * 22}" cy="{y}" r="7" {style}/>')
    return "".join(parts)


def situation_svg(game: GameState) -> str:
    """回傳一段 <svg>，可直接放進 gr.HTML。"""
    if game.score_diff > 0:
        score = f"我方領先 {game.score_diff} 分"
    elif game.score_diff < 0:
        score = f"我方落後 {-game.score_diff} 分"
    else:
        score = "平手"
    role = "我方進攻" if game.our_role == "offense" else "我方防守"
    half = "▲ 上" if game.half == "top" else "▼ 下"
    who = [{"pitcher": "分析投手", "batter": "分析打者"}.get(game.analysis_target, ""),
           {"R": "右投", "L": "左投"}.get(game.pitcher_hand or "", ""),
           {"R": "右打", "L": "左打"}.get(game.batter_hand or "", "")]
    hand = "｜".join(w for w in who if w) or "自動判斷分析對象"

    # 菱形內野：本壘在下、一壘在右
    cx, cy, d = 95, 100, 58
    field = (f'<polygon points="{cx},{cy - d} {cx + d},{cy} {cx},{cy + d} {cx - d},{cy}" fill="{RUNNER}" '
             f'fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.25" stroke-width="1.5"/>')
    home = (f'<polygon points="{cx - 9},{cy + d - 6} {cx + 9},{cy + d - 6} {cx + 9},{cy + d + 2} {cx},{cy + d + 10} '
            f'{cx - 9},{cy + d + 2}" fill="currentColor" fill-opacity="0.5"/>')
    bases = (_base(cx + d, cy, game.on_first, "一壘") + _base(cx, cy - d, game.on_second, "二壘")
             + _base(cx - d, cy, game.on_third, "三壘"))
    x0 = 200
    counts = (_dots(x0, 52, "B", game.balls, 3, BALL) + _dots(x0, 82, "S", game.strikes, 2, STRIKE)
              + _dots(x0, 112, "O", game.outs, 2, OUT))
    info = (f'<text x="{x0}" y="22" font-size="17" font-weight="700" fill="currentColor">{game.inning} 局 {half}</text>'
            f'<text x="{x0}" y="150" font-size="13" fill="currentColor">{escape(score)}</text>'
            f'<text x="{x0}" y="170" font-size="13" fill="currentColor" fill-opacity="0.75">'
            f'{role}｜{hand}</text>')
    desc = escape(game.describe_zh())
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 330 185" role="img" aria-label="{desc}" '
            f'style="width:100%;max-width:360px;display:block;margin:auto;font-family:inherit">'
            f'<title>{desc}</title>{field}{home}{bases}{counts}{info}</svg>')
