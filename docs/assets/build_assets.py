"""Generate the README's SVG plates (banner, committee, architecture).

All text is converted to outlines so the plates render identically on GitHub,
which blocks web fonts inside SVG images. Fonts are the app's own (Fraunces,
Geist Mono, both OFL). They are downloaded from github.com/google/fonts on
first use into docs/assets/.fonts, which is not committed.

Usage (from the repo root):
    uv run --with fonttools --with uharfbuzz python docs/assets/build_assets.py
"""
import urllib.request
from html import escape
from io import BytesIO
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

from equity_trader.config import AGENT_PERSONAS, AGENT_WEIGHTS

OUT = Path(__file__).parent
W = 1280

_FONT_REPO = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
FONT_URLS = {
    "Fraunces.ttf": _FONT_REPO + "fraunces/Fraunces%5BSOFT%2CWONK%2Copsz%2Cwght%5D.ttf",
    "Fraunces-Italic.ttf": (_FONT_REPO
                            + "fraunces/Fraunces-Italic%5BSOFT%2CWONK%2Copsz%2Cwght%5D.ttf"),
    "GeistMono.ttf": _FONT_REPO + "geistmono/GeistMono%5Bwght%5D.ttf",
}


def fetch_fonts() -> Path:
    fonts = OUT / ".fonts"
    fonts.mkdir(exist_ok=True)
    for name, url in FONT_URLS.items():
        if not (fonts / name).exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(url, fonts / name)
    return fonts

THEMES = {
    "light": dict(bg="#FBFAF7", card="#FFFFFF", border="#EAE5DC", rule="#D8D1C4",
                  ink="#14110D", ink3="#57514A", ink4="#8A8378",
                  brand="#0F1B33", gold="#B08D57", track="#EAE5DC",
                  on_brand="#D9C094"),
    "dark": dict(bg="#0B1426", card="#111D36", border="#24324F", rule="#31415F",
                 ink="#F4F1EA", ink3="#C9C2B4", ink4="#8E97AB",
                 brand="#F4F1EA", gold="#D9C094", track="#24324F",
                 on_brand="#7A5C24"),
}

DESKS = [(key, house, discipline) for key, (house, discipline) in AGENT_PERSONAS.items()]


class Face:
    """One static instance of a variable font, shaped with HarfBuzz."""

    def __init__(self, path: Path, **axes: float):
        tt = instantiateVariableFont(TTFont(path), axes)
        buf = BytesIO()
        tt.save(buf)
        self.tt = TTFont(BytesIO(buf.getvalue()))
        self.glyphs = self.tt.getGlyphSet()
        self.order = self.tt.getGlyphOrder()
        self.upm = self.tt["head"].unitsPerEm
        self.hb = hb.Font(hb.Face(buf.getvalue()))

    def _shape(self, text: str, tracking: float):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb, buf)
        x = 0.0
        placed = []
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions, strict=True):
            placed.append((self.order[info.codepoint], x + pos.x_offset, pos.y_offset))
            x += pos.x_advance + tracking * self.upm
        return placed, x - tracking * self.upm

    def width(self, text: str, size: float, tracking: float = 0.0) -> float:
        return self._shape(text, tracking)[1] * size / self.upm

    def draw(self, text: str, x: float, y: float, size: float, fill: str,
             anchor: str = "start", tracking: float = 0.0,
             max_width: float | None = None) -> str:
        """Outline `text` with its baseline at (x, y). Shrinks to fit max_width."""
        placed, advance = self._shape(text, tracking)
        scale = size / self.upm
        if max_width and advance * scale > max_width:
            scale = max_width / advance
        pen = SVGPathPen(self.glyphs, ntos=lambda v: str(round(v)))
        for name, gx, gy in placed:
            self.glyphs[name].draw(TransformPen(pen, (1, 0, 0, 1, gx, gy)))
        w = advance * scale
        x0 = x - w if anchor == "end" else x - w / 2 if anchor == "middle" else x
        return (f'<path fill="{fill}" transform="translate({x0:.2f} {y:.2f}) '
                f'scale({scale:.5f} {-scale:.5f})" d="{pen.getCommands()}"/>')


def svg(height: int, title: str, body: list[str], t: dict) -> str:
    title = escape(title)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" '
        f'width="{W}" height="{height}" role="img" aria-label="{title}">'
        f"<title>{title}</title>"
        f'<rect width="{W}" height="{height}" rx="10" fill="{t["bg"]}"/>'
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" rx="10" '
        f'fill="none" stroke="{t["border"]}"/>'
        + "".join(body) + "</svg>\n"
    )


def line(x1, y1, x2, y2, stroke, width=1.0) -> str:
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{width}"/>')


def rect(x, y, w, h, fill, stroke=None, rx=4) -> str:
    s = f' stroke="{stroke}"' if stroke else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{s}/>'


def banner(f: dict, t: dict) -> str:
    cx = W / 2
    b = [
        rect(40, 0, W - 80, 2, t["gold"], rx=0),
        f["mono"].draw("MULTI-STRATEGY INVESTMENT COMMITTEE", 56, 58, 11, t["ink4"], tracking=0.18),
        f["mono"].draw("3-6 MONTH HORIZON", W - 56, 58, 11, t["ink4"], "end", tracking=0.18),
        # Monogram, matching the app's brand mark
        rect(cx - 24, 96, 48, 48, "none", t["brand"], rx=2),
        rect(cx - 20, 100, 40, 40, "none", t["gold"], rx=1),
        f["display"].draw("EQ", cx, 128, 20, t["brand"], "middle", tracking=0.04),
        f["display"].draw("EquityTrader", cx, 262, 124, t["brand"], "middle", tracking=-0.02),
        line(cx - 36, 300, cx + 36, 300, t["gold"], 1.5),
        f["italic"].draw("Seven specialist desks. One ticker. One verdict.",
                         cx, 350, 30, t["ink3"], "middle"),
        line(56, 404, W - 56, 404, t["border"]),
    ]
    labels = ["FUNDAMENTAL", "MACRO", "TECHNICAL", "QUANT", "OPTIONS", "PATTERN", "BACKTEST"]
    step = (W - 112) / len(labels)
    for i, label in enumerate(labels):
        b.append(f["mono"].draw(label, 56 + step * (i + 0.5), 436, 10.5,
                                t["ink4"], "middle", tracking=0.2))
    return svg(468, "EquityTrader: a multi-strategy investment committee. "
                    "Seven specialist desks, one ticker, one verdict.", b, t)


def committee(f: dict, t: dict) -> str:
    b = [
        f["mono"].draw("THE COMMITTEE", 56, 62, 11, t["ink4"], tracking=0.18),
        f["mono"].draw("FIXED WEIGHTS, SUM TO 100", W - 56, 62, 11, t["ink4"], "end",
                       tracking=0.18),
        line(56, 80, W - 56, 80, t["border"]),
    ]
    cw, ch, gap = 277, 176, 20
    top_weight = max(AGENT_WEIGHTS.values())
    # Every bar starts at the same x, clear of the widest weight label.
    bar_x = 22 + round(max(f["display"].width(f"{w:.0f}%", 46)
                           for w in AGENT_WEIGHTS.values())) + 16
    for i, (key, firm, discipline) in enumerate(DESKS):
        x = 56 + (i % 4) * (cw + gap)
        y = 104 + (i // 4) * (ch + gap)
        weight = AGENT_WEIGHTS[key]
        b += [
            rect(x, y, cw, ch, t["card"], t["border"]),
            f["mono"].draw(f"{i + 1:02d}", x + 22, y + 34, 10.5, t["gold"], tracking=0.14),
            f["text"].draw(firm, x + 22, y + 72, 23, t["ink"], max_width=cw - 44),
            f["mono"].draw(discipline.upper(), x + 22, y + 96, 9.5, t["ink4"], tracking=0.16),
            f["display"].draw(f"{weight:.0f}%", x + 22, y + 148, 46, t["brand"]),
            rect(x + bar_x, y + 138, cw - bar_x - 22, 2, t["track"], rx=1),
            rect(x + bar_x, y + 138, (cw - bar_x - 22) * weight / top_weight, 2, t["gold"],
                 rx=1),
        ]
    return svg(516, "The committee: " + ", ".join(
        f"{firm} ({d}) {AGENT_WEIGHTS[k]:.0f}%" for k, firm, d in DESKS), b, t)


def architecture(f: dict, t: dict) -> str:
    row_y, pitch, row_h = 120, 48, 38
    last_cy = row_y + pitch * (len(DESKS) - 1) + row_h / 2
    cy = (row_y + row_h / 2 + last_cy) / 2
    # Left to right: ticker, gatherer, the desks between two spines, chair, verdict.
    ticker_x, ticker_w = 56, 104
    gather_x, gather_w = 182, 150
    spine_in, desk_x, desk_w, spine_out = 354, 378, 330, 732
    chair_x, chair_w = 756, 244
    verdict_x, verdict_w = 1038, 186
    desks_mid = (spine_in + spine_out) / 2
    b = [
        f["mono"].draw("ARCHITECTURE", 56, 62, 11, t["ink4"], tracking=0.18),
        f["mono"].draw("ONE RUN, END TO END", W - 56, 62, 11, t["ink4"], "end", tracking=0.18),
        line(56, 80, W - 56, 80, t["border"]),
        # Ticker
        rect(ticker_x, cy - 52, ticker_w, 104, t["card"], t["border"]),
        f["mono"].draw("TICKER", ticker_x + ticker_w / 2, cy - 22, 9.5, t["ink4"], "middle",
                       tracking=0.18),
        f["display"].draw("NVDA", ticker_x + ticker_w / 2, cy + 20, 30, t["brand"], "middle"),
        # Gatherer: every desk's data, fetched before any desk starts
        line(ticker_x + ticker_w, cy, gather_x, cy, t["gold"]),
        rect(gather_x, cy - 76, gather_w, 152, t["card"], t["border"]),
        rect(gather_x, cy - 76, gather_w, 2, t["gold"], rx=0),
        f["mono"].draw("GATHERER", gather_x + 16, cy - 44, 9.5, t["gold"], tracking=0.18),
        f["display"].draw("Data, first", gather_x + 16, cy - 12, 21, t["brand"],
                          max_width=gather_w - 32),
        line(gather_x + 16, cy + 4, gather_x + gather_w - 16, cy + 4, t["border"]),
        f["text"].draw("Every desk's data,", gather_x + 16, cy + 28, 13, t["ink3"],
                       max_width=gather_w - 32),
        f["text"].draw("fetched in order", gather_x + 16, cy + 47, 13, t["ink3"],
                       max_width=gather_w - 32),
        f["mono"].draw("YFINANCE  SEC  FRED", gather_x + 16, cy + 66, 7.5, t["ink4"],
                       tracking=0.1),
        f["mono"].draw("CODE, NOT A MODEL", gather_x + gather_w / 2, cy + 104, 9, t["ink4"],
                       "middle", tracking=0.14),
        # Fan-out spine
        line(gather_x + gather_w, cy, spine_in, cy, t["gold"]),
        line(spine_in, row_y + row_h / 2, spine_in, last_cy, t["gold"]),
        line(spine_out, row_y + row_h / 2, spine_out, last_cy, t["gold"]),
        line(spine_out, cy, chair_x, cy, t["gold"]),
        f["mono"].draw(f"{len(DESKS)} SPECIALISTS, ONE MODEL CALL EACH, 3 AT A TIME",
                       desks_mid, 104, 9,
                       t["ink4"], "middle", tracking=0.14),
        f["mono"].draw("EACH DESK IS HANDED ITS DATA AND HAS NO TOOLS", desks_mid,
                       last_cy + 46, 9, t["ink4"], "middle", tracking=0.14),
    ]
    for i, (_, firm, discipline) in enumerate(DESKS):
        y = row_y + i * pitch
        mid = y + row_h / 2
        b += [
            line(spine_in, mid, desk_x, mid, t["gold"]),
            line(desk_x + desk_w, mid, spine_out, mid, t["gold"]),
            rect(desk_x, y, desk_w, row_h, t["card"], t["border"]),
            f["text"].draw(firm, desk_x + 16, mid + 5.5, 15.5, t["ink"], max_width=180),
            f["mono"].draw(discipline.upper(), desk_x + desk_w - 14, mid + 3.5, 8, t["ink4"],
                           "end", tracking=0.1),
        ]
    oc, inner = cy, chair_x + 22
    b += [
        rect(chair_x, oc - 106, chair_w, 212, t["card"], t["brand"]),
        rect(chair_x, oc - 106, chair_w, 2, t["gold"], rx=0),
        f["mono"].draw("ORCHESTRATOR", inner, oc - 70, 9.5, t["gold"], tracking=0.18),
        f["display"].draw("Committee chair", inner, oc - 34, 25, t["brand"],
                          max_width=chair_w - 44),
        line(inner, oc - 14, chair_x + chair_w - 22, oc - 14, t["border"]),
    ]
    for i, text in enumerate(["Deterministic weighted score",
                              "Portfolio manager memo",
                              "Live catalyst calendar",
                              "Facts stamped after the LLM"]):
        b.append(f["text"].draw(text, inner, oc + 14 + i * 25, 14, t["ink3"],
                                max_width=chair_w - 44))
    tip, vx = verdict_x - 4, verdict_x + 22
    b += [
        line(chair_x + chair_w, oc, tip, oc, t["gold"]),
        f'<path d="M{tip} {oc} l-7 -4 v8 z" fill="{t["gold"]}"/>',
        rect(verdict_x, oc - 76, verdict_w, 152, t["brand"], rx=4),
        f["mono"].draw("VERDICT", vx, oc - 42, 9.5, t["on_brand"], tracking=0.18),
        f["display"].draw("Buy / Hold / Sell", vx, oc - 6, 23, t["bg"],
                          max_width=verdict_w - 44),
        f["text"].draw("Conviction, target,", vx, oc + 26, 13.5, t["on_brand"]),
        f["text"].draw("sizing, stop loss", vx, oc + 46, 13.5, t["on_brand"]),
        f["mono"].draw("SQLITE + JSON SNAPSHOT", verdict_x + verdict_w / 2, oc + 104, 9,
                       t["ink4"], "middle", tracking=0.14),
    ]
    return svg(round(last_cy + 65), "Architecture: the gatherer fetches every desk's data "
                    "before any desk starts; seven specialist agents are each handed their "
                    "share and answer in "
                    "one model call, three at a time; the "
                    "orchestrator combines a deterministic "
                    "weighted score with an LLM memo and live catalysts into one verdict, "
                    "persisted to SQLite and JSON.", b, t)


def main() -> None:
    fonts = fetch_fonts()
    plain = dict(SOFT=0, WONK=0)
    f = {
        "display": Face(fonts / "Fraunces.ttf", opsz=144, wght=500, **plain),
        "text": Face(fonts / "Fraunces.ttf", opsz=24, wght=500, **plain),
        "italic": Face(fonts / "Fraunces-Italic.ttf", opsz=72, wght=400, **plain),
        "mono": Face(fonts / "GeistMono.ttf", wght=500),
    }
    for name, build in [("banner", banner), ("committee", committee),
                        ("architecture", architecture)]:
        for theme, t in THEMES.items():
            path = OUT / f"{name}-{theme}.svg"
            path.write_text(build(f, t), encoding="utf-8")
            print(f"{path.name}: {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
