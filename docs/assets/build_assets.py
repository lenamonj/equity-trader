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
        f["italic"].draw("Eight specialist desks. One ticker. One verdict.",
                         cx, 350, 30, t["ink3"], "middle"),
        line(56, 404, W - 56, 404, t["border"]),
    ]
    labels = ["FUNDAMENTAL", "MACRO", "TECHNICAL", "QUANT",
              "ETF FLOW", "OPTIONS", "PATTERN", "BACKTEST"]
    step = (W - 112) / len(labels)
    for i, label in enumerate(labels):
        b.append(f["mono"].draw(label, 56 + step * (i + 0.5), 436, 10.5,
                                t["ink4"], "middle", tracking=0.2))
    return svg(468, "EquityTrader: a multi-strategy investment committee. "
                    "Eight specialist desks, one ticker, one verdict.", b, t)


def committee(f: dict, t: dict) -> str:
    b = [
        f["mono"].draw("THE COMMITTEE", 56, 62, 11, t["ink4"], tracking=0.18),
        f["mono"].draw("FIXED WEIGHTS, SUM TO 100", W - 56, 62, 11, t["ink4"], "end",
                       tracking=0.18),
        line(56, 80, W - 56, 80, t["border"]),
    ]
    cw, ch, gap = 277, 176, 20
    top_weight = max(AGENT_WEIGHTS.values())
    for i, (key, firm, discipline) in enumerate(DESKS):
        x = 56 + (i % 4) * (cw + gap)
        y = 104 + (i // 4) * (ch + gap)
        weight = AGENT_WEIGHTS[key]
        num_w = f["display"].width(f"{weight:.0f}", 46)
        b += [
            rect(x, y, cw, ch, t["card"], t["border"]),
            f["mono"].draw(f"{i + 1:02d}", x + 22, y + 34, 10.5, t["gold"], tracking=0.14),
            f["text"].draw(firm, x + 22, y + 72, 23, t["ink"], max_width=cw - 44),
            f["mono"].draw(discipline.upper(), x + 22, y + 96, 9.5, t["ink4"], tracking=0.16),
            f["display"].draw(f"{weight:.0f}", x + 22, y + 148, 46, t["brand"]),
            f["mono"].draw("%", x + 28 + num_w, y + 148, 12, t["ink4"]),
            rect(x + 120, y + 138, cw - 142, 2, t["track"], rx=1),
            rect(x + 120, y + 138, (cw - 142) * weight / top_weight, 2, t["gold"], rx=1),
        ]
    return svg(516, "The committee: " + ", ".join(
        f"{firm} ({d}) {AGENT_WEIGHTS[k]:.0f}%" for k, firm, d in DESKS), b, t)


def architecture(f: dict, t: dict) -> str:
    row_y, pitch, row_h = 120, 48, 38
    last_cy = row_y + pitch * 7 + row_h / 2
    cy = (row_y + row_h / 2 + last_cy) / 2
    b = [
        f["mono"].draw("ARCHITECTURE", 56, 62, 11, t["ink4"], tracking=0.18),
        f["mono"].draw("ONE RUN, END TO END", W - 56, 62, 11, t["ink4"], "end", tracking=0.18),
        line(56, 80, W - 56, 80, t["border"]),
        # Ticker
        rect(56, cy - 52, 150, 104, t["card"], t["border"]),
        f["mono"].draw("TICKER", 131, cy - 22, 9.5, t["ink4"], "middle", tracking=0.18),
        f["display"].draw("NVDA", 131, cy + 22, 34, t["brand"], "middle"),
        # Fan-out spine
        line(206, cy, 250, cy, t["gold"]),
        line(250, row_y + row_h / 2, 250, last_cy, t["gold"]),
        line(655, row_y + row_h / 2, 655, last_cy, t["gold"]),
        line(655, cy, 690, cy, t["gold"]),
        f["mono"].draw("8 SPECIALISTS, 3 AT A TIME, BACKOFF ON 429", 455, 104, 9, t["ink4"],
                       "middle", tracking=0.14),
        f["mono"].draw("THE GATHERER FETCHES ALL DATA FIRST: YFINANCE, SEC, FRED", 455,
                       last_cy + 46, 9, t["ink4"], "middle", tracking=0.14),
    ]
    for i, (_, firm, discipline) in enumerate(DESKS):
        y = row_y + i * pitch
        mid = y + row_h / 2
        b += [
            line(250, mid, 290, mid, t["gold"]),
            line(620, mid, 655, mid, t["gold"]),
            rect(290, y, 330, row_h, t["card"], t["border"]),
            f["text"].draw(firm, 306, mid + 5.5, 15.5, t["ink"], max_width=180),
            f["mono"].draw(discipline.upper(), 606, mid + 3.5, 8, t["ink4"], "end", tracking=0.1),
        ]
    oc = cy
    b += [
        rect(690, oc - 106, 270, 212, t["card"], t["brand"]),
        rect(690, oc - 106, 270, 2, t["gold"], rx=0),
        f["mono"].draw("ORCHESTRATOR", 712, oc - 70, 9.5, t["gold"], tracking=0.18),
        f["display"].draw("Committee chair", 712, oc - 34, 25, t["brand"]),
        line(712, oc - 14, 938, oc - 14, t["border"]),
    ]
    for i, text in enumerate(["Deterministic weighted score",
                              "Portfolio manager memo",
                              "Live catalyst calendar",
                              "Ground truth stamped after the LLM"]):
        b.append(f["text"].draw(text, 712, oc + 14 + i * 25, 14, t["ink3"], max_width=226))
    b += [
        line(960, oc, 1010, oc, t["gold"]),
        f'<path d="M1010 {oc} l-7 -4 v8 z" fill="{t["gold"]}"/>',
        rect(1014, oc - 76, 210, 152, t["brand"], rx=4),
        f["mono"].draw("VERDICT", 1036, oc - 42, 9.5, t["on_brand"], tracking=0.18),
        f["display"].draw("Buy / Hold / Sell", 1036, oc - 6, 23, t["bg"], max_width=166),
        f["text"].draw("Conviction, target,", 1036, oc + 26, 13.5, t["on_brand"]),
        f["text"].draw("sizing, stop loss", 1036, oc + 46, 13.5, t["on_brand"]),
        f["mono"].draw("SQLITE + JSON SNAPSHOT", 1119, oc + 104, 9, t["ink4"], "middle",
                       tracking=0.14),
    ]
    return svg(540, "Architecture: the gatherer fetches the data, then a ticker fans out to "
                    "eight specialist agents, three at a time, reading it from one cache; "
                    "the orchestrator combines a deterministic "
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
