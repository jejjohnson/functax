"""Render the functax README assets: the icon and the two logos.

Every output is a plain SVG written next to this file. The icon shares the
GeoStack tile (a 120-unit rounded square, r = 26, white glyph) in the functax
cobalt gradient. Its glyph is a basis expansion: four faint bumps φⱼ and the
bold curve x = ∑_d z_d φ_d that is their weighted sum.

The logos come in a light and a dark variant for GitHub's ``<picture>`` /
``prefers-color-scheme`` switch. The README's schematics are drawn in geostack
(``docs/images/libraries/functax_*``) with the stack's other library figures.

Run from the repo root:

    uv run --no-project python docs/assets/render.py
"""

from __future__ import annotations

import math
from pathlib import Path


OUT = Path(__file__).parent

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

# Cobalt: the functax gradient and its wordmark accents.
C1, C2 = "#3B82F6", "#1E40AF"
ACCENT = {"light": "#1D4ED8", "dark": "#60A5FA"}

THEME = {
    "light": {"text": "#1F2937"},
    "dark": {"text": "#E5E7EB"},
}

W = "#FFFFFF"


# ---------------------------------------------------------------- glyph


def _polyline(points: list[tuple[float, float]]) -> str:
    """An SVG path through the given points."""
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in points)


def glyph_expansion() -> str:
    """Four Gaussian bumps φⱼ on a baseline, faint, and their weighted sum, bold."""
    x0, x1, base = 22.0, 106.0, 96.0
    centres = [34.0, 54.0, 74.0, 94.0]
    weights = [30.0, 52.0, 24.0, 42.0]
    width = 11.0
    n = 120
    xs = [x0 + (x1 - x0) * i / n for i in range(n + 1)]

    def bump(x: float, centre: float, weight: float) -> float:
        """The weighted Gaussian bump wⱼ exp(−(x − cⱼ)² / 2σ²)."""
        return weight * math.exp(-0.5 * ((x - centre) / width) ** 2)

    parts = []
    for centre, weight in zip(centres, weights, strict=True):
        points = [(x, base - bump(x, centre, weight)) for x in xs]
        parts.append(
            f'<path d="{_polyline(points)}" fill="none" stroke="{W}" '
            'stroke-width="2.6" stroke-opacity="0.55" stroke-linecap="round"/>'
        )
    total = [
        (x, base - sum(bump(x, c, w) for c, w in zip(centres, weights, strict=True)))
        for x in xs
    ]
    parts.append(
        f'<path d="{_polyline(total)}" fill="none" stroke="{W}" stroke-width="5" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )
    parts.append(
        f'<line x1="{x0}" y1="{base + 2}" x2="{x1}" y2="{base + 2}" stroke="{W}" '
        'stroke-width="3" stroke-linecap="round" stroke-opacity="0.8"/>'
    )
    return "".join(parts)


def tile(gid: str, x: float = 0, y: float = 0, size: float = 128) -> str:
    """The gradient tile with the glyph, placed at (x, y), ``size`` wide."""
    s = size / 128
    return (
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{C1}"/><stop offset="1" stop-color="{C2}"/>'
        "</linearGradient></defs>"
        f'<g transform="translate({x} {y}) scale({s:g})">'
        f'<rect x="4" y="4" width="120" height="120" rx="26" fill="url(#{gid})"/>'
        f"{glyph_expansion()}</g>"
    )


def svg(width: int, height: int, label: str, title: str, body: str) -> str:
    """A standalone SVG document with a title and an accessible label."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{label}">\n'
        f"<title>{title}</title>\n{body}\n</svg>\n"
    )


def wordmark(x: float, y: float, size: float, mode: str) -> str:
    """The functax wordmark, its last letter in the accent colour."""
    t = THEME[mode]
    return (
        f'<text x="{x}" y="{y}" font-family="{SANS}" font-size="{size}" '
        f'font-weight="700" letter-spacing="{-size / 48:.2f}" fill="{t["text"]}">'
        f'functa<tspan fill="{ACCENT[mode]}">x</tspan></text>'
    )


# ---------------------------------------------------------------- icon and logo


def write_icon() -> None:
    """Write icon.svg, the tile on its own."""
    (OUT / "icon.svg").write_text(svg(128, 128, "functax", "functax", tile("ic")))


def write_logos() -> None:
    """Write logo-light.svg and logo-dark.svg, the tile and the wordmark."""
    for mode in THEME:
        body = tile(f"logo-{mode}", x=6, y=6) + wordmark(154, 92, 72, mode)
        (OUT / f"logo-{mode}.svg").write_text(svg(510, 140, "functax", "functax", body))


if __name__ == "__main__":
    write_icon()
    write_logos()
