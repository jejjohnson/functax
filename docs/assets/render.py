"""Render the functax README assets: icon, logos and the two diagrams.

Every output is a plain SVG written next to this file. The icon shares the
GeoStack tile (a 120-unit rounded square, r = 26, white glyph) in the functax
cobalt gradient. Its glyph is a basis expansion: four faint bumps φⱼ and the
bold curve x = ∑_d z_d φ_d that is their weighted sum.

Diagrams come in a light and a dark variant for GitHub's ``<picture>`` /
``prefers-color-scheme`` switch.

Run from the repo root:

    uv run --no-project python docs/assets/render.py
"""

from __future__ import annotations

import math
from pathlib import Path


OUT = Path(__file__).parent

DOT = " \N{MIDDLE DOT} "
SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# Cobalt: the functax gradient and its wordmark accents.
C1, C2 = "#3B82F6", "#1E40AF"
ACCENT = {"light": "#1D4ED8", "dark": "#60A5FA"}

THEME = {
    "light": {
        "text": "#1F2937",
        "muted": "#6B7280",
        "card": "#F9FAFB",
        "line": "#D1D5DB",
        "arrow": "#9CA3AF",
        "hi_fill": "#EFF6FF",
        "hi_line": "#2563EB",
        "site": "#1D4ED8",
        "band": "#F3F4F6",
    },
    "dark": {
        "text": "#E5E7EB",
        "muted": "#9CA3AF",
        "card": "#111827",
        "line": "#374151",
        "arrow": "#6B7280",
        "hi_fill": "#172554",
        "hi_line": "#3B82F6",
        "site": "#60A5FA",
        "band": "#1F2937",
    },
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


def text(
    x: float,
    y: float,
    s: str,
    *,
    size: float = 14,
    weight: int = 400,
    fill: str,
    family: str = SANS,
    anchor: str = "start",
) -> str:
    """An SVG text element, whitespace preserved."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'xml:space="preserve">{s}</text>'
    )


def arrow_defs(mid: str, colour: str) -> str:
    """An arrowhead marker definition with the given id and colour."""
    return (
        f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{colour}"/></marker></defs>'
    )


def line(points: list[tuple[float, float]], colour: str, head: str | None) -> str:
    """An SVG polyline through the points, with an optional arrowhead."""
    d = " ".join(f"{'M' if i == 0 else 'L'}{x} {y}" for i, (x, y) in enumerate(points))
    end = f' marker-end="url(#{head})"' if head else ""
    return (
        f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="1.6" '
        f'stroke-linejoin="round"{end}/>'
    )


def card(
    x: float,
    y: float,
    w: float,
    h: float,
    t: dict[str, str],
    *,
    highlight: bool = False,
) -> str:
    """A rounded card, highlighted in the accent colours when asked."""
    fill, stroke, sw = (
        (t["hi_fill"], t["hi_line"], 2) if highlight else (t["card"], t["line"], 1)
    )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>'
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


def sub(s: str, size: float = 12.5) -> str:
    """A subscript of text set at ``size``, as an SVG tspan."""
    return (
        f'<tspan baseline-shift="{-0.25 * size:g}" font-size="{0.7 * size:g}">'
        f"{s}</tspan>"
    )


def bold(s: str) -> str:
    """A vector or matrix symbol, bold as in the maths standard."""
    return f'<tspan font-weight="700">{s}</tspan>'


# ---------------------------------------------------------------- hero

# (verb, equation, module, names) for the five operations on the expansion.
# x(s) ≈ ∑_d z_d φ_d(s; θ)
EXPANSION = (
    f"x({bold('s')}) ≈ ∑{sub('d', 24)} z{sub('d', 24)} φ{sub('d', 24)}"
    f"({bold('s')}; {bold('θ')})"
)
OPERATIONS = [
    (
        "interpolate",
        f"{bold('y')} = {bold('H')} {bold('x')},  {bold('H')}ᵀ {bold('r')}",
        "interp",
        ["Plan", "Linear", "RBF", "Kriging"],
    ),
    (
        "differentiate",
        f"∇{sub(bold('s'))} x,  ∇²x",
        "diffops",
        ["gradient", "laplacian", "surface_grad"],
    ),
    (
        "integrate",
        f"∫ x d{bold('s')} ≈ ∑ w{sub('n')} x({bold('s')}{sub('n')})",
        "quad",
        ["Rule", "SphereRule", "cell_area"],
    ),
    (
        "project",
        f"{bold('ẑ')} = argmin ‖{bold('Φ')}{bold('z')} − {bold('y')}‖²",
        "project",
        ["least_squares", "galerkin", "Variational"],
    ),
    (
        "transform",
        f"x({bold('s')}{sub('n')}) ↔ {bold('z')}",
        "transform",
        ["SphericalHarmonicTransform", "nonuniform_rfft", "ChebyshevTransform1D"],
    ),
]
BASES = DOT.join(
    ["spherical harmonics", "Slepian", "needlets", "Fourier", "RBF", "wavelets", "EOFs"]
)
GEOMETRY = DOT.join(
    ["Axis", "AffineGrid", "Points", "sphere grids", "meshes", "frames"]
)


def write_hero() -> None:
    """One expansion in the middle, five operations under it, two foundations below."""
    width, height = 960, 470
    for mode, t in THEME.items():
        head = f"hero-{mode}-head"
        a = t["arrow"]
        parts = [arrow_defs(head, a)]

        # The expansion.
        parts.append(card(230, 20, 500, 92, t, highlight=True))
        parts.append(
            # Left-anchored: some renderers misplace centred text with tspans.
            text(
                348,
                62,
                EXPANSION,
                size=24,
                weight=400,
                fill=t["text"],
                family=MONO,
            )
        )
        parts.append(
            text(
                480,
                90,
                "one expansion of the field x: coefficients z, basis functions φ, "
                "parameters θ",
                size=13,
                fill=t["muted"],
                anchor="middle",
            )
        )

        # The five operations.
        # The transform card is wider: its class names are the longest.
        y0, h, gap = 168, 172, 15
        widths = [160, 160, 160, 160, 220]
        for i, (verb, eq, module, names) in enumerate(OPERATIONS):
            w = widths[i]
            x = 20 + sum(widths[:i]) + i * gap
            cx = x + w / 2
            parts.append(line([(480 + (cx - 480) * 0.4, 112), (cx, y0 - 4)], a, head))
            parts.append(card(x, y0, w, h, t))
            parts.append(
                text(x + 14, y0 + 26, verb, size=15, weight=700, fill=t["site"])
            )
            parts.append(
                text(x + 14, y0 + 50, eq, size=12.5, fill=t["text"], family=MONO)
            )
            parts.append(
                text(
                    x + 14,
                    y0 + 74,
                    f"functax.{module}",
                    size=12,
                    fill=t["muted"],
                    family=MONO,
                )
            )
            for j, name in enumerate(names):
                parts.append(
                    text(
                        x + 14,
                        y0 + 104 + 20 * j,
                        name,
                        size=12,
                        weight=600,
                        fill=t["text"],
                        family=MONO,
                    )
                )

        # The foundations: the basis functions φ and where x lives.
        for k, (module, what, items) in enumerate(
            [
                ("functax.basis", "the basis functions φ", BASES),
                ("functax.geometry", "where x lives, CRS-free", GEOMETRY),
            ]
        ):
            yb = 366 + 46 * k
            parts.append(
                f'<rect x="20" y="{yb}" width="920" height="38" rx="10" '
                f'fill="{t["band"]}"/>'
            )
            parts.append(
                text(
                    36,
                    yb + 24,
                    module,
                    size=13.5,
                    weight=700,
                    fill=t["site"],
                    family=MONO,
                )
            )
            parts.append(text(196, yb + 24, what, size=12.5, fill=t["muted"]))
            parts.append(
                text(924, yb + 24, items, size=12.5, fill=t["text"], anchor="end")
            )

        label = (
            "functax is built around one expansion, a field x of the coordinates s "
            "equal to the sum over d of z_d phi_d of s, and five operations on it: "
            "interpolate, differentiate, "
            "integrate, project and transform, each in its own module, over a basis "
            "library and CRS-free geometry."
        )
        (OUT / f"hero-{mode}.svg").write_text(
            svg(
                width, height, label, "One expansion, five operations", "\n".join(parts)
            )
        )


# ---------------------------------------------------------------- layers

USERS = [
    ("imports it today", ["geonnax", "integrix"]),
    (
        "models and numerics, Wave 5",
        ["vardax", "sesmax", "plumax", "radtrax", "finitevolX"],
    ),
    (
        "learning and data, Wave 5",
        ["kernellib", "pyrox", "filterax", "geoarray", "xrops"],
    ),
]


def write_layers() -> None:
    """Where functax sits; arrows point at what each group imports."""
    width, height = 960, 400
    for mode, t in THEME.items():
        head = f"layers-{mode}-head"
        a = t["arrow"]
        parts = [arrow_defs(head, a)]
        xs = [30, 350, 670]
        for (group, names), x in zip(USERS, xs, strict=True):
            parts.append(card(x, 20, 260, 92, t))
            parts.append(
                text(x + 130, 46, group, size=12.5, fill=t["muted"], anchor="middle")
            )
            for row in range(0, len(names), 3):
                parts.append(
                    text(
                        x + 130,
                        72 + 22 * (row // 3),
                        DOT.join(names[row : row + 3]),
                        size=14,
                        weight=700,
                        fill=t["text"],
                        family=MONO,
                        anchor="middle",
                    )
                )
            x_end = x + 130 if x == 350 else x + 130 + (480 - x - 130) * 0.45
            parts.append(line([(x + 130, 112), (x_end, 182)], a, head))
        parts.append(card(170, 186, 620, 96, t, highlight=True))
        parts.append(tile(f"layers-{mode}-icon", 192, 204, 60))
        parts.append(wordmark(268, 238, 26, mode))
        parts.append(
            text(
                268,
                264,
                "Core tier 0: bases, plans with adjoints, rules and transforms",
                size=13,
                fill=t["muted"],
            )
        )
        parts.append(line([(480, 282), (480, 328)], a, head))
        parts.append(
            f'<rect x="20" y="332" width="920" height="48" rx="10" fill="{t["band"]}"/>'
        )
        parts.append(
            text(
                480,
                362,
                "numpy (base)"
                + DOT
                + "jax"
                + DOT
                + "equinox"
                + DOT
                + "lineax"
                + DOT
                + "einx"
                + DOT
                + "jaxtyping  (the [jax] extra)",
                size=14,
                weight=600,
                fill=t["text"],
                family=MONO,
                anchor="middle",
            )
        )
        label = (
            "geonnax and integrix import functax today, and the stack's models, "
            "numerics, learning and data libraries switch to it in Wave 5; functax "
            "depends on NumPy alone, plus JAX, Equinox, lineax, einx and jaxtyping "
            "with the jax extra."
        )
        (OUT / f"layers-{mode}.svg").write_text(
            svg(width, height, label, "Where functax sits", "\n".join(parts))
        )


if __name__ == "__main__":
    write_icon()
    write_logos()
    write_hero()
    write_layers()
