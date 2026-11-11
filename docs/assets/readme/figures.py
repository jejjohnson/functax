"""Render the docs' and the README's figures from the docs' own code.

The quick start (docs/guide/quickstart.md) and the methane example
(docs/guide/methane.md) run their Python blocks verbatim and in order, then
the example's figure is drawn from what they computed, so the numbers the
pages quote and the picture come from the same run (they are logged). The
gallery figures call the bases directly.

Run from the repo root (about a minute on a laptop CPU):

    uv run --group dev --with matplotlib python docs/assets/readme/figures.py
"""

from __future__ import annotations

import re
from pathlib import Path

import einx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from functax.basis.slepian import slepian_cap_basis
from functax.basis.sphere import needlet_basis, real_spherical_harmonics
from functax.geometry import lonlat_to_unit


OUT = Path(__file__).parent
QUICKSTART = OUT.parents[1] / "guide" / "quickstart.md"
METHANE = OUT.parents[1] / "guide" / "methane.md"


def python_blocks(page: Path, heading: str | None = None) -> list[str]:
    """The ```python blocks of a page, or of its ``## <heading>`` section only."""
    text = page.read_text()
    if heading is not None:
        start = text.index(f"\n## {heading}")
        end = text.find("\n## ", start + 1)
        text = text[start : len(text) if end == -1 else end]
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def run(page: Path, heading: str | None = None) -> dict:
    """Execute a page section's blocks in one namespace, as a reader would."""
    namespace: dict = {}
    for block in python_blocks(page, heading):
        exec(compile(block, page.name, "exec"), namespace)
    return namespace


def save(fig: plt.Figure, name: str) -> None:
    """Lay out a figure and write it next to this script at 150 dpi."""
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- quick start


def check_quick_start() -> None:
    """Run the quickstart's blocks and log the numbers its prose quotes."""
    ns = run(QUICKSTART, "Three operations on one expansion")
    # Φ ẑ, the fit at the sample points
    fitted = einx.dot("obs feature, feature -> obs", ns["design"], ns["coefficients"])
    # max |Φ ẑ − y|
    fit_error = float(einx.max("[obs]", jnp.abs(fitted - ns["observations"])))
    logger.info("quick start: max fit error {:.1e}", fit_error)
    logger.info("quick start: values {}", ns["values"].tolist())
    logger.info(
        "quick start: integral {!r}, 4π/3 = {!r}", float(ns["integral"]), 4 * np.pi / 3
    )


# ---------------------------------------------------------------- example


def example_figure() -> None:
    """Run the methane page's blocks, log its numbers and draw example.png."""
    ns = run(METHANE)
    for key in ("read_rmse", "adjoint_gap", "observed_rmse", "cloudy_rmse"):
        logger.info("example: {} = {}", key, ns[key])
    for key in ("truth_integral", "map_integral", "pixel_integral"):
        logger.info("example: {} = {:.1f} ppb km²", key, float(ns[key]) / 1e6)
    worst = float(einx.max("[cell]", jnp.abs(ns["map_error"])))
    logger.info("example: max |map error| = {:.2f} ppb", worst)
    n_pixels = int(einx.sum("[cell]", ns["cloud_free"]))
    logger.info("example: N_Ωy = {}", n_pixels)
    grid_area = float(einx.sum("[cell]", ns["area"].weights)) / 1e6
    logger.info("example: grid area {:.0f} km²", grid_area)

    extent = (-104.2, -103.4, 31.6, 32.4)

    def image(values: jnp.ndarray) -> np.ndarray:
        # (λ, φ) row-major → rows of φ, for imshow
        """Field values on the (lon, lat) grid as an image with rows of latitude."""
        return einx.id("(lon lat) -> lat lon", np.asarray(values), lon=100)

    lo, hi = 1875.0, 1925.0
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    panels = [
        (ns["truth"], "truth x (ppb)"),
        (None, "cloud-free pixels y (ppb)"),
        (ns["xch4_map"], "map x̂ from the pixels (ppb)"),
    ]
    for ax, (field, title) in zip(axes[:3], panels, strict=True):
        if field is None:
            pixels = np.asarray(ns["pixels"])
            ax.scatter(
                pixels[:, 0],
                pixels[:, 1],
                c=np.asarray(ns["observations"]),
                s=1.5,
                vmin=lo,
                vmax=hi,
                cmap="viridis",
            )
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
        else:
            ax.imshow(
                image(field),
                origin="lower",
                extent=extent,
                vmin=lo,
                vmax=hi,
                cmap="viridis",
            )
        ax.set_title(title, fontsize=9)
    error = image(ns["map_error"])
    bound = float(einx.max("[lat lon]", np.abs(error)))
    shown = axes[3].imshow(
        error, origin="lower", extent=extent, vmin=-bound, vmax=bound, cmap="RdBu_r"
    )
    axes[3].set_title(
        f"error x̂ − x: {ns['observed_rmse']:.2f} / {ns['cloudy_rmse']:.2f} ppb RMS",
        fontsize=9,
    )
    fig.colorbar(shown, ax=axes[3], fraction=0.046)
    for ax in axes:
        ax.set_xlabel("longitude (°)", fontsize=8)
        ax.set_xticks([-104.0, -103.8, -103.6])
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("latitude (°)", fontsize=8)
    save(fig, "example.png")


# ---------------------------------------------------------------- gallery

LON, LAT = np.meshgrid(np.linspace(-179, 179, 180), np.linspace(-89, 89, 90))


def mollweide(ax: plt.Axes, field: np.ndarray, title: str) -> None:
    """Draw one field on a Mollweide map with a symmetric colour scale."""
    bound = float(einx.max("[point]", np.abs(field)))
    ax.pcolormesh(
        np.deg2rad(LON),
        np.deg2rad(LAT),
        einx.id("(lat lon) -> lat lon", np.asarray(field), lat=LAT.shape[0]),
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
        shading="auto",
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, lw=0.3, color="k", alpha=0.25)
    ax.set_title(title, fontsize=9)


def gallery_figures() -> None:
    """Draw the spherical-harmonic, Slepian and needlet gallery figures."""
    lonlat = jnp.asarray(einx.id("lat lon, lat lon -> (lat lon) (1 + 1)", LON, LAT))
    xyz = lonlat_to_unit(lonlat, unit="degrees")
    moll = {"projection": "mollweide"}

    # Spherical harmonics: global functions.
    harmonics = np.asarray(real_spherical_harmonics(xyz, 4))
    fig, axes = plt.subplots(2, 3, figsize=(9, 3.6), subplot_kw=moll)
    degrees_orders = [(1, 0), (2, 1), (2, -2), (3, 2), (4, 0), (4, 3)]
    for ax, (degree, order) in zip(axes.flat, degrees_orders, strict=True):
        column = degree * degree + degree + order
        mollweide(ax, harmonics[:, column], f"$Y_{{{degree}}}^{{{order}}}$")
    fig.suptitle("real_spherical_harmonics(u, l_max=4): 25 functions", fontsize=10)
    save(fig, "sh_basis.png")

    # Slepian functions: regional functions, ranked by energy inside the cap.
    centre = jnp.deg2rad(jnp.array([-40.0, 25.0]))
    radius = np.deg2rad(40.0)
    basis = slepian_cap_basis(20, radius, lonlat_centre=centre)
    slepian = np.asarray(basis.evaluate(xyz))
    concentration = np.asarray(basis.eigenvalues)
    pole = np.asarray(lonlat_to_unit(einx.id("d -> 1 d", centre)))[0]
    # cos θ = ⟨u, p⟩, the angle from the cap centre
    cosine = np.clip(einx.dot("point d, d -> point", np.asarray(xyz), pole), -1, 1)
    angle = einx.id("(lat lon) -> lat lon", np.arccos(cosine), lat=LAT.shape[0])
    fig, axes = plt.subplots(2, 3, figsize=(9, 3.6), subplot_kw=moll)
    for ax, mode in zip(axes.flat, [0, 3, 15, 40, 52, 70], strict=True):
        title = f"mode {mode + 1},  λ = {concentration[mode]:.2f}"
        mollweide(ax, slepian[:, mode], title)
        ax.contour(
            np.deg2rad(LON),
            np.deg2rad(LAT),
            angle,
            levels=[radius],
            colors="k",
            linewidths=0.7,
            linestyles="--",
        )
    fig.suptitle(
        "slepian_cap_basis(l_max=20, 40° cap): λ = energy inside the cap", fontsize=10
    )
    save(fig, "slepian.png")

    # Needlets: one centre, four scales.
    needlets, scales, centres = map(np.asarray, needlet_basis(xyz, B=2.0, j_max=3))
    point = np.asarray(lonlat_to_unit(jnp.array([[20.0, 10.0]]), unit="degrees"))[0]
    fig, axes = plt.subplots(1, 4, figsize=(11, 1.75), subplot_kw=moll)
    for ax, scale in zip(axes, range(4), strict=True):
        columns = np.where(scales == scale)[0]
        # the needlet of this scale whose centre is nearest the point
        closeness = einx.dot("atom d, d -> atom", centres[columns], point)
        nearest = columns[int(einx.argmax("[atom]", closeness)[0])]
        mollweide(ax, needlets[:, nearest], f"needlet, scale j = {scale}")
    save(fig, "needlets.png")


if __name__ == "__main__":
    check_quick_start()
    example_figure()
    gallery_figures()
