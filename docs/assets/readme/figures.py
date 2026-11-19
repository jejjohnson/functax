"""Render the docs' and the README's figures from the docs' own code.

The quick start (docs/guide/quickstart.md) and the methane example
(docs/guide/methane.md) run their Python blocks verbatim and in order, then
the example's figure is drawn from what they computed, so the numbers the
pages quote and the picture come from the same run (they are logged). The
gallery figures call the bases directly.

Each figure has a title that says its idea and a subtitle that names the call
and how its colours are scaled. Run from the repo root (about a minute on a
laptop CPU), all figures or the ones named:

    uv run --group dev --with matplotlib python docs/assets/readme/figures.py
    uv run --group dev --with matplotlib python docs/assets/readme/figures.py \\
        --only sh_basis
"""

from __future__ import annotations

import re
from pathlib import Path

import einx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from cyclopts import App
from loguru import logger

from functax.basis.slepian import shannon_number, slepian_cap_basis
from functax.basis.sphere import needlet_basis, real_spherical_harmonics
from functax.geometry import lonlat_to_unit


# Both docs pages switch on float64 in their first block; doing it here too
# keeps a figure drawn alone (--only) the same as in a full run.
jax.config.update("jax_enable_x64", True)

OUT = Path(__file__).parent
app = App(help="Draw the README's and the docs' figures from the docs' own code.")
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
    """Write a figure next to this script at 150 dpi."""
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    logger.info("wrote {}", name)


def headed(
    title: str, subtitle: str, size: tuple[float, float]
) -> tuple[plt.Figure, float]:
    """A figure with a bold title and a muted subtitle in a 0.75-inch band on top.

    Returns the figure and the top of the space below the band, as a fraction of
    the figure height, for the caller's ``fig.subplots_adjust``.
    """
    fig = plt.figure(figsize=size)
    height = size[1]
    fig.text(0.012, 1 - 0.1 / height, title, fontsize=12, fontweight="bold", va="top")
    fig.text(
        0.012,
        1 - 0.36 / height,
        subtitle,
        fontsize=9,
        color="0.35",
        va="top",
        linespacing=1.35,
    )
    return fig, 1 - 0.75 / height


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
    grid_area = float(einx.sum("[cell]", ns["cell_weights"])) / 1e6
    logger.info("example: grid area {:.0f} km²", grid_area)

    extent = (-104.2, -103.4, 31.6, 32.4)

    def image(values: jnp.ndarray) -> np.ndarray:
        # (λ, φ) row-major → rows of φ, for imshow
        """Field values on the (lon, lat) grid as an image with rows of latitude."""
        return einx.id("(lon lat) -> lat lon", np.asarray(values), lon=100)

    lo, hi = 1875.0, 1925.0
    n_cells = int(ns["truth"].shape[0])
    fig, top = headed(
        "functax — a methane plume mapped from cloud-free pixels",
        f"The methane example as written: {n_pixels:,} pixels with 8 ppb of noise, "
        f"mapped onto all {n_cells:,} cells by project.Variational.\n"
        "The first three panels share one colour scale; the error is x̂ − x, red "
        "where the map is too high.",
        (13, 4.1),
    )
    # three panels and their shared bar, a gap, then the error and its bar
    grid = fig.add_gridspec(
        1,
        7,
        width_ratios=[1, 1, 1, 0.05, 0.3, 1, 0.05],
        left=0.05,
        right=0.955,
        bottom=0.13,
        top=top - 0.1,
        wspace=0.12,
    )
    axes = [fig.add_subplot(grid[0, i]) for i in (0, 1, 2, 5)]
    shared_bar, error_bar = fig.add_subplot(grid[0, 3]), fig.add_subplot(grid[0, 6])
    panels = [
        (ns["truth"], "truth x"),
        (None, "cloud-free pixels y"),
        (ns["xch4_map"], "map x̂ from the pixels"),
    ]
    for ax, (field, title) in zip(axes[:3], panels, strict=True):
        if field is None:
            pixels = np.asarray(ns["pixels"])
            shared = ax.scatter(
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
        ax.set_title(title, fontsize=10)
    bar = fig.colorbar(shared, cax=shared_bar)
    bar.set_label("XCH₄ (ppb)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    error = image(ns["map_error"])
    bound = float(einx.max("[lat lon]", np.abs(error)))
    shown = axes[3].imshow(
        error, origin="lower", extent=extent, vmin=-bound, vmax=bound, cmap="RdBu_r"
    )
    axes[3].set_title(
        f"error x̂ − x, RMS {ns['observed_rmse']:.2f} ppb\n"
        f"cloud-free, {ns['cloudy_rmse']:.2f} ppb cloudy",
        fontsize=9,
    )
    bar = fig.colorbar(shown, cax=error_bar)
    bar.set_label("ppb", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("longitude (°)", fontsize=8)
        ax.set_xticks([-104.0, -103.8, -103.6])
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("latitude (°)", fontsize=8)
    # the middle panels share the first panel's latitudes
    for ax in axes[1:3]:
        ax.tick_params(labelleft=False)
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


def sphere_points() -> tuple[jnp.ndarray, jnp.ndarray]:
    """The (λ, φ) of the plotting grid, and its unit vectors."""
    lonlat = jnp.asarray(einx.id("lat lon, lat lon -> (lat lon) (1 + 1)", LON, LAT))
    return lonlat, lonlat_to_unit(lonlat, unit="degrees")


MOLL = {"projection": "mollweide"}


def sh_basis_figure() -> None:
    """Six real spherical harmonics, each a global function on the sphere."""
    _, xyz = sphere_points()
    harmonics = np.asarray(real_spherical_harmonics(xyz, 4))
    fig, top = headed(
        "functax — real spherical harmonics, global on the sphere",
        "Six of the 25 functions of basis.sphere.real_spherical_harmonics(u, "
        "l_max=4), up to degree 4.\nRed is positive and blue negative; each panel "
        "is scaled to its own maximum.",
        (9, 4.4),
    )
    axes = fig.subplots(2, 3, subplot_kw=MOLL)
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=top - 0.05,
        wspace=0.04,
        hspace=0.22,
    )
    degrees_orders = [(1, 0), (2, 1), (2, -2), (3, 2), (4, 0), (4, 3)]
    for ax, (degree, order) in zip(axes.flat, degrees_orders, strict=True):
        column = degree * degree + degree + order
        mollweide(ax, harmonics[:, column], f"$Y_{{{degree}}}^{{{order}}}$")
    save(fig, "sh_basis.png")


def slepian_figure() -> None:
    """Six Slepian functions of a 40° cap, ranked by their concentration."""
    _, xyz = sphere_points()
    centre = jnp.deg2rad(jnp.array([-40.0, 25.0]))
    radius = np.deg2rad(40.0)
    basis = slepian_cap_basis(20, radius, lonlat_centre=centre)
    slepian = np.asarray(basis.evaluate(xyz))
    concentration = np.asarray(basis.eigenvalues)
    # A = 2π (1 − cos r), the cap's area on the unit sphere
    shannon = float(shannon_number(20, 2 * np.pi * (1 - np.cos(radius))))
    logger.info(
        "slepian: {} functions, Shannon number {:.1f}", slepian.shape[1], shannon
    )
    pole = np.asarray(lonlat_to_unit(einx.id("d -> 1 d", centre)))[0]
    # cos θ = ⟨u, p⟩, the angle from the cap centre
    cosine = np.clip(einx.dot("point d, d -> point", np.asarray(xyz), pole), -1, 1)
    angle = einx.id("(lat lon) -> lat lon", np.arccos(cosine), lat=LAT.shape[0])
    fig, top = headed(
        "functax — Slepian functions, concentrated in a region",
        f"Six of the {slepian.shape[1]} functions of basis.slepian.slepian_cap_basis("
        "l_max=20) for a 40° cap (dashed), ranked by λ,\nthe share of their energy "
        f"inside it. Past the Shannon number, about {shannon:.0f}, they leak out; "
        "each panel is scaled to its own maximum.",
        (9, 4.4),
    )
    axes = fig.subplots(2, 3, subplot_kw=MOLL)
    fig.subplots_adjust(
        left=0.01, right=0.99, bottom=0.01, top=top - 0.05, wspace=0.04, hspace=0.22
    )
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
    save(fig, "slepian.png")


def needlets_figure() -> None:
    """One needlet centre at four scales."""
    _, xyz = sphere_points()
    needlets, scales, centres = map(np.asarray, needlet_basis(xyz, B=2.0, j_max=3))
    point = np.asarray(lonlat_to_unit(jnp.array([[20.0, 10.0]]), unit="degrees"))[0]
    fig, top = headed(
        "functax — needlets, one centre at four scales",
        "The needlet of basis.sphere.needlet_basis(u, B=2, j_max=3) nearest "
        "(20°E, 10°N) at each scale j, each twice as fine as the last.\n"
        "Red is positive and blue negative; each panel is scaled to its own maximum.",
        (11, 2.55),
    )
    axes = fig.subplots(1, 4, subplot_kw=MOLL)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=top - 0.09, wspace=0.04)
    for ax, scale in zip(axes, range(4), strict=True):
        columns = np.where(scales == scale)[0]
        # the needlet of this scale whose centre is nearest the point
        closeness = einx.dot("atom d, d -> atom", centres[columns], point)
        nearest = columns[int(einx.argmax("[atom]", closeness)[0])]
        mollweide(ax, needlets[:, nearest], f"scale j = {scale}")
    save(fig, "needlets.png")


FIGURES = {
    "quickstart": check_quick_start,
    "example": example_figure,
    "sh_basis": sh_basis_figure,
    "slepian": slepian_figure,
    "needlets": needlets_figure,
}


@app.default
def main(*, only: list[str] | None = None) -> None:
    """Run the quickstart check and draw every figure, or only the ones named.

    Args:
        only: Names from quickstart, example, sh_basis, slepian and needlets.
    """
    for name in only or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    app()
