"""Render the README figures from the README's own code.

The quick start and the example run their Python blocks verbatim and in order,
then the example's figure is drawn from what they computed, so the numbers the
README quotes and the picture come from the same run (they are logged). The
gallery figures call the bases directly.

Run from the repo root (about a minute on a laptop CPU):

    uv run --group dev --with matplotlib python docs/assets/readme/figures.py
"""

from __future__ import annotations

import re
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from functax.basis.slepian import slepian_cap_basis
from functax.basis.sphere import needlet_basis, real_spherical_harmonics
from functax.geometry import lonlat_to_unit


OUT = Path(__file__).parent
README = OUT.parents[2] / "README.md"


def python_blocks(heading: str) -> list[str]:
    """The ```python blocks between ``## <heading>`` and the next ``## `` heading."""
    text = README.read_text()
    start = text.index(f"\n## {heading}")
    end = text.index("\n## ", start + 1)
    return re.findall(r"```python\n(.*?)```", text[start:end], flags=re.S)


def run(heading: str) -> dict:
    """Execute a README section's blocks in one namespace, as a reader would."""
    namespace: dict = {}
    for block in python_blocks(heading):
        exec(compile(block, "README.md", "exec"), namespace)
    return namespace


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- quick start


def check_quick_start() -> None:
    ns = run("Quick start")
    fit_error = float(
        jnp.abs(ns["design"] @ ns["coefficients"] - ns["observations"]).max()
    )
    logger.info("quick start: max fit error {:.1e}", fit_error)
    logger.info("quick start: values {}", ns["values"].tolist())
    logger.info(
        "quick start: integral {!r}, 4π/3 = {!r}", float(ns["integral"]), 4 * np.pi / 3
    )


# ---------------------------------------------------------------- example


def example_figure() -> None:
    ns = run("Example: map methane from one overpass")
    for key in ("read_rmse", "adjoint_gap", "observed_rmse", "cloudy_rmse"):
        logger.info("example: {} = {}", key, ns[key])
    for key in ("truth_integral", "map_integral", "pixel_integral"):
        logger.info("example: {} = {:.1f} ppb km²", key, float(ns[key]) / 1e6)
    logger.info(
        "example: max |map error| = {:.2f} ppb", float(abs(ns["map_error"]).max())
    )
    logger.info("example: N_Ωy = {}", int(ns["cloud_free"].sum()))
    logger.info("example: grid area {:.0f} km²", float(ns["area"].weights.sum()) / 1e6)

    extent = (-104.2, -103.4, 31.6, 32.4)

    def image(values: jnp.ndarray) -> np.ndarray:
        # (λ, φ) row-major → rows of φ, for imshow
        return np.asarray(values).reshape(100, 100).T

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
    bound = float(np.abs(error).max())
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
    bound = float(np.nanmax(np.abs(field)))
    ax.pcolormesh(
        np.deg2rad(LON),
        np.deg2rad(LAT),
        np.asarray(field).reshape(LAT.shape),
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
    lonlat = jnp.stack([LON.ravel(), LAT.ravel()], -1)
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
    pole = np.asarray(lonlat_to_unit(centre[None]))[0]
    angle = np.arccos(np.clip(np.asarray(xyz) @ pole, -1, 1)).reshape(LAT.shape)
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
        nearest = columns[np.argmax(centres[columns] @ point)]
        mollweide(ax, needlets[:, nearest], f"needlet, scale j = {scale}")
    save(fig, "needlets.png")


if __name__ == "__main__":
    check_quick_start()
    example_figure()
    gallery_figures()
