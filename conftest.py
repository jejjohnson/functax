"""Shared pytest configuration.

The oracle comparisons run in float64, so JAX's 64-bit mode is enabled for the
whole suite, and jaxtyping checks every annotated array at runtime through its
import hook (`python-style`). The guard keeps the suite importable in a
NumPy-only install, where only the JAX-free modules can be tested.
"""

from __future__ import annotations


try:
    import jax
    from jaxtyping import install_import_hook
except ImportError:  # pragma: no cover - exercised only in a NumPy-only install
    pass
else:
    jax.config.update("jax_enable_x64", True)
    # Installed before pytest imports functax for doctest collection, so every
    # functax module imported afterwards is checked with beartype.
    install_import_hook("functax", "beartype.beartype")


# Every test carries exactly one tier marker, from where it lives (see "Test
# tiers" in AGENTS.md):
#
# - tests/fast/ -> ``fast``: under ~0.3 s each on CI, for the edit-test loop;
# - tests/slow/ -> ``slow``: sweeps, oracle, dot-product, gradient and trace
#   checks, anything that compiles many new shapes;
# - tests/integration/ -> ``integration``: subprocesses, other libraries'
#   pinned outputs, workflows, the built site.
#
# Doctests in src/functax are ``fast``, except the examples listed in
# SLOW_DOCTESTS, which run over ~0.3 s on CI and are ``slow``. A test changes
# tier by moving (or, for a doctest, by joining or leaving that list); a
# hand-written tier marker is an error, so the marks cannot drift from the
# directories. Select a tier with ``-m fast``, ``-m slow`` or ``-m integration``.
TIERS = ("fast", "slow", "integration")

# Doctests (by their pytest name, the dotted path of the documented object)
# measured over ~0.3 s on CI. Keep the list sorted.
SLOW_DOCTESTS: frozenset[str] = frozenset(
    {
        "functax._src.basis._divfree.curlfree_basis",
        "functax._src.basis._divfree.divfree_basis",
        "functax._src.basis._eof.eof_basis",
        "functax._src.basis._fourier.fourier_basis",
        "functax._src.basis._fourier.fourier_eigenvalues",
        "functax._src.basis._fourier.fourier_eigenvalues_1d",
        "functax._src.basis._gabor.gabor_frame",
        "functax._src.basis._gabor.gabor_frame_grid",
        "functax._src.basis._legendre.associated_legendre",
        "functax._src.basis._legendre.legendre_series",
        "functax._src.basis._mesh_laplacian.cotangent_laplacian_eigpairs",
        "functax._src.basis._needlets.needlet_basis",
        "functax._src.basis._needlets.needlet_window",
        "functax._src.basis._rbf.rbf_basis",
        "functax._src.basis._rbf.spherical_rbf_basis",
        "functax._src.basis._shell.shell_basis",
        "functax._src.basis._slepian_cap.SlepianCapBasis",
        "functax._src.basis._slepian_cap.SlepianCapBasis.centred_coordinates",
        "functax._src.basis._slepian_cap.SlepianCapBasis.evaluate",
        "functax._src.basis._slepian_cap.SlepianCapBasis.num_modes",
        "functax._src.basis._slepian_cap.SlepianCapBasis.rotate_to",
        "functax._src.basis._slepian_cap.slepian_cap_basis",
        "functax._src.basis._slepian_cap.slepian_cap_eigh_per_m",
        "functax._src.basis._slepian_cap.slepian_concentration_matrix",
        "functax._src.basis._slepian_region.SlepianBasis",
        "functax._src.basis._slepian_region.SlepianBasis.evaluate",
        "functax._src.basis._slepian_region.SlepianBasis.num_modes",
        "functax._src.basis._slepian_region.slepian_polar_gap_basis",
        "functax._src.basis._slepian_region.slepian_region_basis",
        "functax._src.basis._slepian_region.spherical_polygon_mask",
        "functax._src.basis._spherical_harmonics.real_spherical_harmonics",
        "functax._src.basis._vector_harmonics.vector_spherical_harmonics",
        "functax._src.diffops._fields.divergence",
        "functax._src.diffops._sphere.laplace_beltrami",
        "functax._src.diffops._sphere.surface_grad",
        "functax._src.geometry._frames.ecef_to_lonlat",
        "functax._src.geometry._metric.pairwise_distance",
        "functax._src.geometry._sphere.lonlat_to_unit",
        "functax._src.geometry._sphere.spherical_polygon_area",
        "functax._src.geometry._types.Axis",
        "functax._src.geometry._types.RectilinearGrid",
        "functax._src.interp._aggregate.Binning",
        "functax._src.interp._aggregate.TransposedStencil",
        "functax._src.interp._bases.BSplineBasis",
        "functax._src.interp._compose.compose",
        "functax._src.interp._conservative.Conservative",
        "functax._src.interp._covariance.BrownianBridge",
        "functax._src.interp._covariance.KernelCovariance",
        "functax._src.interp._curvilinear.locate_curvilinear",
        "functax._src.interp._fit.KernelSolve",
        "functax._src.interp._gaussian.GaussianPlan",
        "functax._src.interp._interpolant.Interpolant",
        "functax._src.interp._kernels.Wendland",
        "functax._src.interp._kriging.Kriging",
        "functax._src.interp._locate.index_positions",
        "functax._src.interp._neighbourhood.InverseDistance",
        "functax._src.interp._plan.ComposedPlan",
        "functax._src.interp._plan.Plan.derivative",
        "functax._src.interp._plan.Plan.deserialize",
        "functax._src.interp._plan.WindowedPlan",
        "functax._src.interp._prefilter.BandedPrefilter",
        "functax._src.interp._rbf.RBF",
        "functax._src.interp._schemes.BSpline",
        "functax._src.interp._schemes.Linear",
        "functax._src.interp._schemes.Nearest",
        "functax._src.interp._schemes.Tensor",
        "functax._src.interp._series.Akima",
        "functax._src.interp._series.CubicSpline",
        "functax._src.interp._series.Pchip",
        "functax._src.interp._series.SeriesInterpolant",
        "functax._src.interp._spectral.Chebyshev",
        "functax._src.interp._spectral.ChebyshevInterpolant",
        "functax._src.interp._spectral.Fourier",
        "functax._src.interp._spectral.FourierReadout",
        "functax._src.interp._spectral.clenshaw",
        "functax._src.interp._stencil.Stencil.__matmul__",
        "functax._src.interp._triangulation.Delaunay",
        "functax._src.project._fill.fill_gaps",
        "functax._src.project._linear.galerkin",
        "functax._src.project._linear.least_squares",
        "functax._src.project._variational.Variational",
        "functax._src.quad._cell_area.cell_area",
        "functax._src.quad._curve.polyline",
        "functax._src.transform._chebyshev.ChebyshevTransform1D",
        "functax._src.transform._chebyshev.ChebyshevTransform2D",
        "functax._src.transform._nufft.nonuniform_irfft",
        "functax._src.transform._nufft.nonuniform_rfft",
        "functax._src.transform._sht.SphericalHarmonicTransform",
        "functax._src.transform._sht.SphericalHarmonicTransform.forward",
        "functax._src.transform._sht.SphericalHarmonicTransform.init",
        "functax._src.transform._sht.SphericalHarmonicTransform.inverse",
        "functax.exceptions.ValidationError",
    }
)


def _tier(item) -> str:  # type: ignore[no-untyped-def]
    """The tier of a collected test: its directory under tests/, or fast/slow
    for a doctest."""
    parts = item.path.parts
    if "tests" in parts:
        directory = parts[parts.index("tests") + 1]
        if directory in TIERS:
            return directory
    return "slow" if item.name in SLOW_DOCTESTS else "fast"


def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    """Mark every test with its tier, and reject a hand-written tier marker."""
    import pytest

    for item in items:
        tier = _tier(item)
        declared = {name for name in TIERS if item.get_closest_marker(name)}
        if declared - {tier}:
            msg = (
                f"{item.nodeid} is marked {sorted(declared)} but its tier is "
                f"{tier!r}; tiers come from the directory (tests/fast, tests/slow, "
                "tests/integration) and SLOW_DOCTESTS in conftest.py, never from a "
                "hand-written marker"
            )
            raise pytest.UsageError(msg)
        item.add_marker(getattr(pytest.mark, tier))
