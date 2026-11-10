"""Conservative remapping (#40, #41): conservation, block means, adjoint, oracles.

Over a second each, mostly einx compiling each new shape in an eager call.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import (
    assert_adjoint,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
    assert_transpose_matches,
)

from functax._src.interp._clip import planar_polygon_area
from functax._src.interp._conservative import _node_corners, _polygon_cells
from functax.geometry import (
    AffineGrid,
    Axis,
    CurvilinearGrid,
    RectilinearGrid,
    lonlat_to_unit,
    spherical_polygon_area,
    unit_to_lonlat,
)
from functax.interp import Conservative, Policies


def random_edges(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    inner = np.sort(rng.uniform(lo, hi, n - 1))
    return np.concatenate([[lo], inner, [hi]])


def axis_from_edges(edges: np.ndarray, *, period: float | None = None) -> Axis:
    centres = 0.5 * (edges[:-1] + edges[1:])
    return Axis(jnp.asarray(centres), edges=jnp.asarray(edges), period=period)


def grid_from_edges(*edges: np.ndarray) -> RectilinearGrid:
    return RectilinearGrid(tuple(axis_from_edges(e) for e in edges))


def cell_areas(*edges: np.ndarray) -> np.ndarray:
    lengths = [np.abs(np.diff(e)) for e in edges]
    return np.prod(np.meshgrid(*lengths, indexing="ij"), axis=0).ravel()


def source_cell_areas(source: object, *, sphere: bool, unit: str | None) -> np.ndarray:
    """|cⱼ| of the plan's own source cells, for the conservation checks."""
    cells = _polygon_cells(source, "source", unit if sphere else None)
    if sphere:
        return np.abs(np.asarray(spherical_polygon_area(jnp.asarray(cells))))
    counts = np.full(len(cells), 4)
    return np.abs(planar_polygon_area(cells, counts))


def warped_swath(
    n_rows: int, n_cols: int, *, scale: float, shift: tuple
) -> CurvilinearGrid:
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    x = shift[0] + scale * (cols + 0.15 * rows + 0.02 * rows**2)
    y = shift[1] + scale * (rows + 0.1 * np.sin(cols))
    return CurvilinearGrid(x=jnp.asarray(x), y=jnp.asarray(y))


class TestConservation:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_integrals_are_preserved_on_random_rectilinear_grids(
        self, seed: int
    ) -> None:
        # gate G3-conserve: ∑ᵢ |Cᵢ| ŷᵢ = ∑ⱼ |cⱼ| yⱼ when the target covers the source
        rng = np.random.default_rng(seed)
        source = (random_edges(rng, 13, 0.0, 5.0), random_edges(rng, 17, -2.0, 3.0))
        target = (random_edges(rng, 6, 0.0, 5.0), random_edges(rng, 9, -2.0, 3.0))
        plan = Conservative().plan(grid_from_edges(*source), grid_from_edges(*target))
        field = rng.normal(size=(4, 13 * 17))
        remapped = plan(jnp.asarray(field))
        source_integral = field @ cell_areas(*source)
        target_integral = np.asarray(remapped) @ cell_areas(*target)
        assert_matches_oracle(target_integral, source_integral, rtol=1e-12)

    def test_a_wider_target_conserves_under_extrapolate(self) -> None:
        rng = np.random.default_rng(3)
        source = (random_edges(rng, 9, 0.0, 4.0), random_edges(rng, 11, 0.0, 4.0))
        target = (random_edges(rng, 5, -1.0, 5.0), random_edges(rng, 4, -1.0, 5.0))
        plan = Conservative().plan(
            grid_from_edges(*source),
            grid_from_edges(*target),
            policies=Policies(outside="extrapolate", fill_value=0.0),
        )
        field = rng.normal(size=9 * 11)
        remapped = np.asarray(plan(jnp.asarray(field)))
        assert_matches_oracle(
            remapped @ cell_areas(*target), field @ cell_areas(*source), rtol=1e-12
        )

    @pytest.mark.parametrize("factor", [2, 4])
    def test_integer_factor_coarsening_is_a_block_mean_exactly(
        self, factor: int
    ) -> None:
        # dyadic edges and integer data: every product and sum is exact
        fine = RectilinearGrid(
            (Axis(jnp.arange(8.0) + 0.5), Axis(jnp.arange(16.0) + 0.5))
        )
        coarse = RectilinearGrid(
            (
                Axis(jnp.arange(8.0 / factor) * factor + factor / 2),
                Axis(jnp.arange(16.0 / factor) * factor + factor / 2),
            )
        )
        rng = np.random.default_rng(4)
        field = rng.integers(-50, 50, size=(8, 16)).astype(np.float64)
        plan = Conservative().plan(fine, coarse)
        remapped = plan(jnp.asarray(field.ravel()))
        block_mean = field.reshape(8 // factor, factor, 16 // factor, factor).mean(
            axis=(1, 3)
        )
        np.testing.assert_array_equal(np.asarray(remapped), block_mean.ravel())
        assert plan.read.n_taps == factor * factor

    def test_a_factor_of_three_is_a_block_mean_to_rounding(self) -> None:
        fine = Axis(jnp.arange(9.0) + 0.5)
        coarse = Axis(jnp.array([1.5, 4.5, 7.5]))
        field = np.random.default_rng(5).normal(size=9)
        remapped = Conservative().plan(fine, coarse)(jnp.asarray(field))
        # 1/3 is not exact, so the weights round once per tap
        assert_matches_oracle(remapped, field.reshape(3, 3).mean(axis=1), rtol=1e-15)

    def test_refining_then_coarsening_returns_the_field(self) -> None:
        rng = np.random.default_rng(6)
        coarse_edges = (random_edges(rng, 5, 0.0, 1.0), random_edges(rng, 4, 0.0, 1.0))
        # every coarse edge is a fine edge, so each fine cell sits in one coarse cell
        fine_edges = tuple(
            np.unique(np.concatenate([e, rng.uniform(0.0, 1.0, 9)]))
            for e in coarse_edges
        )
        coarse = grid_from_edges(*coarse_edges)
        fine = grid_from_edges(*fine_edges)
        field = jnp.asarray(rng.normal(size=5 * 4))
        refined = Conservative().plan(coarse, fine)(field)
        back = Conservative().plan(fine, coarse)(refined)
        assert_matches_oracle(back, field, rtol=1e-13)


class TestAdjoint:
    def test_dot_product(self) -> None:
        rng = np.random.default_rng(7)
        source = grid_from_edges(
            random_edges(rng, 7, 0.0, 3.0), random_edges(rng, 9, 0.0, 3.0)
        )
        target = grid_from_edges(
            random_edges(rng, 4, -0.5, 3.5), random_edges(rng, 5, 0.5, 2.5)
        )
        plan = Conservative().plan(source, target)
        assert_adjoint(
            plan, plan.adjoint, 7 * 9, 4 * 5, key=jax.random.key(0), rtol=1e-12
        )
        assert_transpose_matches(plan, plan.adjoint, jnp.zeros(7 * 9))


class TestGeometries:
    def test_a_periodic_source_wraps_across_the_seam(self) -> None:
        longitude = Axis(jnp.arange(5.0, 360.0, 10.0), period=360.0)
        shifted = Axis(jnp.array([0.0, 180.0]), edges=jnp.array([-10.0, 10.0, 350.0]))
        field = jnp.arange(36.0)
        remapped = Conservative().plan(longitude, shifted)(field)
        # [−10, 10] reads the cells at 355° and 5°, [10, 350] the 34 between
        assert_matches_oracle(remapped, [17.5, 17.5], rtol=1e-15)

    def test_the_plan_windows_a_grid_target(self) -> None:
        source = RectilinearGrid((Axis(jnp.arange(8.0)),) * 2)
        target = RectilinearGrid((Axis(jnp.arange(0.5, 7.0, 2.0)),) * 2)
        plan = Conservative().plan(source, target)
        field = jnp.arange(64.0)
        window = plan[1:3, 0:2](field)
        expected = einx.id("(y x) -> y x", plan(field), y=4)[1:3, 0:2]
        np.testing.assert_array_equal(np.asarray(window), np.asarray(expected).ravel())

    def test_default_edges_are_midpoints(self) -> None:
        # source cells [−½, ½], [½, 2], [2, 4]
        source = Axis(jnp.array([0.0, 1.0, 3.0]))
        target = Axis(jnp.array([0.0, 3.0]), edges=jnp.array([-0.5, 1.0, 4.0]))
        weights = Conservative().plan(source, target).read.to_dense()
        expected = [[1.0 / 1.5, 0.5 / 1.5, 0.0], [0.0, 1.0 / 3.0, 2.0 / 3.0]]
        assert_matches_oracle(weights, expected, rtol=1e-15)

    def test_descending_axes(self) -> None:
        rng = np.random.default_rng(8)
        edges = random_edges(rng, 6, 0.0, 1.0)
        target_edges = np.array([0.0, 0.3, 1.0])
        field = jnp.asarray(rng.normal(size=6))
        ascending = Conservative().plan(
            axis_from_edges(edges), axis_from_edges(target_edges)
        )(field)
        descending = Conservative().plan(
            axis_from_edges(edges[::-1].copy()),
            axis_from_edges(target_edges[::-1].copy()),
        )(field[::-1])
        assert_matches_oracle(descending, ascending[::-1], rtol=1e-14)

    def test_a_north_up_affine_grid_equals_its_rectilinear_grid(self) -> None:
        source = AffineGrid([2.0, 0.0, 10.0, 0.0, -1.0, 8.0], (8, 6))
        target = AffineGrid([3.0, 0.0, 10.0, 0.0, -2.0, 8.0], (4, 4))
        rows = grid_from_edges(8.0 - np.arange(9.0), 10.0 + 2.0 * np.arange(7.0))
        target_rows = grid_from_edges(
            8.0 - 2.0 * np.arange(5.0), 10.0 + 3.0 * np.arange(5.0)
        )
        field = jnp.asarray(np.random.default_rng(9).normal(size=48))
        affine = Conservative().plan(source, target)
        rectilinear = Conservative().plan(rows, target_rows)
        assert affine.target_shape == (4, 4)
        assert_matches_oracle(affine(field), rectilinear(field), rtol=1e-15)

    def test_keeps_float32_weights(self) -> None:
        axis = Axis(jnp.arange(4.0, dtype=jnp.float32))
        plan = Conservative().plan(axis, Axis(jnp.array([0.5, 2.5], jnp.float32)))
        assert plan.read.weights.dtype == jnp.float32


def test_jit_and_vmap_agree_with_a_loop() -> None:
    rng = np.random.default_rng(10)
    source = grid_from_edges(
        random_edges(rng, 5, 0.0, 1.0), random_edges(rng, 6, 0.0, 1.0)
    )
    target = grid_from_edges(
        random_edges(rng, 3, 0.0, 1.0), random_edges(rng, 4, 0.0, 1.0)
    )
    plan = Conservative().plan(source, target, policies=Policies(missing="renormalise"))
    fields = rng.normal(size=(3, 30))
    fields[rng.uniform(size=fields.shape) < 0.2] = np.nan
    assert_traces(lambda field: plan(field), jnp.asarray(fields))


def test_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(11)
    source = grid_from_edges(
        random_edges(rng, 4, 0.0, 1.0), random_edges(rng, 5, 0.0, 1.0)
    )
    target = grid_from_edges(
        random_edges(rng, 3, 0.0, 1.0), random_edges(rng, 2, 0.0, 1.0)
    )
    plan = Conservative().plan(source, target)
    assert_grad_matches_fd(plan, jnp.asarray(rng.normal(size=20)))


class TestCurvilinear:
    def test_curvilinear_to_regular_conserves_the_integral(self) -> None:
        # DoD of #41: 1e-10 relative; the target covers the swath, and
        # extrapolate keeps every source cell's area in some target cell
        swath = warped_swath(9, 12, scale=1.0, shift=(0.0, 0.0))
        grid = RectilinearGrid(
            (
                Axis(jnp.linspace(-2.0, 18.0, 11)),
                Axis(jnp.linspace(-2.0, 11.0, 8)),
            )
        )
        keep = Policies(outside="extrapolate", fill_value=0.0)
        plan = Conservative().plan(swath, grid, policies=keep)
        field = np.random.default_rng(12).normal(size=9 * 12)
        remapped = np.asarray(plan(jnp.asarray(field)))
        target_areas = cell_areas(
            *(np.asarray(axis.cell_edges()) for axis in grid.axes)
        )
        source_areas = source_cell_areas(swath, sphere=False, unit=None)
        assert_matches_oracle(remapped @ target_areas, field @ source_areas, rtol=1e-10)

    def test_the_polygon_path_equals_the_tensor_path(self) -> None:
        rng = np.random.default_rng(13)
        target = grid_from_edges(
            random_edges(rng, 4, -0.5, 3.5), random_edges(rng, 3, -0.5, 2.5)
        )
        # a uniform grid, so its midpoint corners are the tensor grid's edges
        x_nodes = np.linspace(0.25, 2.75, 6)
        y_nodes = np.linspace(0.25, 1.75, 4)
        tensor = RectilinearGrid(
            (Axis(jnp.asarray(x_nodes)), Axis(jnp.asarray(y_nodes)))
        )
        x, y = np.meshgrid(x_nodes, y_nodes, indexing="ij")
        swath = CurvilinearGrid(x=jnp.asarray(x), y=jnp.asarray(y))
        policies = Policies(outside="extrapolate")
        field = jnp.asarray(rng.normal(size=24))
        expected = Conservative().plan(tensor, target, policies=policies)(field)
        actual = Conservative().plan(swath, target, policies=policies)(field)
        assert_matches_oracle(actual, expected, rtol=1e-12, atol=1e-15)

    def test_dot_product(self) -> None:
        swath = warped_swath(5, 6, scale=1.0, shift=(0.0, 0.0))
        grid = RectilinearGrid(
            (Axis(jnp.linspace(0.0, 8.0, 5)), Axis(jnp.linspace(0.0, 5.0, 4)))
        )
        plan = Conservative().plan(swath, grid, policies=Policies(outside="clamp"))
        assert_adjoint(plan, plan.adjoint, 30, 20, key=jax.random.key(1))
        assert_transpose_matches(plan, plan.adjoint, jnp.zeros(30))


class TestSphere:
    def test_a_lonlat_swath_conserves_on_the_sphere(self) -> None:
        # a rotated, warped 0.5° swath over Europe onto a 2° lon/lat grid
        swath = warped_swath(20, 24, scale=0.5, shift=(-5.0, 40.0))
        grid = RectilinearGrid(
            (Axis(jnp.arange(-9.0, 18.0, 2.0)), Axis(jnp.arange(37.0, 56.0, 2.0)))
        )
        scheme = Conservative(sphere=True, unit="degrees")
        keep = Policies(outside="extrapolate", fill_value=0.0)
        plan = scheme.plan(swath, grid, policies=keep)
        field = np.random.default_rng(14).normal(size=20 * 24)
        remapped = np.asarray(plan(jnp.asarray(field)))
        target_cells = _polygon_cells(grid, "target", "degrees")
        target_areas = np.abs(
            np.asarray(spherical_polygon_area(jnp.asarray(target_cells)))
        )
        source_areas = source_cell_areas(swath, sphere=True, unit="degrees")
        assert_matches_oracle(remapped @ target_areas, field @ source_areas, rtol=1e-10)

    def test_global_lonlat_grids_tile_the_sphere(self) -> None:
        source = RectilinearGrid(
            (Axis(jnp.arange(5.0, 360.0, 10.0)), Axis(jnp.arange(-85.0, 90.0, 10.0)))
        )
        target = RectilinearGrid(
            (
                Axis(jnp.arange(-172.5, 180.0, 15.0), period=360.0),
                Axis(jnp.arange(-82.5, 90.0, 15.0)),
            )
        )
        plan = Conservative(sphere=True, unit="degrees").plan(source, target)
        assert bool(plan.valid.all())
        target_cells = _polygon_cells(target, "target", "degrees")
        target_areas = np.abs(
            np.asarray(spherical_polygon_area(jnp.asarray(target_cells)))
        )
        assert_matches_oracle(target_areas.sum(), 4 * np.pi, rtol=1e-12)
        field = np.random.default_rng(15).normal(size=36 * 18)
        remapped = np.asarray(plan(jnp.asarray(field)))
        source_areas = source_cell_areas(source, sphere=True, unit="degrees")
        assert_matches_oracle(remapped @ target_areas, field @ source_areas, rtol=1e-10)
        assert_matches_oracle(plan(jnp.ones(36 * 18)), np.ones(24 * 12), rtol=1e-10)


def test_matches_xesmf_on_a_curvilinear_grid() -> None:
    # DoD of #41: xESMF's conservative remap (great-circle cells, destarea) to
    # 1e-8 on target cells inside the swath; skipped where xESMF is not installed
    xesmf = pytest.importorskip("xesmf")
    xarray = pytest.importorskip("xarray")
    swath = warped_swath(16, 20, scale=0.5, shift=(-5.0, 40.0))
    grid = RectilinearGrid(
        (Axis(jnp.arange(-3.0, 8.0, 1.0)), Axis(jnp.arange(42.0, 48.0, 1.0)))
    )
    # the same corners as the plan, handed to ESMF as lon_b and lat_b
    nodes = np.stack([np.asarray(swath.x), np.asarray(swath.y)], axis=-1)
    unit_nodes = np.asarray(lonlat_to_unit(jnp.asarray(nodes), unit="degrees"))
    corners = unit_to_lonlat(
        jnp.asarray(_node_corners(unit_nodes, True)), unit="degrees"
    )
    lon_edges, lat_edges = (np.asarray(axis.cell_edges()) for axis in grid.axes)
    source_ds = xarray.Dataset(
        coords={
            "lon": (("y", "x"), nodes[..., 0]),
            "lat": (("y", "x"), nodes[..., 1]),
            "lon_b": (("y_b", "x_b"), np.asarray(corners[..., 0])),
            "lat_b": (("y_b", "x_b"), np.asarray(corners[..., 1])),
        }
    )
    target_ds = xarray.Dataset(
        coords={
            "lon": ("lon", np.asarray(grid.axes[0].values)),
            "lat": ("lat", np.asarray(grid.axes[1].values)),
            "lon_b": ("lon_b", lon_edges),
            "lat_b": ("lat_b", lat_edges),
        }
    )
    regridder = xesmf.Regridder(source_ds, target_ds, "conservative")
    field = np.random.default_rng(16).normal(size=(16, 20))
    expected = np.asarray(regridder(xarray.DataArray(field, dims=("y", "x"))))
    plan = Conservative(sphere=True, unit="degrees").plan(swath, grid)
    actual = einx.id("(x y) -> y x", plan(jnp.asarray(field.ravel())), x=11)
    inside = np.asarray(einx.id("(x y) -> y x", plan.valid, x=11))
    assert inside.any()
    assert_matches_oracle(actual, expected, rtol=1e-8, where=inside)


class TestPolicies:
    source = Axis(jnp.arange(4.0) + 0.5)
    target = Axis(
        jnp.array([0.0, 2.0, 3.5, 5.0]), edges=jnp.array([-1.0, 1.0, 3.0, 4.0, 6.0])
    )
    field = jnp.array([1.0, 2.0, 3.0, 4.0])

    def test_nan_marks_partly_covered_cells_invalid(self) -> None:
        plan = Conservative().plan(self.source, self.target)
        assert plan.valid.tolist() == [False, True, True, False]
        assert_matches_oracle(plan(self.field), [np.nan, 2.5, 4.0, np.nan], rtol=1e-15)

    def test_renormalise_averages_the_valid_covered_area(self) -> None:
        plan = Conservative().plan(
            self.source, self.target, policies=Policies(missing="renormalise")
        )
        masked = self.field.at[2].set(jnp.nan)
        # a coastal mask: cell [1, 3] reads only its valid half
        assert_matches_oracle(plan(masked), [1.0, 2.0, 4.0, np.nan], rtol=1e-15)


class TestPolygonPath:
    @staticmethod
    def swath(y: np.ndarray, x: np.ndarray) -> CurvilinearGrid:
        rows, cols = np.meshgrid(y, x, indexing="ij")
        return CurvilinearGrid(x=jnp.asarray(cols), y=jnp.asarray(rows))

    def test_degenerate_cells_give_zero_weight(self) -> None:
        # three equal node rows: the middle one's cells have no area
        collapsed = self.swath(np.array([0.0, 1.0, 1.0, 1.0, 2.0]), np.arange(3.0))
        grid = RectilinearGrid(
            (
                Axis(jnp.array([0.0, 2.0]), edges=jnp.array([-1.0, 1.0, 3.0])),
                Axis(jnp.array([0.0, 2.0]), edges=jnp.array([-1.0, 1.0, 3.0])),
            )
        )
        keep = Policies(outside="extrapolate")
        plan = Conservative().plan(collapsed, grid, policies=keep)
        weights = np.asarray(plan.read.to_dense())
        middle = einx.id("(y x) -> y x", np.arange(15), y=5)[2]
        assert np.all(weights[:, middle] == 0.0)
        assert np.all(weights.sum(axis=0)[np.setdiff1d(np.arange(15), middle)] > 0)
        # as a target, a cell of no area is invalid
        plan = Conservative().plan(grid, collapsed, policies=Policies(outside="clamp"))
        rows = einx.id("(y x) -> y x", plan.valid, y=5)
        assert not np.any(np.asarray(rows[2]))
        assert np.all(np.asarray(plan.read.weights)[middle] == 0.0)
