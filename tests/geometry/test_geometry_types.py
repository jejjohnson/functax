"""Tests for the CRS-free geometry types in `functax.geometry`."""

from __future__ import annotations

import types

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_traces

from functax import GeometryError
from functax.geometry import AffineGrid, Axis, CurvilinearGrid, Points, RectilinearGrid


NORTH_UP = [20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0]


def _stack(*modules: eqx.Module) -> eqx.Module:
    """Stack equal-structure modules along a new leading batch axis."""
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *modules)


def _assert_same_tree(actual: object, expected: object) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for got, want in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert got.dtype == want.dtype
        np.testing.assert_array_equal(got, want)


class TestAxis:
    def test_keeps_the_input_dtype(self) -> None:
        axis = Axis(jnp.linspace(0.0, 1.0, 5, dtype=jnp.float32))
        assert axis.values.dtype == jnp.float32

    def test_accepts_a_list(self) -> None:
        assert Axis([0.0, 1.0, 2.0]).values.shape == (3,)

    @pytest.mark.parametrize(
        ("values", "regular"),
        [
            (jnp.linspace(-90.0, 90.0, 181), True),
            (jnp.arange(600000.0, 612000.0, 20.0), True),
            (jnp.logspace(0.0, 2.0, 10), False),
            (jnp.array([0.0, 1.0, 2.0 + 1e-9]), False),
        ],
    )
    def test_is_regular(self, values: jax.Array, regular: bool) -> None:
        assert Axis(values).is_regular is regular

    def test_is_descending_on_a_concrete_axis_under_jit(self) -> None:
        axis = Axis(jnp.array([3.0, 2.0, 1.0]))
        flipped = jax.jit(lambda x: -x if axis.is_descending else x)(jnp.ones(2))
        assert flipped.tolist() == [-1.0, -1.0]

    def test_descending_axes_are_allowed(self) -> None:
        axis = Axis(jnp.array([3.0, 2.0, 1.0]))
        assert axis.is_descending
        assert axis.is_regular

    def test_period_is_kept_as_a_float(self) -> None:
        axis = Axis(jnp.linspace(0.0, 350.0, 36), period=360)
        assert axis.period == 360.0
        assert isinstance(axis.period, float)

    @pytest.mark.parametrize(
        ("values", "match"),
        [
            (jnp.zeros((2, 2)), r"'values' must be 1-D, got shape \(2, 2\)"),
            (jnp.array([1.0]), "'values' must have at least two nodes, got 1"),
            (jnp.array([0, 1, 2]), "'values' must have a floating dtype, got int"),
            (jnp.array([0.0, 2.0, 1.0]), "'values' must be strictly monotone"),
            (jnp.array([0.0, 1.0, 1.0]), "'values' must be strictly monotone"),
            (jnp.array([0.0, jnp.nan, 2.0]), "'values' must be strictly monotone"),
        ],
    )
    def test_rejects_bad_values(self, values: jax.Array, match: str) -> None:
        with pytest.raises(GeometryError, match=match):
            Axis(values)

    def test_the_message_shows_the_bad_values(self) -> None:
        with pytest.raises(GeometryError, match=r"got array\(\[0\., 2\., 1\.\]\)"):
            Axis(jnp.array([0.0, 2.0, 1.0]))

    @pytest.mark.parametrize("period", [0.0, -360.0, float("nan"), float("inf")])
    def test_rejects_bad_periods(self, period: float) -> None:
        with pytest.raises(GeometryError, match="'period' must be positive and finite"):
            Axis(jnp.array([0.0, 1.0]), period=period)

    def test_rejects_nodes_spanning_the_period(self) -> None:
        with pytest.raises(GeometryError, match=r"'values' must span less than 360\.0"):
            Axis(jnp.linspace(0.0, 360.0, 5), period=360.0)

    def test_checks_values_when_built_under_jit(self) -> None:
        build = jax.jit(lambda values: Axis(values).values)
        np.testing.assert_array_equal(build(jnp.array([0.0, 1.0])), [0.0, 1.0])
        with pytest.raises(Exception, match="'values' must be strictly monotone"):
            build(jnp.array([1.0, 0.0, 2.0])).block_until_ready()

    def test_rejects_a_bad_epoch(self) -> None:
        with pytest.raises(GeometryError, match="'epoch' must be a datetime64"):
            Axis(jnp.array([0.0, 1.0]), epoch="not a time")
        with pytest.raises(GeometryError, match="'epoch' must not be NaT"):
            Axis(jnp.array([0.0, 1.0]), epoch=np.datetime64("NaT"))


class TestDatetimes:
    def test_epoch_defaults_to_the_first_time(self) -> None:
        times = np.array(["2026-03-01T06", "2026-03-01T12"], dtype="datetime64[h]")
        axis = Axis.from_datetimes(times)
        assert axis.epoch == "2026-03-01T06:00:00.000000000"
        assert axis.values.tolist() == [0.0, 21600.0]
        assert axis.values.dtype == jnp.float64

    def test_an_explicit_epoch_allows_negative_offsets(self) -> None:
        times = np.array(
            ["2025-12-31T23:59:59", "2026-01-01T00:00:01"], dtype="datetime64[s]"
        )
        axis = Axis.from_datetimes(times, epoch="2026-01-01")
        assert axis.values.tolist() == [-1.0, 1.0]

    def test_round_trips_nanoseconds_within_48_days(self) -> None:
        # Float64 seconds resolve 1 ns only while |t| < 2²² s (about 48.5 days):
        # there the spacing of doubles is 2⁻³⁰ s ≈ 0.93 ns, under the 1 ns grid.
        rng = np.random.default_rng(0)
        limit = 2**22 * 1_000_000_000
        offsets = np.sort(rng.integers(-limit, limit, size=2000))
        epoch = np.datetime64("2026-01-01", "ns")
        times = epoch + np.unique(offsets).astype("timedelta64[ns]")
        axis = Axis.from_datetimes(times, epoch=epoch)
        np.testing.assert_array_equal(axis.to_datetimes(unit="ns"), times)

    def test_round_trips_microseconds_within_68_years(self) -> None:
        # Below 2³¹ s (about 68 years) doubles are spaced 2⁻²¹ s ≈ 0.48 µs, so
        # rounding to the microsecond recovers the input exactly. Nanoseconds
        # over ±100 years would need 62 bits, more than float64's 53.
        rng = np.random.default_rng(1)
        limit = 2**31 * 1_000_000
        offsets = np.unique(rng.integers(-limit, limit, size=2000))
        epoch = np.datetime64("2026-01-01", "ns")
        times = epoch + (offsets * 1_000).astype("timedelta64[ns]")
        axis = Axis.from_datetimes(times, epoch=epoch)
        np.testing.assert_array_equal(axis.to_datetimes(unit="us"), times)

    def test_rounds_half_up(self) -> None:
        axis = Axis(jnp.array([0.5, 1.5]), epoch="2026-01-01")
        rounded = axis.to_datetimes(unit="s")
        assert rounded.astype(str).tolist() == [
            "2026-01-01T00:00:01.000000000",
            "2026-01-01T00:00:02.000000000",
        ]

    def test_rejects_nat_and_bad_shapes(self) -> None:
        with pytest.raises(GeometryError, match="'times' must not contain NaT"):
            Axis.from_datetimes(np.array(["2026-01-01", "NaT"], dtype="datetime64[D]"))
        with pytest.raises(GeometryError, match="'times' must be 1-D and non-empty"):
            Axis.from_datetimes(np.array([], dtype="datetime64[D]"))

    def test_to_datetimes_needs_an_epoch_and_a_known_unit(self) -> None:
        with pytest.raises(GeometryError, match="'epoch' is None"):
            Axis(jnp.array([0.0, 1.0])).to_datetimes()
        axis = Axis(jnp.array([0.0, 1.0]), epoch="2026-01-01")
        with pytest.raises(GeometryError, match="'unit' must be one of"):
            axis.to_datetimes(unit="h")


class TestAffineGrid:
    def test_stores_float64(self) -> None:
        grid = AffineGrid(jnp.asarray(NORTH_UP, dtype=jnp.float32), (600, 600))
        assert grid.transform.dtype == jnp.float64
        assert grid.shape == (600, 600)

    def test_from_affine_reads_attributes(self) -> None:
        affine = types.SimpleNamespace(
            a=20.0, b=0.0, c=600000.0, d=0.0, e=-20.0, f=3550000.0
        )
        grid = AffineGrid.from_affine(affine, (2, 3))
        assert grid.transform.tolist() == NORTH_UP

    @pytest.mark.parametrize(
        "transform",
        [[*NORTH_UP, 0.0, 0.0, 1.0], [NORTH_UP[:3], NORTH_UP[3:]]],
    )
    def test_from_affine_reads_matrices(self, transform: object) -> None:
        assert AffineGrid.from_affine(transform, (2, 3)).transform.tolist() == NORTH_UP

    @pytest.mark.parametrize(
        ("transform", "match"),
        [
            ([*NORTH_UP, 1.0, 0.0, 1.0], r"last row \(0, 0, 1\)"),
            (NORTH_UP[:5], r"'transform' must have 6 numbers"),
            ("north up", "'transform' must describe an affine transform"),
        ],
    )
    def test_from_affine_rejects_non_transforms(
        self, transform: object, match: str
    ) -> None:
        with pytest.raises(GeometryError, match=match):
            AffineGrid.from_affine(transform, (2, 3))

    @pytest.mark.parametrize(
        ("transform", "match"),
        [
            (NORTH_UP[:5], r"'transform' must have shape \(6,\)"),
            ([1.0, 2.0, 0.0, 2.0, 4.0, 0.0], "'transform' must be invertible"),
            ([1.0, 0.0, jnp.nan, 0.0, 1.0, 0.0], "'transform' must be finite"),
        ],
    )
    def test_rejects_bad_transforms(self, transform: list[float], match: str) -> None:
        with pytest.raises(GeometryError, match=match):
            AffineGrid(transform, (2, 3))

    def test_needs_x64(self) -> None:
        with (
            jax.enable_x64(False),
            pytest.warns(UserWarning, match="truncated"),
            pytest.raises(GeometryError, match="enable jax_enable_x64"),
        ):
            AffineGrid(NORTH_UP, (2, 3))

    @pytest.mark.parametrize("shape", [(0, 3), (2,), (2, 3, 4), (True, 3), (2, -1)])
    def test_rejects_bad_shapes(self, shape: object) -> None:
        with pytest.raises(GeometryError, match="'shape' must be"):
            AffineGrid(NORTH_UP, shape)  # type: ignore[arg-type]

    def test_accepts_numpy_integer_shapes(self) -> None:
        grid = AffineGrid(NORTH_UP, (np.int64(4), np.int64(5)))
        assert grid.shape == (4, 5)
        assert all(type(n) is int for n in grid.shape)

    def test_checks_values_when_built_under_jit(self) -> None:
        build = jax.jit(lambda transform: AffineGrid(transform, (2, 2)).transform)
        with pytest.raises(Exception, match="'transform' must be invertible"):
            build(jnp.array([1.0, 2.0, 0.0, 2.0, 4.0, 0.0])).block_until_ready()


class TestRectilinearGrid:
    def test_shape_follows_the_axes(self) -> None:
        latitude = Axis(jnp.linspace(-90.0, 90.0, 73))
        longitude = Axis(jnp.linspace(0.0, 357.5, 144), period=360.0)
        grid = RectilinearGrid([latitude, longitude])
        assert isinstance(grid.axes, tuple)
        assert grid.shape == (73, 144)

    def test_rejects_empty_and_foreign_axes(self) -> None:
        with pytest.raises(GeometryError, match="'axes' must hold at least one Axis"):
            RectilinearGrid(())
        with pytest.raises(
            GeometryError, match=r"'axes\[1\]' must be an Axis, got Array"
        ):
            RectilinearGrid((Axis(jnp.array([0.0, 1.0])), jnp.array([0.0, 1.0])))


class TestCurvilinearGrid:
    def test_shape(self) -> None:
        x, y = jnp.meshgrid(jnp.arange(4.0), jnp.arange(3.0))
        assert CurvilinearGrid(x=x, y=y).shape == (3, 4)

    @pytest.mark.parametrize(
        ("x", "y", "match"),
        [
            (jnp.zeros(4), jnp.zeros(4), "'x' must be 2-D"),
            (jnp.zeros((1, 4)), jnp.zeros((1, 4)), r"'x' must be at least 2 × 2"),
            (
                jnp.zeros((2, 2), dtype=int),
                jnp.zeros((2, 2)),
                "'x' must have a floating dtype",
            ),
            (
                jnp.zeros((2, 3)),
                jnp.zeros((3, 2)),
                "'x' and 'y' must have the same shape",
            ),
        ],
    )
    def test_rejects_bad_coordinates(
        self, x: jax.Array, y: jax.Array, match: str
    ) -> None:
        with pytest.raises(GeometryError, match=match):
            CurvilinearGrid(x=x, y=y)


class TestPoints:
    def test_defaults(self) -> None:
        points = Points(jnp.zeros((4, 2)))
        assert points.z is None
        assert points.valid is None
        assert points.metric == "euclidean"

    def test_keeps_z_and_valid(self) -> None:
        points = Points(
            jnp.zeros((3, 2)),
            z=jnp.array([1.0, 2.0, 3.0]),
            valid=jnp.array([True, False, True]),
            metric="geodesic",
        )
        assert points.z.shape == (3,)
        assert points.valid.dtype == jnp.bool_

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"xy": jnp.zeros(3)}, "'xy' must be 2-D"),
            ({"xy": jnp.zeros((3, 2), dtype=int)}, "'xy' must have a floating dtype"),
            (
                {"xy": jnp.zeros((3, 2)), "z": jnp.zeros(2)},
                r"'z' must have shape \(3,\)",
            ),
            (
                {"xy": jnp.zeros((3, 2)), "z": jnp.zeros(3, dtype=int)},
                "'z' must have a floating dtype",
            ),
            (
                {"xy": jnp.zeros((3, 2)), "valid": jnp.ones(2, dtype=bool)},
                r"'valid' must have shape \(3,\)",
            ),
            (
                {"xy": jnp.zeros((3, 2)), "valid": jnp.ones(3)},
                "'valid' must be boolean",
            ),
            (
                {"xy": jnp.zeros((3, 2)), "metric": "manhattan"},
                "'metric' must be one of",
            ),
        ],
    )
    def test_rejects_bad_fields(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(GeometryError, match=match):
            Points(**kwargs)  # type: ignore[arg-type]


def _examples() -> list[eqx.Module]:
    longitude = Axis(jnp.linspace(0.0, 350.0, 36), period=360.0)
    x, y = jnp.meshgrid(jnp.arange(4.0), jnp.arange(3.0))
    return [
        Axis.from_datetimes(
            np.arange("2026-01-01", "2026-01-05", dtype="datetime64[D]")
        ),
        AffineGrid(NORTH_UP, (600, 600)),
        RectilinearGrid((Axis(jnp.linspace(-90.0, 90.0, 19)), longitude)),
        CurvilinearGrid(x=x, y=y),
        Points(jnp.ones((5, 2)), z=jnp.zeros(5), valid=jnp.ones(5, dtype=bool)),
    ]


@pytest.mark.parametrize("module", _examples(), ids=lambda m: type(m).__name__)
def test_passes_through_jit_unchanged(module: eqx.Module) -> None:
    _assert_same_tree(jax.jit(lambda m: m)(module), module)


@pytest.mark.parametrize("module", _examples(), ids=lambda m: type(m).__name__)
def test_passes_through_vmap_unchanged(module: eqx.Module) -> None:
    batch = _stack(module, module, module)
    assert_traces(lambda m: m, batch)
