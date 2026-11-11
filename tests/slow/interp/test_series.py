"""Tests for the 1-D series schemes and `NonlinearPlan`. The slow tier: oracle sweeps,
dot-product tests across policies, gradient and trace checks, each a second or more
(mostly einx compiling new shapes).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.interpolate
from _harness import (
    assert_adjoint,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_transpose_matches,
)

from functax import GeometryError, PolicyError
from functax.geometry import Axis, Points
from functax.interp import (
    Akima,
    CubicSpline,
    Linear,
    NonlinearPlan,
    Pchip,
    Plan,
    Policies,
    SeriesInterpolant,
    Steffen,
)


def _axis_and_data(seed: int, n: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Random uneven increasing knots and random values."""
    rng = np.random.default_rng(seed)
    knots = np.cumsum(rng.uniform(0.2, 1.5, n))
    return knots, rng.normal(size=n)


def _steffen_reference(
    knots: np.ndarray, values: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """A direct, loop-based port of Steffen (1990), eqs. 11, 13, 26 and 27."""
    n = knots.size
    widths = np.diff(knots)
    secants = np.diff(values) / widths
    slopes = np.zeros(n)
    for k in range(1, n - 1):
        parabola = (secants[k - 1] * widths[k] + secants[k] * widths[k - 1]) / (
            widths[k - 1] + widths[k]
        )
        slopes[k] = (np.sign(secants[k - 1]) + np.sign(secants[k])) * min(
            abs(secants[k - 1]), abs(secants[k]), 0.5 * abs(parabola)
        )
    for end, (h0, h1, d0, d1) in (
        (0, (widths[0], widths[1], secants[0], secants[1])),
        (n - 1, (widths[-1], widths[-2], secants[-1], secants[-2])),
    ):
        parabola = d0 * (1 + h0 / (h0 + h1)) - d1 * h0 / (h0 + h1)
        if parabola * d0 <= 0:
            slopes[end] = 0.0
        elif abs(parabola) > 2 * abs(d0):
            slopes[end] = 2 * d0
        else:
            slopes[end] = parabola
    return scipy.interpolate.CubicHermiteSpline(knots, values, slopes)(targets)


def _scipy(scheme: object, knots: np.ndarray, values: np.ndarray):  # type: ignore[no-untyped-def]
    """The SciPy interpolator matching ``scheme`` (increasing knots)."""
    if isinstance(scheme, CubicSpline):
        return scipy.interpolate.CubicSpline(knots, values, bc_type=scheme.bc)
    if isinstance(scheme, Pchip):
        return scipy.interpolate.PchipInterpolator(knots, values)
    return scipy.interpolate.Akima1DInterpolator(knots, values, extrapolate=True)


ORACLE_SCHEMES = [
    CubicSpline("not-a-knot"),
    CubicSpline("natural"),
    CubicSpline("clamped"),
    CubicSpline("periodic"),
    Pchip(),
    Akima(),
]


ORACLE_IDS = ["not-a-knot", "natural", "clamped", "periodic", "pchip", "akima"]


class TestOracles:
    @pytest.mark.parametrize("scheme", ORACLE_SCHEMES, ids=ORACLE_IDS)
    @pytest.mark.parametrize(
        "descending", [False, True], ids=["ascending", "descending"]
    )
    def test_equals_scipy(self, scheme, descending: bool) -> None:  # type: ignore[no-untyped-def]
        knots, values = _axis_and_data(
            seed=len(scheme.__class__.__name__) + 7 * descending
        )
        if isinstance(scheme, CubicSpline) and scheme.bc == "periodic":
            values[-1] = values[0]
        rng = np.random.default_rng(1)
        targets = rng.uniform(knots[0] - 0.5, knots[-1] + 0.5, size=60)
        expected = _scipy(scheme, knots, values)(targets)
        source_knots, source_values = (
            (knots[::-1], values[::-1]) if descending else (knots, values)
        )
        plan = scheme.plan(
            Axis(jnp.asarray(source_knots)),
            Points(jnp.asarray(targets)[:, None]),
            policies=Policies(outside="extrapolate"),
        )
        assert_matches_oracle(
            plan(jnp.asarray(source_values)), expected, rtol=1e-12, atol=1e-12
        )

    def test_batched_data(self) -> None:
        knots, _ = _axis_and_data(seed=5)
        data = jax.random.normal(jax.random.key(6), (3, 4, 11))
        targets = Points(jnp.linspace(knots[0], knots[-1], 7)[:, None])
        for scheme in (CubicSpline(), Pchip()):
            values = scheme.plan(Axis(jnp.asarray(knots)), targets)(data)
            assert values.shape == (3, 4, 7)
            single = scheme.plan(Axis(jnp.asarray(knots)), targets)(data[1, 2])
            assert_matches_oracle(values[1, 2], single, rtol=1e-14)

    @pytest.mark.parametrize(
        "descending", [False, True], ids=["ascending", "descending"]
    )
    def test_steffen_equals_the_paper(self, descending: bool) -> None:
        knots, values = _axis_and_data(seed=3)
        targets = np.random.default_rng(4).uniform(knots[0], knots[-1], size=60)
        expected = _steffen_reference(knots, values, targets)
        source_knots, source_values = (
            (knots[::-1], values[::-1]) if descending else (knots, values)
        )
        plan = Steffen().plan(
            Axis(jnp.asarray(source_knots)), Points(jnp.asarray(targets)[:, None])
        )
        assert_matches_oracle(
            plan(jnp.asarray(source_values)), expected, rtol=1e-12, atol=1e-12
        )

    def test_two_knots_are_linear(self) -> None:
        for scheme in (CubicSpline("natural"), Pchip(), Akima(), Steffen()):
            plan = scheme.plan(Axis(jnp.array([0.0, 2.0])), Points(jnp.array([[0.5]])))
            assert_matches_oracle(
                plan(jnp.array([1.0, 3.0])), jnp.array([1.5]), rtol=1e-15
            )


class TestReproduction:
    def test_clamped_reproduces_cubics_with_flat_ends(self) -> None:
        # f = 2x³ − 3x² has f′(0) = f′(1) = 0
        knots = np.sort(
            np.concatenate(
                [[0.0, 1.0], np.random.default_rng(9).uniform(0.05, 0.95, 8)]
            )
        )
        values = 2.0 * knots**3 - 3.0 * knots**2
        targets = np.linspace(0.0, 1.0, 31)
        plan = CubicSpline("clamped").plan(
            Axis(jnp.asarray(knots)), Points(jnp.asarray(targets)[:, None])
        )
        assert_matches_oracle(
            plan(jnp.asarray(values)),
            2.0 * targets**3 - 3.0 * targets**2,
            rtol=1e-11,
            atol=1e-12,
        )

    def test_not_a_knot_reproduces_cubics(self) -> None:
        knots, _ = _axis_and_data(seed=7)
        values = 0.3 * knots**3 - knots**2 + 2.0 * knots - 4.0
        targets = np.random.default_rng(8).uniform(knots[0], knots[-1], size=40)
        plan = CubicSpline().plan(
            Axis(jnp.asarray(knots)), Points(jnp.asarray(targets)[:, None])
        )
        expected = 0.3 * targets**3 - targets**2 + 2.0 * targets - 4.0
        assert_matches_oracle(
            plan(jnp.asarray(values)), expected, rtol=1e-11, atol=1e-11
        )


class TestMonotone:
    @pytest.mark.parametrize("scheme", [Pchip(), Steffen()], ids=["pchip", "steffen"])
    def test_monotone_data_gives_monotone_curves_in_10000_cases(self, scheme) -> None:  # type: ignore[no-untyped-def]
        # 5 uneven axes × 2,000 random non-decreasing data sets, plateaus included
        rng = np.random.default_rng(10)
        for seed in range(5):
            knots, _ = _axis_and_data(seed=100 + seed, n=9)
            steps = rng.exponential(size=(2_000, 8)) * (rng.random((2_000, 8)) < 0.7)
            data = np.concatenate(
                [np.zeros((2_000, 1)), np.cumsum(steps, axis=1)], axis=1
            )
            targets = np.linspace(knots[0], knots[-1], 400)
            values = scheme.plan(
                Axis(jnp.asarray(knots)), Points(jnp.asarray(targets)[:, None])
            )(jnp.asarray(data))
            assert float(jnp.min(jnp.diff(values, axis=-1))) >= -1e-12

    def test_gradients_are_finite_at_flat_segments(self) -> None:
        knots = jnp.arange(6.0)
        flat = jnp.array([0.0, 1.0, 1.0, 1.0, 2.0, 2.0])
        targets = Points(jnp.linspace(0.0, 5.0, 23)[:, None])
        for scheme in (Pchip(), Akima(), Steffen()):
            plan = scheme.plan(Axis(knots), targets)
            gradient = jax.grad(lambda y, plan=plan: jnp.sum(plan(y)))(flat)
            assert bool(jnp.all(jnp.isfinite(gradient))), type(scheme).__name__


class TestLinearize:
    @pytest.mark.parametrize(
        "scheme", [Pchip(), Akima(), Steffen()], ids=["pchip", "akima", "steffen"]
    )
    @pytest.mark.parametrize(
        "descending", [False, True], ids=["ascending", "descending"]
    )
    def test_equals_the_dense_jacobian(self, scheme, descending: bool) -> None:  # type: ignore[no-untyped-def]
        knots, values = _axis_and_data(seed=11)
        if descending:
            knots, values = knots[::-1], values[::-1]
        targets = Points(
            jnp.asarray(
                np.random.default_rng(12).uniform(knots.min(), knots.max(), 25)
            )[:, None]
        )
        plan = scheme.plan(Axis(jnp.asarray(knots)), targets)
        assert isinstance(plan, NonlinearPlan)
        tangent = plan.linearize(jnp.asarray(values))
        dense = jax.jacfwd(lambda y: plan(y))(jnp.asarray(values))
        assert tangent.n_taps == sum(scheme.window) + 1
        assert_matches_oracle(tangent.to_dense(), dense, rtol=1e-12, atol=1e-14)


class TestCubicSplineIsLinear:
    @pytest.mark.parametrize("bc", ["not-a-knot", "natural", "clamped", "periodic"])
    @pytest.mark.parametrize(
        "descending", [False, True], ids=["ascending", "descending"]
    )
    def test_dot_product(self, bc: str, descending: bool) -> None:
        knots, _ = _axis_and_data(seed=13)
        if descending:
            knots = knots[::-1]
        targets = Points(
            jnp.asarray(
                np.random.default_rng(14).uniform(knots.min(), knots.max(), 17)
            )[:, None]
        )
        plan = CubicSpline(bc).plan(Axis(jnp.asarray(knots)), targets)
        assert type(plan) is Plan
        operator = plan.operator()
        assert operator.in_size() == 11
        assert_adjoint(
            lambda y: operator.mv(y), plan.adjoint, 11, 17, key=jax.random.key(15)
        )
        assert_transpose_matches(lambda y: operator.mv(y), plan.adjoint, jnp.zeros(11))


class TestFitInTheTrace:
    @pytest.mark.parametrize(
        "scheme",
        [CubicSpline(), Pchip(), Akima(), Steffen()],
        ids=["spline", "pchip", "akima", "steffen"],
    )
    def test_gradients_in_the_data_and_the_knots(self, scheme) -> None:  # type: ignore[no-untyped-def]
        knots, values = _axis_and_data(seed=18, n=8)
        # targets away from the knots, where the curve is smooth in both
        targets = jnp.asarray(0.5 * (knots[1:] + knots[:-1]))[:, None]

        def in_data(y: jax.Array) -> jax.Array:
            return scheme.fit(Axis(jnp.asarray(knots)), y)(targets)

        def in_knots(x: jax.Array) -> jax.Array:
            return scheme.fit(Axis(x), jnp.asarray(values))(targets)

        assert_grad_matches_fd(in_data, jnp.asarray(values))
        assert_grad_matches_fd(in_knots, jnp.asarray(knots))

    def test_equals_the_plan(self) -> None:
        knots, values = _axis_and_data(seed=16)
        targets = jnp.asarray(
            np.random.default_rng(17).uniform(knots[0], knots[-1], 9)
        )[:, None]
        for scheme in (CubicSpline(), Pchip(), Akima(), Steffen()):
            field = scheme.fit(Axis(jnp.asarray(knots)), jnp.asarray(values))
            assert isinstance(field, SeriesInterpolant)
            expected = scheme.plan(Axis(jnp.asarray(knots)), Points(targets))(
                jnp.asarray(values)
            )
            assert_matches_oracle(field(targets), expected, rtol=0.0)
            assert_matches_oracle(
                field.plan(Points(targets))(jnp.asarray(values)), expected, rtol=0.0
            )


class TestPoliciesAndValidation:
    knots = Axis(jnp.arange(6.0))

    def test_outside_policies(self) -> None:
        targets = Points(jnp.array([[-1.0], [2.5], [7.0]]))
        values = jnp.arange(6.0) ** 2
        nan = Pchip().plan(self.knots, targets)(values)
        assert bool(jnp.isnan(nan[0])) and bool(jnp.isnan(nan[2]))
        clamp = Pchip().plan(self.knots, targets, policies=Policies(outside="clamp"))(
            values
        )
        assert clamp[0] == 0.0 and clamp[2] == 25.0
        with pytest.raises(Exception, match="outside the domain"):
            Pchip().plan(self.knots, targets, policies=Policies(outside="raise"))

    def test_missing_policies(self) -> None:
        values = jnp.arange(6.0).at[2].set(jnp.nan)
        targets = Points(jnp.array([[4.5]]))
        filled = CubicSpline().plan(
            self.knots, targets, policies=Policies(missing="fill", fill_value=0.0)
        )(values)
        assert bool(jnp.isfinite(filled[0]))
        with pytest.raises(PolicyError, match="needs a local plan"):
            Pchip().plan(self.knots, targets, policies=Policies(missing="renormalise"))(
                values
            )


def _pieces(
    lengths: tuple[int, ...], seed: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Knots and values of pieces of the given lengths, end to end.

    Each piece starts where the last ends, so the joined knots repeat there.
    """
    rng = np.random.default_rng(seed)
    knots, values = [], []
    start = 0.0
    for length in lengths:
        steps = rng.uniform(0.3, 1.0, length - 1)
        piece = start + np.concatenate([[0.0], np.cumsum(steps)])
        knots.append(piece)
        values.append(rng.normal(size=length))
        start = piece[-1]
    return knots, values


def _piece_reference(scheme, knots: np.ndarray, values: np.ndarray, targets):  # type: ignore[no-untyped-def]
    """The scheme on one piece alone; SciPy's line and parabola for short not-a-knot."""
    if isinstance(scheme, CubicSpline) and scheme.bc == "not-a-knot":
        if knots.size == 2:
            return np.interp(targets, knots, values)
        if knots.size == 3:
            return scipy.interpolate.CubicSpline(knots, values)(targets)
    plan = scheme.plan(Axis(jnp.asarray(knots)), Points(jnp.asarray(targets)[:, None]))
    return np.asarray(plan(jnp.asarray(values)))


SPLIT_SCHEMES = [
    Pchip(),
    Steffen(),
    Akima(),
    CubicSpline(),
    CubicSpline(bc="natural"),
    CubicSpline(bc="clamped"),
]
SPLIT_IDS = ["pchip", "steffen", "akima", "not-a-knot", "natural", "clamped"]


class TestRepeatedKnots:
    """A repeat splits the axis, and each piece is read as an axis of its own (#117)."""

    @pytest.mark.parametrize("scheme", SPLIT_SCHEMES, ids=SPLIT_IDS)
    @pytest.mark.parametrize("descending", [False, True])
    def test_each_piece_equals_the_scheme_on_it_alone(
        self, scheme, descending: bool
    ) -> None:  # type: ignore[no-untyped-def]
        # pieces of 7, 1, 2, 3, 4 and 6 knots, so every short-piece rule runs
        knots, values = _pieces((7, 1, 2, 3, 4, 6), seed=100)
        nodes = np.concatenate(knots)
        data = np.concatenate(values)
        targets, expected = [], []
        for piece_knots, piece_values in zip(knots, values, strict=True):
            if piece_knots.size < 2:
                continue
            # right-open: the right end of a piece is read by the next piece
            inside = np.linspace(piece_knots[0], piece_knots[-1], 17)[:-1]
            targets.append(inside)
            expected.append(_piece_reference(scheme, piece_knots, piece_values, inside))
        targets = np.concatenate(targets)
        if descending:
            nodes, data = nodes[::-1], data[::-1]
        axis = Axis(jnp.asarray(nodes), repeats="allow")
        plan = scheme.plan(axis, Points(jnp.asarray(targets)[:, None]))
        assert_matches_oracle(
            plan(jnp.asarray(data)), np.concatenate(expected), rtol=1e-12, atol=1e-13
        )

    @pytest.mark.parametrize("scheme", SPLIT_SCHEMES, ids=SPLIT_IDS)
    def test_a_repeat_reads_its_last_knot(self, scheme) -> None:  # type: ignore[no-untyped-def]
        axis = Axis(jnp.array([0.0, 0.5, 1.0, 1.0, 1.7, 2.5, 3.0]), repeats="allow")
        data = jnp.array([0.0, 1.0, 2.0, 5.0, 6.0, 4.0, 7.0])
        plan = scheme.plan(axis, Points(jnp.array([[1.0], [3.0]])))
        assert_matches_oracle(plan(data), np.array([5.0, 7.0]), rtol=1e-14)

    @pytest.mark.parametrize("scheme", SPLIT_SCHEMES, ids=SPLIT_IDS)
    def test_without_repeats_it_equals_a_strict_axis(self, scheme) -> None:  # type: ignore[no-untyped-def]
        knots, data = _axis_and_data(seed=101)
        targets = Points(jnp.asarray(np.linspace(knots[0], knots[-1], 31))[:, None])
        strict = scheme.plan(Axis(jnp.asarray(knots)), targets)
        allowing = scheme.plan(Axis(jnp.asarray(knots), repeats="allow"), targets)
        assert_matches_oracle(
            allowing(jnp.asarray(data)), strict(jnp.asarray(data)), rtol=1e-12
        )

    def test_extrapolation_holds_a_repeated_end(self) -> None:
        axis = Axis(jnp.array([0.0, 0.0, 1.0, 2.0, 3.0, 3.0]), repeats="allow")
        data = jnp.array([9.0, 0.0, 1.0, 4.0, 9.0, -2.0])
        plan = Pchip().plan(
            axis,
            Points(jnp.array([[-1.0], [3.0], [4.0]])),
            policies=Policies(outside="extrapolate"),
        )
        assert plan(data).tolist() == [9.0, -2.0, -2.0]

    @pytest.mark.parametrize("scheme", SPLIT_SCHEMES, ids=SPLIT_IDS)
    def test_knots_traced_under_vmap_and_finite_gradients(self, scheme) -> None:  # type: ignore[no-untyped-def]
        knots, values = _pieces((5, 4), seed=102)
        nodes = jnp.asarray(np.concatenate(knots))
        data = jnp.asarray(np.concatenate(values))
        targets = jnp.linspace(nodes[0], nodes[-1], 13)[:, None]

        def read(knots: jax.Array, data: jax.Array) -> jax.Array:
            return scheme.fit(Axis(knots, repeats="allow"), data)(targets)

        eager = read(nodes, data)
        batched = jax.vmap(read)(jnp.stack([nodes, nodes]), jnp.stack([data, data]))
        assert_matches_oracle(batched[1], eager, rtol=1e-14)
        gradient = jax.grad(lambda k, y: read(k, y).sum(), argnums=(0, 1))
        knots_gradient, data_gradient = gradient(nodes, data)
        assert bool(jnp.all(jnp.isfinite(knots_gradient)))
        assert bool(jnp.all(jnp.isfinite(data_gradient)))

    def test_cubic_spline_is_still_linear(self) -> None:
        knots, _ = _pieces((5, 2, 4), seed=103)
        axis = Axis(jnp.asarray(np.concatenate(knots)), repeats="allow")
        targets = Points(jnp.asarray(np.linspace(0.0, 4.0, 21))[:, None])
        plan = CubicSpline().plan(axis, targets)
        operator = plan.operator()
        assert_adjoint(
            lambda y: operator.mv(y), plan.adjoint, 11, 21, key=jax.random.key(104)
        )

    def test_periodic_spline_names_the_repeat(self) -> None:
        axis = Axis(jnp.array([0.0, 1.0, 1.0, 2.0, 3.0]), repeats="allow")
        with pytest.raises(GeometryError, match=r"bc='periodic'.* x\[1\] = x\[2\]"):
            CubicSpline(bc="periodic").plan(axis, Points(jnp.array([[0.5]])))

    @pytest.mark.parametrize("scheme", [Linear(), Pchip()], ids=["linear", "pchip"])
    def test_a_float32_cdf_inverse_with_repeats_matches_float64(self, scheme) -> None:  # type: ignore[no-untyped-def]
        # gauss_flows's HistogramCDF: 1000 samples in 128 bins, counts + 1e-6,
        # whose float32 CDF repeats in the empty tail bins near 1
        samples = np.random.default_rng(105).normal(size=1000)
        edges = np.linspace(samples.min(), samples.max(), 129)
        counts = np.histogram(samples, edges)[0] + 1e-6
        cdf64 = np.concatenate([[0.0], np.cumsum(counts)])
        cdf64 = cdf64 / cdf64[-1]
        cdf32 = cdf64.astype(np.float32)
        assert np.any(np.diff(cdf32) == 0)
        # the inverse at the samples' own CDF values, as the round trip reads it
        levels = np.interp(samples, edges, cdf64).astype(np.float32)
        inverse32 = scheme.fit(
            Axis(jnp.asarray(cdf32), repeats="allow"), jnp.asarray(edges, jnp.float32)
        )(jnp.asarray(levels)[:, None])
        inverse64 = scheme.fit(Axis(jnp.asarray(cdf64)), jnp.asarray(edges))(
            jnp.asarray(levels.astype(np.float64))[:, None]
        )
        assert inverse32.dtype == jnp.float32
        if isinstance(scheme, Linear):
            assert_matches_oracle(inverse32, inverse64, rtol=0.0, atol=1e-5)
            return
        # PCHIP's slopes within two cells of a repeat are a piece's end slopes,
        # where float64 still sees cells 1e-6 wide; elsewhere they agree
        cell = np.searchsorted(cdf32, levels, side="right") - 1
        repeat = np.diff(cdf32) == 0
        near = np.array([repeat[max(k - 2, 0) : k + 3].any() for k in cell])
        assert_matches_oracle(inverse32, inverse64, rtol=0.0, atol=1e-5, where=~near)
        # and the round trip to the samples is no worse than float64's
        error32 = np.abs(np.asarray(inverse32) - samples).max()
        error64 = np.abs(np.asarray(inverse64) - samples).max()
        assert error32 <= error64 + 1e-5
