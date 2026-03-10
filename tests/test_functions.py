"""Unit tests for DLIM function evaluators and evaluation engine.

Each test verifies against hand-computed values from LIMBO's formulas.
"""

from __future__ import annotations

import math
import random

import pytest

from dlim.parser import (
    Combinator,
    Constant,
    Container,
    DlimModel,
    ExponentialIncreaseAndDecline,
    ExponentialTrend,
    LinearIncreaseAndDecline,
    LinearTrend,
    Sequence,
    SinTrend,
    UniformNoise,
)
from dlim import functions as fn
from dlim.evaluator import (
    evaluate_container,
    evaluate_model,
    evaluate_sequence,
    sample_model,
)


# ---------------------------------------------------------------------------
# Constant
# ---------------------------------------------------------------------------


class TestConstant:
    def test_constant_any_x(self):
        f = Constant(constant=5.0)
        assert fn.eval_constant(f, 0, 100) == 5.0
        assert fn.eval_constant(f, 50, 100) == 5.0
        assert fn.eval_constant(f, 99, 100) == 5.0


# ---------------------------------------------------------------------------
# LinearTrend
# ---------------------------------------------------------------------------


class TestLinearTrend:
    def test_at_start(self):
        f = LinearTrend(start=2.0, end=10.0)
        assert fn.eval_linear_trend(f, 0, 100) == 2.0

    def test_at_end(self):
        f = LinearTrend(start=2.0, end=10.0)
        assert fn.eval_linear_trend(f, 100, 100) == 10.0

    def test_midpoint(self):
        f = LinearTrend(start=2.0, end=10.0)
        assert fn.eval_linear_trend(f, 50, 100) == 6.0

    def test_declining(self):
        f = LinearTrend(start=10.0, end=2.0)
        assert fn.eval_linear_trend(f, 50, 100) == 6.0


# ---------------------------------------------------------------------------
# SinTrend
# ---------------------------------------------------------------------------


class TestSinTrend:
    def test_rising_start(self):
        f = SinTrend(start=0.0, end=10.0)
        assert fn.eval_sin_trend(f, 0, 100) == pytest.approx(0.0, abs=1e-10)

    def test_rising_end(self):
        f = SinTrend(start=0.0, end=10.0)
        assert fn.eval_sin_trend(f, 100, 100) == pytest.approx(10.0, abs=1e-10)

    def test_rising_midpoint(self):
        f = SinTrend(start=0.0, end=10.0)
        assert fn.eval_sin_trend(f, 50, 100) == pytest.approx(5.0, abs=1e-10)

    def test_falling_start(self):
        f = SinTrend(start=10.0, end=0.0)
        assert fn.eval_sin_trend(f, 0, 100) == pytest.approx(10.0, abs=1e-10)

    def test_falling_end(self):
        f = SinTrend(start=10.0, end=0.0)
        assert fn.eval_sin_trend(f, 100, 100) == pytest.approx(0.0, abs=1e-10)

    def test_falling_midpoint(self):
        f = SinTrend(start=10.0, end=0.0)
        assert fn.eval_sin_trend(f, 50, 100) == pytest.approx(5.0, abs=1e-10)


# ---------------------------------------------------------------------------
# ExponentialTrend
# ---------------------------------------------------------------------------


class TestExponentialTrend:
    def test_at_start(self):
        f = ExponentialTrend(start=1.0, end=100.0)
        assert fn.eval_exponential_trend(f, 0, 100) == pytest.approx(1.0, abs=1e-10)

    def test_at_end(self):
        f = ExponentialTrend(start=1.0, end=100.0)
        assert fn.eval_exponential_trend(f, 100, 100) == pytest.approx(100.0, abs=1e-10)

    def test_geometric_midpoint(self):
        # At x=50, D=100: exp(ln(1) + (ln(100)-ln(1))*0.5) = exp(ln(10)) = 10
        f = ExponentialTrend(start=1.0, end=100.0)
        assert fn.eval_exponential_trend(f, 50, 100) == pytest.approx(10.0, abs=1e-10)


# ---------------------------------------------------------------------------
# UniformNoise
# ---------------------------------------------------------------------------


class TestUniformNoise:
    def test_within_bounds(self):
        f = UniformNoise(min=0.0, max=2.0)
        rng = random.Random(42)
        for _ in range(100):
            val = fn.eval_uniform_noise(f, 0, 100, rng)
            assert 0.0 <= val <= 2.0


# ---------------------------------------------------------------------------
# LinearIncreaseAndDecline
# ---------------------------------------------------------------------------


class TestLinearIncreaseAndDecline:
    def test_at_base(self):
        f = LinearIncreaseAndDecline(base=0.0, peak=10.0, peak_time=25.0)
        assert fn.eval_linear_increase_and_decline(f, 0, 100) == 0.0

    def test_at_peak(self):
        f = LinearIncreaseAndDecline(base=0.0, peak=10.0, peak_time=25.0)
        assert fn.eval_linear_increase_and_decline(f, 25, 100) == 10.0

    def test_after_peak(self):
        # At x=100 (end), the mirrored x should bring value back to base
        f = LinearIncreaseAndDecline(base=0.0, peak=10.0, peak_time=25.0)
        assert fn.eval_linear_increase_and_decline(f, 100, 100) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# ExponentialIncreaseAndDecline
# ---------------------------------------------------------------------------


class TestExponentialIncreaseAndDecline:
    def test_at_peak(self):
        f = ExponentialIncreaseAndDecline(base=1.0, peak=10.0, peak_time=50.0)
        val = fn.eval_exponential_increase_and_decline(f, 50, 100)
        assert val == pytest.approx(10.0, abs=1e-10)

    def test_at_start(self):
        f = ExponentialIncreaseAndDecline(base=1.0, peak=10.0, peak_time=50.0)
        val = fn.eval_exponential_increase_and_decline(f, 0, 100)
        assert val == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Container & Sequence evaluation
# ---------------------------------------------------------------------------


class TestContainerEvaluation:
    def test_active_container(self):
        c = Container(
            name="test",
            duration=100.0,
            first_iteration_start=0.0,
            first_iteration_end=100.0,
            function=Constant(constant=5.0),
        )
        rng = random.Random(42)
        assert evaluate_container(c, 50, rng) == 5.0

    def test_inactive_before(self):
        c = Container(
            name="test",
            duration=100.0,
            first_iteration_start=100.0,
            first_iteration_end=200.0,
            function=Constant(constant=5.0),
        )
        rng = random.Random(42)
        assert evaluate_container(c, 50, rng) == 0.0

    def test_inactive_after(self):
        c = Container(
            name="test",
            duration=100.0,
            first_iteration_start=0.0,
            first_iteration_end=100.0,
            function=Constant(constant=5.0),
        )
        rng = random.Random(42)
        assert evaluate_container(c, 100, rng) == 0.0


class TestSequenceEvaluation:
    def test_two_containers(self):
        seq = Sequence(
            name="test",
            first_iteration_end=200.0,
            loop_duration=200.0,
            final_duration=200.0,
            containers=[
                Container("a", 100, 0.0, 100.0, Constant(3.0)),
                Container("b", 100, 100.0, 200.0, Constant(7.0)),
            ],
        )
        rng = random.Random(42)
        assert evaluate_sequence(seq, 50, rng) == 3.0
        assert evaluate_sequence(seq, 150, rng) == 7.0

    def test_additive_combinator(self):
        seq = Sequence(
            name="test",
            first_iteration_end=100.0,
            loop_duration=100.0,
            final_duration=100.0,
            containers=[
                Container("base", 100, 0.0, 100.0, Constant(5.0)),
            ],
            combinators=[
                Combinator("ADD", Constant(constant=2.0)),
            ],
        )
        rng = random.Random(42)
        # Base(5) + Combinator(2) = 7
        assert evaluate_sequence(seq, 50, rng) == 7.0


class TestSampleModel:
    def test_deterministic_no_noise(self):
        model = DlimModel(
            root_sequence=Sequence(
                name="test",
                first_iteration_end=10.0,
                loop_duration=10.0,
                final_duration=10.0,
                containers=[
                    Container("const", 10, 0.0, 10.0, Constant(5.0)),
                ],
            )
        )
        t, v = sample_model(model, step=1.0, noise=False)
        assert len(t) == 10
        assert all(val == 5.0 for val in v)

    def test_noise_suppression(self):
        model = DlimModel(
            root_sequence=Sequence(
                name="test",
                first_iteration_end=10.0,
                loop_duration=10.0,
                final_duration=10.0,
                containers=[
                    Container("const", 10, 0.0, 10.0, Constant(5.0)),
                ],
                combinators=[
                    Combinator("ADD", UniformNoise(min=0.0, max=10.0)),
                ],
            )
        )
        _, v_no_noise = sample_model(model, step=1.0, noise=False)
        assert all(val == 5.0 for val in v_no_noise)

        _, v_with_noise = sample_model(model, step=1.0, noise=True)
        # With noise, at least one value should differ from 5.0
        assert any(val != 5.0 for val in v_with_noise)
