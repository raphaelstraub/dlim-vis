"""Mathematical evaluation of individual DLIM function types.

All formulas are ported 1-to-1 from LIMBO's ``FunctionValueCalculator.java``.
See: https://github.com/joakimkistowski/LIMBO/blob/master/dlim.generator/
     src/tools/descartes/dlim/generator/util/FunctionValueCalculator.java
"""

from __future__ import annotations

import math
import random

from dlim.parser import (
    Constant,
    ExponentialIncreaseAndDecline,
    ExponentialIncreaseLogarithmicDecline,
    ExponentialTrend,
    LinearIncreaseAndDecline,
    LinearTrend,
    SinTrend,
    UniformNoise,
)


def eval_constant(f: Constant, x: float, duration: float) -> float:
    """``f(x) = constant``"""
    return f.constant


def eval_linear_trend(f: LinearTrend, x: float, duration: float) -> float:
    """``f(x) = S + x * (E - S) / D``"""
    if duration == 0:
        return f.start
    return f.start + x * (f.end - f.start) / duration


def eval_sin_trend(f: SinTrend, x: float, duration: float) -> float:
    """Half-sine S-curve interpolation between *start* and *end*.

    Rising (S ≤ E): ``phase = -π/2``
    Falling (S > E): ``phase = +π/2``

    ``f(x) = min + (max-min)/2 + (max-min)/2 * sin(phase + x*π/D)``
    """
    if duration == 0:
        return f.start
    min_val = min(f.start, f.end)
    max_val = max(f.start, f.end)
    phase = -math.pi / 2 if f.start <= f.end else math.pi / 2
    half_range = (max_val - min_val) / 2.0
    return min_val + half_range + half_range * math.sin(phase + x * math.pi / duration)


def _exponential_trend_core(x: float, start: float, end: float, duration: float) -> float:
    """Shared exponential interpolation in log-space.

    Handles the edge case where start or end ≤ 0 by applying an offset,
    exactly as LIMBO does.
    """
    if duration == 0:
        return start
    offset = 0.0
    s, e = start, end
    min_val = min(s, e)
    if min_val <= 0:
        offset = min_val - 0.5
        s = s + 0.5 - min_val
        e = e + 0.5 - min_val
    return offset + math.exp(math.log(s) + (math.log(e) - math.log(s)) * (x / duration))


def eval_exponential_trend(f: ExponentialTrend, x: float, duration: float) -> float:
    """Exponential interpolation between *start* and *end*."""
    return _exponential_trend_core(x, f.start, f.end, duration)


def _logarithmic_trend_core(
    x: float, start: float, end: float, order: float, duration: float
) -> float:
    """Logarithmic interpolation, matching LIMBO's ``calculateLogarithmicTrendValue``."""
    if duration == 0:
        return start
    if start > end:
        tmp_x = abs(x - duration)
        return end + (start - end) * (1.0 / order) * math.log(
            (tmp_x * (math.exp(order) - 1) / duration) + 1
        )
    return start + (end - start) * (1.0 / order) * math.log(
        (x * (math.exp(order) - 1) / duration) + 1
    )


def eval_uniform_noise(f: UniformNoise, x: float, duration: float, rng: random.Random) -> float:
    """``f(x) = min + (max - min) * random()``"""
    return f.min + (f.max - f.min) * rng.random()


# ---------------------------------------------------------------------------
# Burst functions
# ---------------------------------------------------------------------------


def _mirror_x(x: float, peak_time: float, duration: float) -> float:
    """Mirror x past peakTime back into the rising phase.

    ``tmpX = peakTime - (x - peakTime) * peakTime / (duration - peakTime)``
    """
    if duration == peak_time:
        return peak_time
    return peak_time - (x - peak_time) * peak_time / (duration - peak_time)


def eval_linear_increase_and_decline(
    f: LinearIncreaseAndDecline, x: float, duration: float
) -> float:
    """Triangular spike: linear rise then mirrored linear decline."""
    tmp_x = x
    if x > f.peak_time:
        tmp_x = _mirror_x(x, f.peak_time, duration)
    if f.peak_time == 0:
        return f.base
    return f.base + (f.peak - f.base) / f.peak_time * tmp_x


def eval_exponential_increase_and_decline(
    f: ExponentialIncreaseAndDecline, x: float, duration: float
) -> float:
    """Exponential rise then mirrored exponential decline."""
    tmp_x = x
    if x > f.peak_time:
        tmp_x = _mirror_x(x, f.peak_time, duration)
    return _exponential_trend_core(tmp_x, f.base, f.peak, f.peak_time)


def eval_exponential_increase_logarithmic_decline(
    f: ExponentialIncreaseLogarithmicDecline, x: float, duration: float
) -> float:
    """Exponential rise, logarithmic decay."""
    if x > f.peak_time:
        tmp_x = _mirror_x(x, f.peak_time, duration)
        return _logarithmic_trend_core(tmp_x, f.base, f.peak, f.log_order, f.peak_time)
    return _exponential_trend_core(x, f.base, f.peak, f.peak_time)
