"""Recursive evaluation engine for DLIM models.

Mirrors the evaluation pipeline from LIMBO's ``ModelEvaluator.java``:
  Sequence → Containers (time-guarded, summed) → Combinators (MULT then ADD/SUB).
"""

from __future__ import annotations

import random

import numpy as np

from dlim.parser import (
    Combinator,
    Constant,
    Container,
    DlimModel,
    ExponentialIncreaseAndDecline,
    ExponentialIncreaseLogarithmicDecline,
    ExponentialTrend,
    Function,
    LinearIncreaseAndDecline,
    LinearTrend,
    Sequence,
    SinTrend,
    UniformNoise,
)
from dlim import functions as fn


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_function(func: Function | None, x: float, duration: float, rng: random.Random) -> float:
    """Evaluate a single function node at container-relative time *x*."""
    if func is None:
        return 0.0

    if isinstance(func, Constant):
        return fn.eval_constant(func, x, duration)
    if isinstance(func, LinearTrend):
        return fn.eval_linear_trend(func, x, duration)
    if isinstance(func, SinTrend):
        return fn.eval_sin_trend(func, x, duration)
    if isinstance(func, ExponentialTrend):
        return fn.eval_exponential_trend(func, x, duration)
    if isinstance(func, UniformNoise):
        if not _noise_enabled:
            return 0.0
        return fn.eval_uniform_noise(func, x, duration, rng)
    if isinstance(func, LinearIncreaseAndDecline):
        return fn.eval_linear_increase_and_decline(func, x, duration)
    if isinstance(func, ExponentialIncreaseAndDecline):
        return fn.eval_exponential_increase_and_decline(func, x, duration)
    if isinstance(func, ExponentialIncreaseLogarithmicDecline):
        return fn.eval_exponential_increase_logarithmic_decline(func, x, duration)
    if isinstance(func, Sequence):
        return evaluate_sequence(func, x, rng)

    return 0.0


def _get_function_node_value(
    func: Function | None,
    x: float,
    duration: float,
    rng: random.Random,
) -> float:
    """Evaluate a function and its child combinators (if it has them)."""
    if func is None:
        return 0.0

    value = evaluate_function(func, x, duration, rng)

    # Only Sequence has combinators at the function level.
    if isinstance(func, Sequence):
        value = _apply_combinators(value, func.combinators, x, rng)

    return value


def _apply_combinators(
    base_value: float,
    combinators: list[Combinator],
    x: float,
    rng: random.Random,
) -> float:
    """Apply combinators in LIMBO order: MULT first, then ADD/SUB."""
    value = base_value

    # MULT combinators first
    for c in combinators:
        if "MULT" in c.operator:
            comb_val = _get_function_node_value(c.function, x, 0.0, rng)
            value *= comb_val

    # ADD / SUB combinators second
    for c in combinators:
        if "ADD" in c.operator:
            comb_val = _get_function_node_value(c.function, x, 0.0, rng)
            value += comb_val
        elif "SUB" in c.operator:
            comb_val = _get_function_node_value(c.function, x, 0.0, rng)
            value -= comb_val

    return value


def evaluate_container(container: Container, guard_time: float, rng: random.Random) -> float:
    """Evaluate a single time-bounded container at *guard_time*.

    Returns 0.0 if the container is not active at *guard_time*.
    """
    # Time guard: container is active in [firstIterationStart, firstIterationEnd)
    if guard_time < container.first_iteration_start:
        return 0.0
    if guard_time >= container.first_iteration_end:
        return 0.0

    # Convert to container-relative time (CONTAINERCLOCK — the default)
    x = guard_time - container.first_iteration_start

    if container.function is None:
        return 0.0

    return _get_function_node_value(container.function, x, container.duration, rng)


def evaluate_sequence(seq: Sequence, t: float, rng: random.Random) -> float:
    """Evaluate a sequence at absolute time *t*.

    Handles time guards, loop wrapping, and combinator application.
    """
    # Time guard for the sequence itself
    if t < seq.first_iteration_start or t >= seq.first_iteration_end:
        return 0.0

    # Loop wrapping: convert to internal time within the first loop
    internal_t = t
    if seq.loop_duration > 0:
        while internal_t >= seq.first_iteration_start + seq.loop_duration:
            internal_t -= seq.loop_duration

    # Evaluate all active containers and sum their values
    seq_value = 0.0
    for container in seq.containers:
        seq_value += evaluate_container(container, internal_t, rng)

    # Apply combinators
    seq_value = _apply_combinators(seq_value, seq.combinators, t, rng)

    return seq_value


def evaluate_model(model: DlimModel, t: float, rng: random.Random) -> float:
    """Evaluate the full model at time *t*."""
    return evaluate_sequence(model.root_sequence, t, rng)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


# Module-level noise flag — set by sample_model.
_noise_enabled: bool = True


def sample_model(
    model: DlimModel,
    step: float = 1.0,
    seed: int = 42,
    noise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the model at regular intervals and return ``(times, values)``.

    Parameters
    ----------
    model : DlimModel
        The parsed DLIM model.
    step : float
        Sampling interval in seconds.
    seed : int
        Random seed for noise reproducibility.
    noise : bool
        If ``False``, noise functions return 0.
    """
    global _noise_enabled
    _noise_enabled = noise

    duration = model.root_sequence.final_duration
    times = np.arange(0, duration, step)
    values = np.empty_like(times)

    rng = random.Random(seed)

    for i, t in enumerate(times):
        values[i] = evaluate_model(model, t, rng)

    _noise_enabled = True  # restore default
    return times, values
