"""Scientific matplotlib rendering for DLIM load intensity profiles."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from dlim.parser import DlimModel


# ---------------------------------------------------------------------------
# Academic style constants
# ---------------------------------------------------------------------------

_CURVE_COLOR = "#1f77b4"     # matplotlib default blue
_FILL_ALPHA = 0.15
_GRID_ALPHA = 0.3
_ANNOTATION_COLOR = "#888888"
_ANNOTATION_ALPHA = 0.5


def _format_time(seconds: float) -> str:
    """Human-readable time label."""
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m" if m else f"{h}h"
    if seconds >= 60:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m{s:02d}s" if s else f"{m}m"
    return f"{int(seconds)}s"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(
    model: DlimModel,
    times: np.ndarray,
    values: np.ndarray,
    output_path: str | Path,
    *,
    dpi: int = 200,
    annotations: bool = False,
) -> Path:
    """Render a DLIM profile as a clean scientific chart.

    Parameters
    ----------
    model : DlimModel
        The parsed model (used for title and annotation data).
    times, values : np.ndarray
        Sampled time / arrival-rate arrays.
    output_path : str | Path
        Destination file. Extension determines format (png/pdf/svg).
    dpi : int
        Output resolution.
    annotations : bool
        If ``True``, draw vertical lines at container boundaries.

    Returns
    -------
    Path
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a clean academic style
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(10, 4))

        # Main curve
        ax.plot(times, values, color=_CURVE_COLOR, linewidth=0.8, zorder=3)

        # Light fill under the curve
        ax.fill_between(times, values, alpha=_FILL_ALPHA, color=_CURVE_COLOR, zorder=2)

        # Container boundary annotations
        if annotations:
            for container in model.root_sequence.containers:
                t_start = container.first_iteration_start
                if t_start > 0:
                    ax.axvline(
                        x=t_start,
                        color=_ANNOTATION_COLOR,
                        linewidth=0.5,
                        linestyle="--",
                        alpha=_ANNOTATION_ALPHA,
                        zorder=1,
                    )
                # Label at the midpoint
                t_mid = (container.first_iteration_start + container.first_iteration_end) / 2
                if container.name:
                    ax.text(
                        t_mid,
                        ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0,
                        container.name,
                        fontsize=5.5,
                        color=_ANNOTATION_COLOR,
                        ha="center",
                        va="bottom",
                        rotation=45,
                        alpha=0.7,
                        zorder=4,
                    )

        # Axes
        ax.set_xlim(times[0], times[-1])
        y_max = max(values.max() * 1.1, 1.0)
        ax.set_ylim(0, y_max)

        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Arrival Rate (req/s)", fontsize=10)

        # X-axis: human-readable time labels
        duration = times[-1]
        if duration >= 7200:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1800))
        elif duration >= 3600:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(600))
        elif duration >= 600:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
        else:
            ax.xaxis.set_major_locator(ticker.AutoLocator())

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: _format_time(x)))

        ax.tick_params(labelsize=8)

        # Title
        model_name = model.root_sequence.name or (
            model.source_path.stem if model.source_path else "DLIM Model"
        )
        n_containers = len(model.root_sequence.containers)
        duration_str = _format_time(model.root_sequence.final_duration)
        ax.set_title(
            f"{model_name}  ({duration_str}, {n_containers} segments)",
            fontsize=11,
            pad=10,
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return output_path
