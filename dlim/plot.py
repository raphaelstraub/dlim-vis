"""Scientific matplotlib rendering for DLIM load intensity profiles."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dlim.parser import DlimModel


# ---------------------------------------------------------------------------
# Academic style constants
# ---------------------------------------------------------------------------

_CURVE_COLOR = "#1f77b4"     # matplotlib default blue
_FILL_ALPHA = 0.15
_ZONE_COLORS = ["#e8e8e8", "#f5f5f5"]  # alternating light greys for zone spans
_ZONE_ALPHA = 0.4
_ZONE_LINE_COLOR = "#aaaaaa"
_ZONE_TEXT_COLOR = "#555555"


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
        The parsed model (used for annotation data).
    times, values : np.ndarray
        Sampled time / arrival-rate arrays.
    output_path : str | Path
        Destination file. Extension determines format (png/pdf/svg).
    dpi : int
        Output resolution.
    annotations : bool
        If ``True``, draw shaded zone spans with labels at container boundaries.

    Returns
    -------
    Path
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(10, 4))

        # Axes limits (set early so annotations can reference y_max)
        ax.set_xlim(times[0], times[-1])
        y_max = max(values.max() * 1.1, 1.0)
        ax.set_ylim(0, y_max)

        # Container zone annotations (drawn first, behind everything)
        if annotations:
            containers = model.root_sequence.containers
            for i, container in enumerate(containers):
                t0 = container.first_iteration_start
                t1 = container.first_iteration_end

                # Alternating shaded spans
                ax.axvspan(
                    t0, t1,
                    color=_ZONE_COLORS[i % 2],
                    alpha=_ZONE_ALPHA,
                    zorder=0,
                )

                # Boundary line (skip the very first one at t=0)
                if t0 > 0:
                    ax.axvline(
                        x=t0,
                        color=_ZONE_LINE_COLOR,
                        linewidth=0.6,
                        linestyle="-",
                        alpha=0.6,
                        zorder=1,
                    )

                # Zone label at top center
                if container.name:
                    t_mid = (t0 + t1) / 2
                    ax.text(
                        t_mid,
                        y_max * 0.96,
                        container.name,
                        fontsize=7,
                        fontweight="bold",
                        color=_ZONE_TEXT_COLOR,
                        ha="center",
                        va="top",
                        zorder=5,
                    )

        # Main curve
        ax.plot(times, values, color=_CURVE_COLOR, linewidth=0.8, zorder=3)

        # Light fill under the curve
        ax.fill_between(times, values, alpha=_FILL_ALPHA, color=_CURVE_COLOR, zorder=2)

        # Axis labels — no units
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Arrival Rate", fontsize=10)

        # Raw numeric x-axis (no time formatting)
        ax.xaxis.set_major_locator(plt.AutoLocator())
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.ticklabel_format(axis="x", style="plain")

        ax.tick_params(labelsize=8)

        # No title

        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return output_path
