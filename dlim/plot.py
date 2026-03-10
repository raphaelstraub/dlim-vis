"""Premium matplotlib rendering for DLIM load intensity profiles."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no GUI needed

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from dlim.parser import DlimModel


# ---------------------------------------------------------------------------
# Color palette & style constants
# ---------------------------------------------------------------------------

_BG_COLOR = "#0f1117"
_PLOT_BG = "#161922"
_GRID_COLOR = "#2a2d3a"
_TEXT_COLOR = "#c8cdd5"
_ACCENT_TOP = "#6366f1"    # Indigo-500
_ACCENT_BOTTOM = "#312e81"  # Indigo-900
_ACCENT_LINE = "#818cf8"    # Indigo-400
_ANNOTATION_COLOR = "#4b5563"
_ANNOTATION_TEXT = "#9ca3af"

_FONT_FAMILY = "sans-serif"


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
    """Render a DLIM profile as a premium-styled chart.

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

    # Figure setup
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(_BG_COLOR)
    ax.set_facecolor(_PLOT_BG)

    # Gradient fill under the curve using a vertical gradient image
    # We stack the curve fill on top of a transparent→accent gradient.
    ax.fill_between(
        times, values, alpha=0.0,  # invisible — just to set data range
    )

    # Draw the gradient fill manually via imshow + clip path
    y_min = 0
    y_max = max(values.max() * 1.15, 1.0)

    # Create gradient image (vertical, bottom→top: dark→bright)
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack([gradient, gradient])

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("dlim", [_ACCENT_BOTTOM, _ACCENT_TOP])

    extent = [times[0], times[-1], y_min, y_max]
    im = ax.imshow(
        gradient,
        aspect="auto",
        cmap=cmap,
        extent=extent,
        origin="lower",
        alpha=0.6,
        zorder=1,
    )

    # Clip the gradient to the area under the curve
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    verts = list(zip(times, values))
    verts.append((times[-1], y_min))
    verts.append((times[0], y_min))
    verts.append((times[0], values[0]))
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 1)
    clip_path = MplPath(verts, codes)
    patch = PathPatch(clip_path, transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(patch)
    im.set_clip_path(patch)

    # Main curve line
    ax.plot(times, values, color=_ACCENT_LINE, linewidth=1.2, zorder=3)

    # Container boundary annotations
    if annotations:
        for container in model.root_sequence.containers:
            t_start = container.first_iteration_start
            if t_start > 0:
                ax.axvline(
                    x=t_start,
                    color=_ANNOTATION_COLOR,
                    linewidth=0.6,
                    linestyle="--",
                    alpha=0.5,
                    zorder=2,
                )
            # Label at the midpoint of the container
            t_mid = (container.first_iteration_start + container.first_iteration_end) / 2
            label_y = y_max * 0.97
            if container.name:
                ax.text(
                    t_mid,
                    label_y,
                    container.name,
                    fontsize=6,
                    color=_ANNOTATION_TEXT,
                    ha="center",
                    va="top",
                    fontfamily=_FONT_FAMILY,
                    alpha=0.6,
                    rotation=45,
                    zorder=4,
                )

    # Axes styling
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Time", fontsize=10, color=_TEXT_COLOR, fontfamily=_FONT_FAMILY, labelpad=8)
    ax.set_ylabel(
        "Arrival Rate",
        fontsize=10,
        color=_TEXT_COLOR,
        fontfamily=_FONT_FAMILY,
        labelpad=8,
    )

    # Format x-axis with human-readable time labels
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
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Grid
    ax.grid(True, color=_GRID_COLOR, linewidth=0.4, linestyle=":", alpha=0.5)

    # Tick styling
    ax.tick_params(colors=_TEXT_COLOR, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title
    model_name = model.root_sequence.name or (
        model.source_path.stem if model.source_path else "DLIM Model"
    )
    ax.set_title(
        model_name,
        fontsize=14,
        color=_TEXT_COLOR,
        fontfamily=_FONT_FAMILY,
        fontweight="bold",
        pad=16,
    )

    # Subtitle with duration info
    duration_str = _format_time(model.root_sequence.final_duration)
    n_containers = len(model.root_sequence.containers)
    subtitle = f"{duration_str} duration · {n_containers} segments"
    ax.text(
        0.5,
        1.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=8,
        color=_ANNOTATION_TEXT,
        ha="center",
        va="bottom",
        fontfamily=_FONT_FAMILY,
    )

    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return output_path
