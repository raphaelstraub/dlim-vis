"""CLI entry point for dlim-vis: DLIM in, picture out."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dlim import __version__
from dlim.evaluator import sample_model
from dlim.parser import parse_file
from dlim.plot import render


def _discover_dlim_files(path: Path) -> list[Path]:
    """Return a list of ``.dlim`` files at *path* (file or directory).

    Matches both ``*.dlim`` and ``*.dlim *`` patterns to handle files like
    ``default-linear.dlim 2``.
    """
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(
            f for f in path.iterdir()
            if f.is_file() and ".dlim" in f.name
        )
        if not files:
            print(f"No .dlim files found in {path}", file=sys.stderr)
            sys.exit(1)
        return files
    print(f"Path does not exist: {path}", file=sys.stderr)
    sys.exit(1)


def _output_path_for(
    source: Path, output_arg: str | None, fmt: str, batch: bool
) -> Path:
    """Determine the output path for a given source file."""
    # Create a safe filename: replace .dlim and spaces
    safe_name = source.name.replace(".dlim", "_dlim").replace(" ", "_")
    # Strip trailing _dlim if it's the only suffix
    if safe_name.endswith("_dlim"):
        safe_name = safe_name[: -len("_dlim")]

    if output_arg:
        out = Path(output_arg)
        if batch or out.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            return out / f"{safe_name}.{fmt}"
        return out
    # Default: ./output/<safe_name>.<fmt>
    default_dir = Path("output")
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir / f"{safe_name}.{fmt}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dlim-vis",
        description="DLIM in, picture out — lightweight DLIM visualizer.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help=".dlim file or directory containing .dlim files",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file (single mode) or directory (batch mode). Default: ./output/",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default=None,
        help="Output format (default: inferred from extension, or png)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise (default: 42)",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Disable noise functions",
    )
    parser.add_argument(
        "--annotations",
        action="store_true",
        help="Show container boundary lines and labels",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI (default: 200)",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args(argv)

    dlim_files = _discover_dlim_files(args.input)
    batch = len(dlim_files) > 1

    # Determine output format
    fmt = args.format
    if fmt is None and args.output and not Path(args.output).is_dir():
        ext = Path(args.output).suffix.lstrip(".")
        fmt = ext if ext in ("png", "pdf", "svg") else "png"
    elif fmt is None:
        fmt = "png"

    for source in dlim_files:
        model = parse_file(source)
        times, values = sample_model(
            model,
            step=args.step,
            seed=args.seed,
            noise=not args.no_noise,
        )

        out_path = _output_path_for(source, args.output, fmt, batch)
        render(
            model,
            times,
            values,
            out_path,
            dpi=args.dpi,
            annotations=args.annotations,
        )

        print(f"  ✓ {source.name} → {out_path}")

    if batch:
        print(f"\nRendered {len(dlim_files)} models.")
