# dlim-vis

**DLIM in, picture out.**

A lightweight Python CLI that parses [Descartes Load Intensity Model](http://descartes.tools/limbo) (`.dlim`) files and renders publication-quality load intensity profiles — no Eclipse required.

---

## Why this exists

The original [LIMBO](https://github.com/joakimkistowski/LIMBO) tool is an Eclipse plugin. To look at a load curve, you need to:

1. Install Eclipse.
2. Install the correct EMF SDK version.
3. Import LIMBO as a set of Eclipse plugin projects.
4. Hope the target platform resolves.
5. Launch a nested Eclipse runtime.
6. Create a "DLIM Generator Project."
7. Open the graphical editor.
8. Squint at a tiny, non-exportable chart widget.

Life is short. `.dlim` files are just XML. The math is just `sin`, `exp`, and `log`. You shouldn't need a 2 GB IDE to plot a curve.

**dlim-vis** reads the XML, evaluates the exact same formulas LIMBO uses, and writes a clean image to disk. One command. No Java. No Eclipse. No OSGi. No target platform resolution. No "workspace." No "perspective." Just the plot.

---

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

```bash
# Render a single model
dlim-vis models/espresso.dlim -o espresso.png

# Render all models in a directory
dlim-vis models/ -o output/

# Disable noise for a clean deterministic curve
dlim-vis models/espresso.dlim -o espresso.png --no-noise

# Show container boundary annotations
dlim-vis models/espresso.dlim -o espresso.png --annotations
```

## Usage

```
dlim-vis <input> [options]

positional arguments:
  input                 .dlim file or directory of .dlim files

options:
  -o, --output PATH     Output file or directory (default: ./output/)
  --format FORMAT       png, pdf, or svg (default: inferred from extension, or png)
  --step SECONDS        Sampling interval in seconds (default: 1.0)
  --seed INT            Random seed for noise (default: 42)
  --no-noise            Disable noise functions
  --annotations         Show container boundary lines and labels
  --dpi INT             Output DPI (default: 200)
  -h, --help            Show help
```

## Supported DLIM function types

| Category | Type | Status |
|----------|------|--------|
| **Trend** | `Constant` | ✅ |
| | `LinearTrend` | ✅ |
| | `SinTrend` | ✅ |
| | `ExponentialTrend` | ✅ |
| **Burst** | `LinearIncreaseAndDecline` | ✅ |
| | `ExponentialIncreaseAndDecline` | ✅ |
| | `ExponentialIncreaseLogarithmicDecline` | ✅ |
| **Noise** | `UniformNoise` | ✅ |
| **Combinator** | Nested `Sequence` (ADD) | ✅ |

All formulas are ported 1-to-1 from LIMBO's [`FunctionValueCalculator.java`](https://github.com/joakimkistowski/LIMBO/blob/master/dlim.generator/src/tools/descartes/dlim/generator/util/FunctionValueCalculator.java).

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Lint
ruff check dlim/
```

## How it works

A `.dlim` file is XML describing a **Sequence** of time-bounded function segments, optionally layered with additive/multiplicative **Combinators**:

```
Sequence
├── sequenceFunctionContainers[0]  →  Constant(5.0)        [0s – 300s]
├── sequenceFunctionContainers[1]  →  SinTrend(5.0, 60.0)  [300s – 600s]
├── combine[0]                     →  UniformNoise(-1, 0)   [global overlay]
└── combine[1]                     →  nested Sequence        [burst spikes]
```

dlim-vis parses this tree, samples the model at regular time steps, and renders the result with matplotlib.

## License

[Apache 2.0](LICENSE)

## Acknowledgments

DLIM and LIMBO were created by [Joakim von Kistowski](https://github.com/joakimkistowski) and colleagues at the [Descartes Research Group](http://descartes.tools), University of Würzburg. This project is a standalone re-implementation of the evaluation engine; it does not share any code with LIMBO.
