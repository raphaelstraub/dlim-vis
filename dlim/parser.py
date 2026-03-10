"""Parse DLIM XML files into a clean dataclass tree."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


# ---------------------------------------------------------------------------
# Namespace helpers
# ---------------------------------------------------------------------------

_DLIM_NS = "http://descartes.tools/dlim/0.1"
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
_XSI_TYPE = f"{{{_XSI_NS}}}type"


def _float(elem: ET.Element, attr: str, default: float = 0.0) -> float:
    """Read an optional float attribute, returning *default* if absent."""
    val = elem.get(attr)
    return float(val) if val is not None else default


def _str(elem: ET.Element, attr: str, default: str = "") -> str:
    val = elem.get(attr)
    return val if val is not None else default


# ---------------------------------------------------------------------------
# Dataclasses — function types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Constant:
    constant: float = 0.0


@dataclass(frozen=True)
class LinearTrend:
    start: float = 0.0
    end: float = 0.0


@dataclass(frozen=True)
class SinTrend:
    start: float = 0.0
    end: float = 0.0


@dataclass(frozen=True)
class ExponentialTrend:
    start: float = 0.0
    end: float = 0.0


@dataclass(frozen=True)
class UniformNoise:
    min: float = 0.0
    max: float = 1.0


@dataclass(frozen=True)
class LinearIncreaseAndDecline:
    base: float = 0.0
    peak: float = 0.0
    peak_time: float = 0.0


@dataclass(frozen=True)
class ExponentialIncreaseAndDecline:
    base: float = 0.0
    peak: float = 0.0
    peak_time: float = 0.0


@dataclass(frozen=True)
class ExponentialIncreaseLogarithmicDecline:
    base: float = 0.0
    peak: float = 0.0
    peak_time: float = 0.0
    log_order: float = 1.0


# A Function is any of the leaf types or a nested Sequence.
Function = Union[
    Constant,
    LinearTrend,
    SinTrend,
    ExponentialTrend,
    UniformNoise,
    LinearIncreaseAndDecline,
    ExponentialIncreaseAndDecline,
    ExponentialIncreaseLogarithmicDecline,
    "Sequence",
]


# ---------------------------------------------------------------------------
# Dataclasses — structural elements
# ---------------------------------------------------------------------------


@dataclass
class Combinator:
    """An additive / subtractive / multiplicative overlay."""

    operator: str = "ADD"  # ADD | SUB | MULT
    function: Function | None = None


@dataclass
class Container:
    """A time-bounded segment holding one function."""

    name: str = ""
    duration: float = 0.0
    first_iteration_start: float = 0.0
    first_iteration_end: float = 0.0
    function: Function | None = None


@dataclass
class Sequence:
    """Root or nested sequence of time-dependent containers + combinators."""

    name: str = ""
    terminate_after_loops: int = 1
    first_iteration_start: float = 0.0
    first_iteration_end: float = 0.0
    loop_duration: float = 0.0
    final_duration: float = 0.0
    containers: list[Container] = field(default_factory=list)
    combinators: list[Combinator] = field(default_factory=list)


@dataclass
class DlimModel:
    """Top-level wrapper."""

    root_sequence: Sequence
    source_path: Path | None = None


# ---------------------------------------------------------------------------
# Parsing logic
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, str] = {
    "tools.descartes.dlim:Constant": "Constant",
    "tools.descartes.dlim:LinearTrend": "LinearTrend",
    "tools.descartes.dlim:SinTrend": "SinTrend",
    "tools.descartes.dlim:ExponentialTrend": "ExponentialTrend",
    "tools.descartes.dlim:UniformNoise": "UniformNoise",
    "tools.descartes.dlim:LinearIncreaseAndDecline": "LinearIncreaseAndDecline",
    "tools.descartes.dlim:ExponentialIncreaseAndDecline": "ExponentialIncreaseAndDecline",
    "tools.descartes.dlim:ExponentialIncreaseLogarithmicDecline": "ExponentialIncreaseLogarithmicDecline",
    "tools.descartes.dlim:Sequence": "Sequence",
}


def _parse_function(elem: ET.Element) -> Function | None:
    """Parse a ``<function>`` element into the appropriate dataclass."""
    xsi_type = elem.get(_XSI_TYPE, "")
    kind = _TYPE_MAP.get(xsi_type)

    if kind == "Constant":
        return Constant(constant=_float(elem, "constant"))
    if kind == "LinearTrend":
        return LinearTrend(
            start=_float(elem, "functionOutputAtStart"),
            end=_float(elem, "functionOutputAtEnd"),
        )
    if kind == "SinTrend":
        return SinTrend(
            start=_float(elem, "functionOutputAtStart"),
            end=_float(elem, "functionOutputAtEnd"),
        )
    if kind == "ExponentialTrend":
        return ExponentialTrend(
            start=_float(elem, "functionOutputAtStart"),
            end=_float(elem, "functionOutputAtEnd"),
        )
    if kind == "UniformNoise":
        return UniformNoise(min=_float(elem, "min"), max=_float(elem, "max", 1.0))
    if kind == "LinearIncreaseAndDecline":
        return LinearIncreaseAndDecline(
            base=_float(elem, "base"),
            peak=_float(elem, "peak"),
            peak_time=_float(elem, "peakTime"),
        )
    if kind == "ExponentialIncreaseAndDecline":
        return ExponentialIncreaseAndDecline(
            base=_float(elem, "base"),
            peak=_float(elem, "peak"),
            peak_time=_float(elem, "peakTime"),
        )
    if kind == "ExponentialIncreaseLogarithmicDecline":
        return ExponentialIncreaseLogarithmicDecline(
            base=_float(elem, "base"),
            peak=_float(elem, "peak"),
            peak_time=_float(elem, "peakTime"),
            log_order=_float(elem, "logarithmicOrder", 1.0),
        )
    if kind == "Sequence":
        return _parse_sequence(elem)

    # Unknown type — return None (will be treated as 0 output).
    return None


def _parse_container(elem: ET.Element) -> Container:
    func_elem = elem.find("function")
    return Container(
        name=_str(elem, "name"),
        duration=_float(elem, "duration"),
        first_iteration_start=_float(elem, "firstIterationStart"),
        first_iteration_end=_float(elem, "firstIterationEnd"),
        function=_parse_function(func_elem) if func_elem is not None else None,
    )


def _parse_combinator(elem: ET.Element) -> Combinator:
    func_elem = elem.find("function")
    return Combinator(
        operator=_str(elem, "operator", "ADD").upper(),
        function=_parse_function(func_elem) if func_elem is not None else None,
    )


def _parse_sequence(elem: ET.Element) -> Sequence:
    containers: list[Container] = []
    combinators: list[Combinator] = []

    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "sequenceFunctionContainers":
            containers.append(_parse_container(child))
        elif tag == "combine":
            combinators.append(_parse_combinator(child))

    return Sequence(
        name=_str(elem, "name"),
        terminate_after_loops=int(_float(elem, "terminateAfterLoops", 1)),
        first_iteration_start=_float(elem, "firstIterationStart"),
        first_iteration_end=_float(elem, "firstIterationEnd"),
        loop_duration=_float(elem, "loopDuration"),
        final_duration=_float(elem, "finalDuration"),
        containers=containers,
        combinators=combinators,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_file(path: str | Path) -> DlimModel:
    """Parse a ``.dlim`` file and return a :class:`DlimModel`."""
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()
    seq = _parse_sequence(root)
    return DlimModel(root_sequence=seq, source_path=path)
