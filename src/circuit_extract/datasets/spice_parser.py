"""Parse HSPICE ``.sp`` netlists into :class:`~circuit_extract.schema.Netlist`.

Handles two formats:

1. **Dataset format** (``hanky2397/schematic_images``): topology-only netlists
   wrapped in ``.subckt`` / ``.ends``. Each instance line ends with a bare
   model token (``r``, ``pmos4``, ``npn``, …) instead of a numeric value.
   Node count is determined by the model token.

2. **Standard HSPICE**: flat netlists with values (``10k``, ``100nF``),
   ``+`` continuation lines, comments, directives. Node count determined
   by the instance prefix letter.

The parser auto-detects which format it's reading and dispatches accordingly.
"""

from __future__ import annotations

import re
from pathlib import Path

from circuit_extract.schema import Component, ComponentType, Net, Netlist, PinRef

# -----------------------------------------------------------------------
# Dataset format: model token → (component type, node count, pin names)
# -----------------------------------------------------------------------

_MODEL_TOKEN_MAP: dict[str, tuple[ComponentType, int, list[str]]] = {
    "pmos4": ("pmos", 4, ["D", "G", "S", "B"]),
    "nmos4": ("nmos", 4, ["D", "G", "S", "B"]),
    "npn": ("bjt_npn", 3, ["C", "B", "E"]),
    "pnp": ("bjt_pnp", 3, ["C", "B", "E"]),
    "r": ("resistor", 2, ["1", "2"]),
    "c": ("capacitor", 2, ["1", "2"]),
    "l": ("inductor", 2, ["1", "2"]),
    "v": ("voltage_source", 2, ["+", "-"]),
    "i": ("current_source", 2, ["+", "-"]),
    "diode": ("diode", 2, ["+", "-"]),
    "inverter": ("ic", 2, ["in", "out"]),
    "op": ("opamp", 5, ["+", "-", "out", "vdd", "vss"]),
    "tgate": ("switch", 4, ["in", "out", "ctrl", "ctrlb"]),
    "dflipflop": ("ic", 4, ["D", "CLK", "Q", "Qb"]),
}

# Regex for parameterised logic gates: and2, nand3, or4, nor2, xor2, xnor2
_GATE_PATTERN = re.compile(r"^(x?n?(?:and|or|xor|xnor))(\d+)$", re.IGNORECASE)

# -----------------------------------------------------------------------
# Standard HSPICE: prefix → (component type, node count, has_value, pin names)
# -----------------------------------------------------------------------

_PREFIX_MAP: dict[str, tuple[ComponentType, int | None, bool, list[str]]] = {
    "R": ("resistor", 2, True, ["1", "2"]),
    "C": ("capacitor", 2, True, ["1", "2"]),
    "L": ("inductor", 2, True, ["1", "2"]),
    "D": ("diode", 2, False, ["+", "-"]),
    "Q": ("bjt_npn", 3, False, ["C", "B", "E"]),
    "M": ("nmos", 4, False, ["D", "G", "S", "B"]),
    "J": ("jfet", 3, False, ["D", "G", "S"]),
    "V": ("voltage_source", 2, True, ["+", "-"]),
    "I": ("current_source", 2, True, ["+", "-"]),
    "X": ("ic", None, False, []),
    "E": ("other", 4, True, ["out+", "out-", "in+", "in-"]),
    "F": ("other", 2, True, ["+", "-"]),
    "G": ("other", 4, True, ["out+", "out-", "in+", "in-"]),
    "H": ("other", 2, True, ["+", "-"]),
}


def _normalise_net(name: str) -> str:
    """Normalise a SPICE net name to a canonical form."""
    low = name.lower().strip()
    if low in ("0", "gnd", "gnd!"):
        return "GND"
    return name


def _join_continuation_lines(text: str) -> list[str]:
    """Join SPICE continuation lines (``+`` at start) with their predecessor."""
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("+") and lines:
            lines[-1] += " " + stripped[1:].strip()
        else:
            lines.append(stripped)
    return lines


def _resolve_model_token(token: str) -> tuple[ComponentType, int, list[str]] | None:
    """Try to resolve a model token from the dataset format.

    Returns (component_type, node_count, pin_names) or None.
    """
    low = token.lower()
    if low in _MODEL_TOKEN_MAP:
        return _MODEL_TOKEN_MAP[low]

    gate_match = _GATE_PATTERN.match(low)
    if gate_match:
        n_inputs = int(gate_match.group(2))
        pins = [f"in{i}" for i in range(n_inputs)] + ["out"]
        return ("ic", n_inputs + 1, pins)

    return None


def _is_dataset_format(lines: list[str]) -> bool:
    """Detect if this is the hanky2397 dataset format (subckt-wrapped, no values)."""
    return any(line.lower().startswith(".subckt") for line in lines)


def _parse_dataset_format(lines: list[str]) -> Netlist:
    """Parse the topology-only .subckt format from hanky2397/schematic_images."""
    components: list[Component] = []
    net_pins: dict[str, list[PinRef]] = {}

    def _add_to_net(net_name: str, comp_id: str, pin: str) -> None:
        net_name = _normalise_net(net_name)
        net_pins.setdefault(net_name, []).append(PinRef(component_id=comp_id, pin=pin))

    for line in lines:
        if line.startswith("*") or line.startswith("."):
            continue

        tokens = line.split()
        if len(tokens) < 3:
            continue

        inst_name = tokens[0]
        model_token = tokens[-1]
        node_tokens = tokens[1:-1]

        resolved = _resolve_model_token(model_token)
        if resolved is None:
            continue

        comp_type, expected_nodes, pin_names = resolved
        if len(node_tokens) != expected_nodes:
            pin_names = [str(i + 1) for i in range(len(node_tokens))]

        comp = Component(
            id=inst_name,
            type=comp_type,
            value=model_token if comp_type == "ic" else None,
            pins=list(pin_names),
        )
        components.append(comp)
        for pin_name, net_name in zip(pin_names, node_tokens, strict=False):
            _add_to_net(net_name, comp.id, pin_name)

    nets = [Net(name=name, pins=pins) for name, pins in sorted(net_pins.items())]
    return Netlist(components=components, nets=nets, extractor="ground_truth:spice")


def _extract_value(tokens: list[str]) -> str | None:
    """Pull a value string from remaining tokens after nodes."""
    if not tokens:
        return None
    cleaned = [t for t in tokens if not re.match(r"^[wl]=$", t, re.IGNORECASE)]
    val = " ".join(cleaned).strip()
    return val or None


def _parse_standard_format(lines: list[str]) -> Netlist:
    """Parse a standard HSPICE netlist with values and prefix-based node counts."""
    components: list[Component] = []
    net_pins: dict[str, list[PinRef]] = {}

    def _add_to_net(net_name: str, comp_id: str, pin: str) -> None:
        net_name = _normalise_net(net_name)
        net_pins.setdefault(net_name, []).append(PinRef(component_id=comp_id, pin=pin))

    for line in lines:
        if line.startswith("*") or line.startswith("."):
            continue

        tokens = line.split()
        if not tokens:
            continue

        inst_name = tokens[0]
        prefix = inst_name[0].upper()

        if prefix not in _PREFIX_MAP:
            continue

        comp_type, node_count, has_value, pin_names = _PREFIX_MAP[prefix]

        if prefix == "X":
            non_param = [t for t in tokens[1:] if "=" not in t]
            if len(non_param) < 2:
                continue
            subckt_name = non_param[-1]
            node_tokens = non_param[:-1]
            pin_names = [str(i + 1) for i in range(len(node_tokens))]
            comp = Component(id=inst_name, type="ic", value=subckt_name, pins=pin_names)
        else:
            assert node_count is not None
            if len(tokens) < 1 + node_count:
                continue
            node_tokens = tokens[1 : 1 + node_count]
            remaining = tokens[1 + node_count :]
            value = _extract_value(remaining)
            comp = Component(id=inst_name, type=comp_type, value=value, pins=list(pin_names))

        components.append(comp)
        for pin_name, net_name in zip(pin_names, node_tokens, strict=False):
            _add_to_net(net_name, comp.id, pin_name)

    nets = [Net(name=name, pins=pins) for name, pins in sorted(net_pins.items())]
    return Netlist(components=components, nets=nets, extractor="ground_truth:spice")


def parse_spice(path: str | Path) -> Netlist:
    """Parse a ``.sp`` file and return a :class:`Netlist`.

    Auto-detects whether the file uses the hanky2397 dataset format
    (``.subckt``-wrapped, topology-only) or standard HSPICE.
    """
    text = Path(path).read_text()
    lines = _join_continuation_lines(text)

    if _is_dataset_format(lines):
        return _parse_dataset_format(lines)
    return _parse_standard_format(lines)
