"""Round-trip a :class:`~circuit_extract.schema.Netlist` to and from JSON."""

from __future__ import annotations

from pathlib import Path

from circuit_extract.schema import Netlist


def dump_json(netlist: Netlist, path: str | Path, *, indent: int = 2) -> None:
    Path(path).write_text(netlist.model_dump_json(indent=indent))


def load_json(path: str | Path) -> Netlist:
    return Netlist.model_validate_json(Path(path).read_text())
