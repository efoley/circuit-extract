"""Dataset loading and ground-truth parsing.

Provides loaders for public circuit schematic datasets and parsers that
convert ground-truth annotations into :class:`~circuit_extract.schema.Netlist`.

Currently supported:

- ``hanky2397/schematic_images`` — small (~700), topology-only HSPICE
  netlists + YOLO component bboxes. Schematics are computer-rendered.
- ``bshada/open-schematics`` — ~84k realistic KiCad renderings from
  open-source hardware projects on GitHub, with structured-JSON ground
  truth derived from the ``.kicad_sch`` source.
"""

from circuit_extract.datasets.kicad_parser import (
    parse_kicad_components,
    parse_kicad_json,
)
from circuit_extract.datasets.loader import DatasetItem, SchematicDataset
from circuit_extract.datasets.open_schematics import (
    OpenSchematicsDataset,
    OpenSchematicsItem,
)

__all__ = [
    "DatasetItem",
    "OpenSchematicsDataset",
    "OpenSchematicsItem",
    "SchematicDataset",
    "parse_kicad_components",
    "parse_kicad_json",
]
