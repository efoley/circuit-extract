"""Dataset loading and ground-truth parsing.

Provides loaders for public circuit schematic datasets (starting with
``hanky2397/schematic_images`` on HuggingFace) and parsers that convert
ground-truth annotations into :class:`~circuit_extract.schema.Netlist`.
"""

from circuit_extract.datasets.loader import DatasetItem, SchematicDataset

__all__ = ["DatasetItem", "SchematicDataset"]
