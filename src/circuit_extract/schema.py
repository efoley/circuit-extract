"""Pydantic schema for the extracted netlist.

This is the canonical in-memory representation produced by every extraction
approach (VLM zero-shot, YOLO + CV, etc.). Keep it deliberately small and
JSON-friendly so it round-trips through ``model_dump_json`` /
``model_validate_json`` and is usable as a structured-output schema for VLM
providers (Gemini, Claude, OpenAI, ...).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# A loose set of canonical component types. We keep this open ("other") because
# the VLM will encounter symbols we haven't enumerated, and we'd rather capture
# them than reject the extraction.
ComponentType = Literal[
    "resistor",
    "capacitor",
    "inductor",
    "diode",
    "led",
    "zener",
    "bjt_npn",
    "bjt_pnp",
    "nmos",
    "pmos",
    "jfet",
    "opamp",
    "ic",
    "switch",
    "voltage_source",
    "current_source",
    "ac_source",
    "battery",
    "ground",
    "vcc",
    "node_label",
    "transformer",
    "crystal",
    "fuse",
    "connector",
    "other",
]


class BBox(BaseModel):
    """Axis-aligned bounding box in *pixel* coordinates of the source image.

    Coordinates are normalised to integers; (x1, y1) is the top-left corner.
    """

    x1: int
    y1: int
    x2: int
    y2: int


class Component(BaseModel):
    """A single circuit element identified in the schematic."""

    id: str = Field(
        description="Reference designator, e.g. 'R1', 'C2', 'U3'. Unique within a netlist.",
    )
    type: ComponentType = Field(
        description="Canonical component category. Use 'other' if unknown.",
    )
    value: str | None = Field(
        default=None,
        description=(
            "Value or part number as printed on the schematic, e.g. '10k', '100nF', 'LM358'."
        ),
    )
    pins: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of pin identifiers for this component, e.g. ['1','2'] for a "
            "two-terminal part or ['B','C','E'] for a BJT. Pin ids are local to the component."
        ),
    )
    bbox: BBox | None = Field(
        default=None,
        description="Optional pixel bbox of the component symbol in the source image.",
    )
    notes: str | None = Field(
        default=None,
        description="Free-form notes captured by the extractor (orientation, polarity, etc.).",
    )


class PinRef(BaseModel):
    """A reference to a specific pin of a specific component."""

    component_id: str
    pin: str


class Net(BaseModel):
    """An electrical net: a set of pins that are wired together."""

    name: str = Field(
        description=(
            "Net name. Use schematic labels (VCC, GND, CLK) when present, else 'N1', 'N2', ..."
        ),
    )
    pins: list[PinRef] = Field(
        description=(
            "All pins that belong to this net. Must contain at least 2 entries for a valid wire."
        ),
    )


class Netlist(BaseModel):
    """Top-level netlist produced by an extraction pipeline."""

    components: list[Component]
    nets: list[Net]
    source_image: str | None = Field(
        default=None,
        description="Path or identifier of the source schematic image.",
    )
    extractor: str | None = Field(
        default=None,
        description=(
            "Name of the pipeline that produced this netlist (e.g. 'vlm:gemini-2.5-flash')."
        ),
    )

    def validate_consistency(self) -> list[str]:
        """Return a list of human-readable consistency warnings.

        We *don't* raise: extraction pipelines will routinely produce slightly
        inconsistent output, and downstream code should be free to repair it.
        """
        warnings: list[str] = []
        comp_by_id = {c.id: c for c in self.components}

        # Duplicate component ids
        if len(comp_by_id) != len(self.components):
            warnings.append("Duplicate component ids detected.")

        # Every pin reference must point to a real component + pin
        for net in self.nets:
            for ref in net.pins:
                comp = comp_by_id.get(ref.component_id)
                if comp is None:
                    warnings.append(
                        f"Net '{net.name}' references unknown component '{ref.component_id}'."
                    )
                    continue
                if comp.pins and ref.pin not in comp.pins:
                    warnings.append(
                        f"Net '{net.name}' references unknown pin "
                        f"'{ref.component_id}.{ref.pin}' (component pins: {comp.pins})."
                    )
            if len(net.pins) < 2:
                warnings.append(f"Net '{net.name}' has fewer than 2 pins.")

        return warnings
