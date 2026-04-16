"""Prompt templates for the multi-step VLM extraction pipeline.

These are deliberately kept as plain strings (not f-strings or Jinja templates)
so they're easy to edit and diff. Iterating on these prompts is most of the
work for approach 1.
"""

SYSTEM_PROMPT = """\
You are an expert electronics engineer reading hand-drawn or printed circuit
schematics. You extract structured data exactly as instructed. You never invent
components or connections that are not visually present. When uncertain, you
say so by lowering specificity (e.g. type='other', value=null) rather than
guessing.
"""


# ---------------------------------------------------------------------------
# Step 1: component identification
# ---------------------------------------------------------------------------

COMPONENTS_PROMPT = """\
Examine the attached schematic image and identify every circuit component.

For each component, return:
- id: a unique reference designator. Use the printed designator (R1, C2, U3,
  Q1, ...) when visible. If none is printed, assign one of the form
  R1, R2, ... C1, C2, ... using the standard prefix for that component type.
- type: one of the canonical categories in the schema. Use 'other' if unsure.
- value: the printed value or part number (e.g. '10k', '100nF', 'LM358').
  Use null if no value is shown.
- pins: an ordered list of pin identifiers local to that component, e.g.
  ['1','2'] for a two-terminal part, ['B','C','E'] for a BJT,
  ['1','2','3','4','5','6','7','8'] for an 8-pin IC. Use the labels printed
  on the schematic when present.
- bbox: optional pixel bbox of the component symbol if you can localise it
  confidently. Omit otherwise.
- notes: anything noteworthy about orientation or polarity.

Do NOT enumerate wires, nets, junctions, or labels in this step. Power and
ground SYMBOLS (VCC, +5V, GND) ARE components — include them with type
'vcc' or 'ground' and a single pin '1'.

Return ONLY the component list as JSON matching the provided schema.
"""


# ---------------------------------------------------------------------------
# Step 2: net tracing
# ---------------------------------------------------------------------------

NETS_PROMPT = """\
You previously identified the following components in this schematic:

{components_json}

Now trace every electrical net (wire-connected group of pins) in the same
image.

Rules:
- A net is a maximal set of pins that are connected by wires (and junction
  dots). Every pin of every component must belong to exactly one net.
- Wires that CROSS without a junction dot are NOT connected.
- All ground symbols share a single net named 'GND'. All VCC/+5V symbols
  with the same label share a single net named after that label
  (e.g. 'VCC', '+5V', '+12V').
- For other nets, use the schematic's printed label if any (CLK, RESET,
  OUT, ...). Otherwise name them 'N1', 'N2', ... in the order you encounter
  them.
- Every net must reference at least two pins. If a pin appears to be
  unconnected in the drawing, place it on its own single-pin net named
  'NC_<component>_<pin>' so we can flag it later.
- Do not invent components: every component_id MUST appear in the list above.

Return ONLY the netlist (components + nets) as JSON matching the provided
schema. Re-emit the components exactly as given.
"""


# ---------------------------------------------------------------------------
# Optional single-shot baseline (for comparison against multi-step)
# ---------------------------------------------------------------------------

ONESHOT_PROMPT = """\
Extract the complete netlist from the attached schematic image.

Identify every component (with id, type, value, pins) and every net (with
name and the pins it connects). Follow the rules:
- Use printed reference designators when visible; otherwise assign R1, C1, ...
- Treat ground and VCC symbols as components.
- Wires crossing without a junction dot are NOT connected.
- Every component pin must belong to exactly one net.

Return ONLY JSON matching the provided schema.
"""
