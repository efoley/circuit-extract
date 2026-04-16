# circuit-extract

Experiments in extracting circuit netlists from schematic drawings.

We're exploring two approaches and comparing them on the same inputs:

1. **Zero-shot VLM** — prompt a multimodal LLM (starting with Gemini 2.5
   Flash) in multiple stages: identify components → enumerate pins → trace
   nets. Provider-agnostic so we can swap in Claude / OpenAI / local models.
2. **YOLO + classic CV** *(planned)* — detect components with a YOLO model,
   crop each one and parse it with a VLM, then trace wires with OpenCV
   skeletonisation + junction analysis to recover topology.

Both pipelines emit the same JSON netlist format
(`circuit_extract.schema.Netlist`).

## Setup

```bash
uv sync
cp .env.example .env       # then add GEMINI_API_KEY
```

## Checks

```bash
uv run ruff check src tests       # lint
uv run ruff format --check src tests  # format check (use `ruff format` to fix)
uv run ty check                   # type-check (Astral's ty)
uv run pytest -q                  # tests
```

## Usage

```bash
# Multi-step VLM pipeline (default)
uv run circuit-extract vlm path/to/schematic.png

# Single-prompt baseline for comparison
uv run circuit-extract vlm path/to/schematic.png --oneshot

# Use a different Gemini model
uv run circuit-extract vlm schematic.png --model gemini-2.5-pro

# Write to a file
uv run circuit-extract vlm schematic.png -o out.netlist.json
```

## Layout

```
src/circuit_extract/
├── schema.py         # Pydantic Netlist / Component / Net (canonical format)
├── providers/        # VLM provider abstraction
│   ├── base.py       #   VLMProvider interface
│   └── gemini.py     #   Gemini implementation (google-genai SDK)
├── vlm/              # Approach 1: zero-shot VLM
│   ├── prompts.py    #   Prompt templates (edit me!)
│   └── pipeline.py   #   Multi-step extraction pipeline
├── detect/           # Approach 2 (planned): YOLO component detection
├── wires/            # Approach 2 (planned): OpenCV wire tracing
├── netlist/          # JSON I/O for the netlist schema
└── cli.py            # `circuit-extract` typer CLI
```

## Status

- [x] Project scaffold, schema, Gemini provider, multi-step VLM pipeline, CLI
- [ ] Sample dataset wired up (HuggingFace)
- [ ] Evaluation harness (precision/recall on components & nets)
- [ ] Approach 2: YOLO detection
- [ ] Approach 2: OpenCV wire tracing
- [ ] Additional providers (Claude, OpenAI, local)
- [ ] SPICE / KiCad netlist exporters
