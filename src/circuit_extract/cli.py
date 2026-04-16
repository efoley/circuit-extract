"""Command-line entry point for circuit-extract.

Commands:

    circuit-extract vlm <image>            # multi-step VLM extraction
    circuit-extract vlm <image> --oneshot   # single-prompt baseline
    circuit-extract eval [--max-items N]   # run VLM pipeline on dataset and report metrics

Output is JSON on stdout (or to ``--output``) and matches
:class:`circuit_extract.schema.Netlist`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from circuit_extract.providers import get_provider
from circuit_extract.vlm import VLMExtractionPipeline

app = typer.Typer(
    name="circuit-extract",
    help="Extract circuit netlists from schematic drawings.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("vlm")
def vlm_command(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Schematic image."),
    provider: str = typer.Option("gemini", "--provider", "-p", help="VLM provider name."),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Override the provider's default model."
    ),
    oneshot: bool = typer.Option(
        False, "--oneshot", help="Use a single-prompt baseline instead of the multi-step pipeline."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON here instead of stdout."
    ),
    show_warnings: bool = typer.Option(
        True, "--warnings/--no-warnings", help="Print consistency warnings to stderr."
    ),
) -> None:
    """Extract a netlist from IMAGE using a multimodal LLM."""
    load_dotenv()

    provider_kwargs: dict[str, object] = {}
    if model is not None:
        provider_kwargs["model"] = model
    vlm = get_provider(provider, **provider_kwargs)

    pipeline = VLMExtractionPipeline(provider=vlm, multi_step=not oneshot)
    netlist = pipeline.run(image)

    payload = netlist.model_dump_json(indent=2)
    if output is None:
        typer.echo(payload)
    else:
        output.write_text(payload)
        typer.echo(f"Wrote {output}", err=True)

    if show_warnings:
        for warning in netlist.validate_consistency():
            typer.echo(f"warning: {warning}", err=True)


@app.command("eval")
def eval_command(
    provider: str = typer.Option("gemini", "--provider", "-p", help="VLM provider name."),
    model: str | None = typer.Option(None, "--model", "-m", help="Override provider model."),
    max_items: int = typer.Option(
        50, "--max-items", "-n", help="Max images to evaluate (default 50)."
    ),
    oneshot: bool = typer.Option(False, "--oneshot", help="Use single-prompt baseline."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON results to this file."
    ),
) -> None:
    """Evaluate VLM extraction against ground-truth SPICE netlists."""
    from circuit_extract.datasets import SchematicDataset
    from circuit_extract.eval import EvalResult, evaluate

    load_dotenv()

    provider_kwargs: dict[str, object] = {}
    if model is not None:
        provider_kwargs["model"] = model
    vlm = get_provider(provider, **provider_kwargs)
    pipeline = VLMExtractionPipeline(provider=vlm, multi_step=not oneshot)

    typer.echo(f"Loading dataset (max_items={max_items})...", err=True)
    dataset = SchematicDataset(max_items=max_items).load()
    typer.echo(f"Loaded {len(dataset)} items.", err=True)

    results: list[EvalResult] = []
    for i, item in enumerate(dataset):
        typer.echo(f"[{i + 1}/{len(dataset)}] {item.stem}...", err=True)
        try:
            predicted = pipeline.run(item.image_path)
            result = evaluate(predicted, item.ground_truth, stem=item.stem)
            results.append(result)
            typer.echo(
                f"  components F1={result.components.f1:.2f}  "
                f"nets ARI={result.nets.adjusted_rand_index:.2f}",
                err=True,
            )
        except Exception as e:
            typer.echo(f"  ERROR: {e}", err=True)

    if not results:
        typer.echo("No results to report.", err=True)
        raise typer.Exit(1)

    # Aggregate
    avg_f1 = sum(r.components.f1 for r in results) / len(results)
    avg_prec = sum(r.components.precision for r in results) / len(results)
    avg_rec = sum(r.components.recall for r in results) / len(results)
    avg_ari = sum(r.nets.adjusted_rand_index for r in results) / len(results)

    summary = {
        "n_images": len(results),
        "components": {
            "avg_precision": round(avg_prec, 4),
            "avg_recall": round(avg_rec, 4),
            "avg_f1": round(avg_f1, 4),
        },
        "nets": {
            "avg_adjusted_rand_index": round(avg_ari, 4),
        },
        "per_image": [
            {
                "stem": r.stem,
                "components": {
                    "precision": round(r.components.precision, 4),
                    "recall": round(r.components.recall, 4),
                    "f1": round(r.components.f1, 4),
                    "matched": r.components.matched,
                    "predicted": r.components.predicted,
                    "ground_truth": r.components.ground_truth,
                },
                "nets": {
                    "ari": round(r.nets.adjusted_rand_index, 4),
                    "common_pins": r.nets.common_pins,
                    "predicted_pins": r.nets.predicted_pins,
                    "ground_truth_pins": r.nets.ground_truth_pins,
                },
            }
            for r in results
        ],
    }

    payload = json.dumps(summary, indent=2)
    if output is None:
        sys.stdout.write(payload + "\n")
    else:
        output.write_text(payload)
        typer.echo(f"Wrote {output}", err=True)

    typer.echo(
        f"\n=== Aggregate ({len(results)} images) ===\n"
        f"  Components: P={avg_prec:.3f}  R={avg_rec:.3f}  F1={avg_f1:.3f}\n"
        f"  Nets:       ARI={avg_ari:.3f}",
        err=True,
    )


@app.command("parse-spice")
def parse_spice_command(
    spice_file: Path = typer.Argument(..., exists=True, readable=True, help="SPICE .sp file."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON here instead of stdout."
    ),
) -> None:
    """Parse a SPICE netlist into our JSON schema (useful for debugging)."""
    from circuit_extract.datasets.spice_parser import parse_spice

    netlist = parse_spice(spice_file)
    payload = netlist.model_dump_json(indent=2)
    if output is None:
        typer.echo(payload)
    else:
        output.write_text(payload)
        typer.echo(f"Wrote {output}", err=True)


@app.command("pipeline")
def pipeline_command(
    image: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """(Approach 2) YOLO + OpenCV pipeline. Not implemented yet."""
    raise typer.Exit(
        typer.echo(
            "approach 2 (YOLO detect + wire tracing) is not implemented yet — "
            "use `circuit-extract vlm` for now.",
            err=True,
        )
        or 2
    )


if __name__ == "__main__":
    app()
