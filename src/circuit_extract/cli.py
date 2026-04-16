"""Command-line entry point for circuit-extract.

Two commands today (more to come for approach 2):

    circuit-extract vlm <image>          # multi-step VLM extraction
    circuit-extract vlm <image> --oneshot  # single-prompt baseline

Output is JSON on stdout (or to ``--output``) and matches
:class:`circuit_extract.schema.Netlist`.
"""

from __future__ import annotations

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
