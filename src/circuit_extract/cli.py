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
import logging
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from circuit_extract.providers import get_provider
from circuit_extract.vlm import VLMExtractionPipeline


def _configure_logging(verbose: bool) -> None:
    """Route circuit_extract.* loggers to stderr.

    We only touch our own namespace so we don't affect root handlers added by
    dependencies (google-genai's http client in particular can be very chatty).
    """
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger("circuit_extract")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False


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
    dataset: str = typer.Option(
        "open-schematics",
        "--dataset",
        help=(
            "Which dataset to evaluate against: 'open-schematics' (bshada/open-schematics, "
            "realistic KiCad renderings — default) or 'hanky2397' (hanky2397/schematic_images, "
            "synthetic + HSPICE)."
        ),
    ),
    extractor: str = typer.Option(
        "vlm",
        "--extractor",
        help=(
            "Which extraction pipeline to use: 'vlm' (default — multi-step VLM) or "
            "'detect' (zero-shot Grounding DINO detection; requires [detect] extra)."
        ),
    ),
    provider: str = typer.Option("gemini", "--provider", "-p", help="VLM provider name."),
    model: str | None = typer.Option(None, "--model", "-m", help="Override provider model."),
    max_items: int = typer.Option(
        50, "--max-items", "-n", help="Max images to evaluate (default 50)."
    ),
    oneshot: bool = typer.Option(False, "--oneshot", help="Use single-prompt baseline."),
    detect_model: str = typer.Option(
        "IDEA-Research/grounding-dino-tiny",
        "--detect-model",
        help="HuggingFace model id for the --extractor=detect path.",
    ),
    box_threshold: float = typer.Option(
        0.3,
        "--box-threshold",
        help="Minimum objectness score for a detection (detect only).",
    ),
    text_threshold: float = typer.Option(
        0.25,
        "--text-threshold",
        help="Minimum text-alignment score for a detection (detect only).",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON results to this file."
    ),
    data_dir: Path | None = typer.Option(
        None,
        "--data-dir",
        "-d",
        help=(
            "Pre-downloaded dataset directory (hanky2397 only: needs 'images/' and 'sp/'). "
            "Skips HuggingFace download."
        ),
    ),
    min_components: int = typer.Option(
        6,
        "--min-components",
        help="Filter: minimum real components per schematic (open-schematics only).",
    ),
    max_components: int = typer.Option(
        20,
        "--max-components",
        help="Filter: maximum real components per schematic (open-schematics only).",
    ),
    shards: str = typer.Option(
        "0",
        "--shards",
        help=(
            "Comma-separated shard indices to read (open-schematics only; 0..77). "
            "Default: shard 0 (~1,000 schematics, ~200 MB)."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """Evaluate component / net extraction against ground-truth netlists."""
    from circuit_extract.datasets import OpenSchematicsDataset, SchematicDataset
    from circuit_extract.eval import EvalResult, evaluate

    load_dotenv()
    _configure_logging(verbose)
    log = logging.getLogger("circuit_extract.eval")

    if extractor == "vlm":
        provider_kwargs: dict[str, object] = {}
        if model is not None:
            provider_kwargs["model"] = model
        vlm = get_provider(provider, **provider_kwargs)
        runnable: object = VLMExtractionPipeline(provider=vlm, multi_step=not oneshot)
        log.info(
            "extractor=vlm provider=%s model=%s multi_step=%s",
            vlm.name,
            vlm.model,
            not oneshot,
        )
    elif extractor == "detect":
        from circuit_extract.detect import DetectionPipeline, GroundingDinoDetector

        detector = GroundingDinoDetector(model=detect_model)
        runnable = DetectionPipeline(
            detector=detector,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        log.info("extractor=detect model=%s", detect_model)
    else:
        raise typer.BadParameter(f"unknown --extractor {extractor!r}; choose 'vlm' or 'detect'")

    log.info("loading dataset=%s (max_items=%d)...", dataset, max_items)
    loaded: object
    if dataset == "open-schematics":
        try:
            shard_indices = tuple(int(s) for s in shards.split(",") if s.strip())
        except ValueError as e:
            raise typer.BadParameter(f"invalid --shards value: {shards!r}") from e
        loaded = OpenSchematicsDataset(
            max_items=max_items,
            min_components=min_components,
            max_components=max_components,
            shard_indices=shard_indices,
        ).load()
    elif dataset == "hanky2397":
        loaded = SchematicDataset(max_items=max_items, data_dir=data_dir).load()
    else:
        raise typer.BadParameter(
            f"unknown --dataset {dataset!r}; choose 'open-schematics' or 'hanky2397'"
        )
    log.info("loaded %d items", len(loaded))  # type: ignore[arg-type]

    results: list[EvalResult] = []
    total = len(loaded)  # type: ignore[arg-type]
    for i, item in enumerate(loaded):  # type: ignore[arg-type]
        log.info("[%d/%d] %s", i + 1, total, item.stem)
        try:
            predicted = runnable.run(item.image_path)  # type: ignore[attr-defined]
            result = evaluate(predicted, item.ground_truth, stem=item.stem)
            results.append(result)
            log.info(
                "  components: pred=%d gt=%d matched=%d F1=%.2f | nets ARI=%.2f",
                result.components.predicted,
                result.components.ground_truth,
                result.components.matched,
                result.components.f1,
                result.nets.adjusted_rand_index,
            )
        except Exception as e:
            log.error("  ERROR: %s", e)

    if not results:
        log.error("No results to report.")
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


@app.command("detect")
def detect_command(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Schematic image."),
    model: str = typer.Option(
        "IDEA-Research/grounding-dino-tiny",
        "--model",
        "-m",
        help="HuggingFace model id (tiny is ~170 MB; -base is larger and more accurate).",
    ),
    classes: str | None = typer.Option(
        None,
        "--classes",
        help=(
            "Comma-separated class prompt. Defaults to the built-in schematic vocabulary "
            "(resistor, capacitor, diode, ...)."
        ),
    ),
    box_threshold: float = typer.Option(0.3, "--box-threshold", help="Minimum objectness score."),
    text_threshold: float = typer.Option(
        0.25, "--text-threshold", help="Minimum text-alignment score."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON here instead of stdout."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """Run zero-shot Grounding DINO detection on IMAGE, output a Netlist."""
    from circuit_extract.detect import (
        DEFAULT_CLASSES,
        DetectionPipeline,
        GroundingDinoDetector,
    )

    _configure_logging(verbose)

    class_tuple: tuple[str, ...]
    if classes is None:
        class_tuple = DEFAULT_CLASSES
    else:
        class_tuple = tuple(c.strip() for c in classes.split(",") if c.strip())

    detector = GroundingDinoDetector(model=model)
    pipeline = DetectionPipeline(
        detector=detector,
        classes=class_tuple,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    netlist = pipeline.run(image)

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
