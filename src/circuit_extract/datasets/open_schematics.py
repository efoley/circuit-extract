"""Loader for the ``bshada/open-schematics`` HuggingFace dataset.

~84k realistic KiCad schematics scraped from open-source hardware projects on
GitHub, each row containing an image plus a parsed-JSON dump of the
``.kicad_sch`` source. We download parquet shards on demand and extract
images to a local cache directory so the rest of the pipeline can treat each
item as a plain file pair.

Component-only ground truth is derived from the ``json`` field; net topology
(Phase B) will additionally parse ``.kicad_sch`` for labels.
"""

from __future__ import annotations

import json as _json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from circuit_extract.datasets.kicad_parser import parse_kicad_components
from circuit_extract.schema import Netlist

logger = logging.getLogger(__name__)

REPO_ID = "bshada/open-schematics"

# Parquet files live under ``data/train-00000-of-00078.parquet`` etc.
_SHARD_TEMPLATE = "data/train-{:05d}-of-{:05d}.parquet"
_TOTAL_SHARDS = 78


def _shard_filename(shard_idx: int, total: int = _TOTAL_SHARDS) -> str:
    return _SHARD_TEMPLATE.format(shard_idx, total)


@dataclass
class OpenSchematicsItem:
    """A single (image, KiCad ground truth) pair from open-schematics."""

    image_path: Path
    kicad_sch_path: Path
    stem: str
    _kicad_json: dict[str, Any] | None = field(default=None, repr=False)
    _netlist: Netlist | None = field(default=None, repr=False)

    @property
    def spice_path(self) -> Path:
        """Alias so items are drop-in-compatible with the hanky2397 loader's items."""
        return self.kicad_sch_path

    @property
    def yolo_path(self) -> Path | None:
        return None

    @property
    def ground_truth(self) -> Netlist:
        """Component-only netlist for Phase A."""
        if self._netlist is None:
            assert self._kicad_json is not None, "OpenSchematicsItem requires parsed json"
            components = parse_kicad_components(self._kicad_json)
            self._netlist = Netlist(
                components=components,
                nets=[],
                source_image=str(self.image_path),
                extractor="ground_truth:kicad_json",
            )
        return self._netlist


@dataclass
class OpenSchematicsDataset:
    """Paired schematic images + KiCad structural ground truth.

    Parameters
    ----------
    cache_dir:
        Where images and kicad_sch files are extracted to for downstream use.
    shard_indices:
        Which parquet shards to read. Defaults to shard 0 only (1,083 rows,
        one ~200 MB download) which is plenty for iterative eval.
    max_items:
        Cap on items returned after filtering.
    min_components, max_components:
        Complexity filters on the *real* component count (power/mechanical
        symbols excluded). Defaults keep the eval range manageable for a
        VLM: schematics with 6–20 real components.
    """

    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "circuit-extract" / "open_schematics"
    )
    shard_indices: tuple[int, ...] = (0,)
    max_items: int | None = None
    min_components: int = 6
    max_components: int = 20
    items: list[OpenSchematicsItem] = field(default_factory=list, repr=False)

    def load(self) -> OpenSchematicsDataset:
        """Download requested shards, filter by complexity, extract images + json."""
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        images_dir = self.cache_dir / "images"
        sch_dir = self.cache_dir / "sch"
        images_dir.mkdir(exist_ok=True)
        sch_dir.mkdir(exist_ok=True)

        accepted: list[OpenSchematicsItem] = []

        for shard_idx in self.shard_indices:
            fname = _shard_filename(shard_idx)
            logger.info("fetching shard %d (%s)...", shard_idx, fname)
            shard_path = Path(hf_hub_download(repo_id=REPO_ID, filename=fname, repo_type="dataset"))
            logger.info("reading shard %d rows...", shard_idx)
            table = pq.read_table(shard_path)
            rows = table.to_pylist()
            logger.info("shard %d has %d rows; filtering...", shard_idx, len(rows))

            for row_idx, row in enumerate(rows):
                if self.max_items is not None and len(accepted) >= self.max_items:
                    break

                img_struct = row.get("image") or {}
                img_bytes = img_struct.get("bytes")
                if not img_bytes:
                    continue

                json_text = row.get("json")
                if not json_text:
                    continue

                kicad_json = _json.loads(json_text)
                components = parse_kicad_components(kicad_json)
                if not (self.min_components <= len(components) <= self.max_components):
                    continue

                stem = _safe_stem(row.get("name") or f"shard{shard_idx}_row{row_idx}", row_idx)
                img_path = images_dir / f"{stem}.png"
                sch_path = sch_dir / f"{stem}.kicad_sch"

                if not img_path.exists():
                    img_path.write_bytes(img_bytes)
                if not sch_path.exists():
                    sch_path.write_text(row.get("schematic") or "")

                accepted.append(
                    OpenSchematicsItem(
                        image_path=img_path,
                        kicad_sch_path=sch_path,
                        stem=stem,
                        _kicad_json=kicad_json,
                    )
                )

            if self.max_items is not None and len(accepted) >= self.max_items:
                break

        logger.info(
            "accepted %d items (filter: %d <= components <= %d)",
            len(accepted),
            self.min_components,
            self.max_components,
        )
        self.items = accepted
        return self

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> OpenSchematicsItem:
        return self.items[idx]

    def __iter__(self):  # type: ignore[override]
        return iter(self.items)


def _safe_stem(name: str, row_idx: int) -> str:
    """Make a filesystem-safe stem from a dataset row's name."""
    import re

    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    cleaned = cleaned.strip("_.") or f"row{row_idx}"
    # Disambiguate with the row index so collisions across projects don't
    # clobber each other in the cache dir.
    return f"{cleaned}__{row_idx}"
