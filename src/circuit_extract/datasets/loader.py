"""Download and load the ``hanky2397/schematic_images`` HuggingFace dataset.

The dataset ships as zip archives, not a standard HF dataset format, so we
use ``huggingface_hub`` to download and then unzip + pair images with their
ground-truth SPICE netlists by filename stem.

Alternatively, pass ``data_dir`` to skip the download and load from a local
directory that already contains ``images/`` and ``sp/`` subdirectories.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from circuit_extract.datasets.spice_parser import parse_spice
from circuit_extract.schema import Netlist

REPO_ID = "hanky2397/schematic_images"


@dataclass
class DatasetItem:
    """A single (image, ground-truth netlist) pair."""

    image_path: Path
    spice_path: Path
    stem: str
    _netlist: Netlist | None = field(default=None, repr=False)

    @property
    def ground_truth(self) -> Netlist:
        if self._netlist is None:
            self._netlist = parse_spice(self.spice_path)
        return self._netlist


def _download_and_extract(filename: str, cache_dir: Path) -> Path:
    """Download a zip from the HF repo and extract it."""
    from huggingface_hub import hf_hub_download

    out_dir = cache_dir / filename.removesuffix(".zip")
    if out_dir.exists() and any(out_dir.iterdir()):
        return out_dir
    zip_path = Path(hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    return out_dir


def _find_files(root: Path, suffix: str) -> dict[str, Path]:
    """Build a stem → path mapping for all files with the given suffix."""
    return {p.stem: p for p in root.rglob(f"*{suffix}")}


@dataclass
class SchematicDataset:
    """Paired schematic images + ground-truth SPICE netlists.

    Parameters
    ----------
    cache_dir:
        Where to store downloaded/extracted files. Defaults to
        ``~/.cache/circuit-extract/schematic_images``.
    max_items:
        Cap on the number of items to load (``None`` = all). Useful for
        fast iteration.
    data_dir:
        Path to a pre-downloaded dataset directory. Must contain ``images/``
        and ``sp/`` subdirectories with matching filenames. When set, skips
        the HuggingFace download entirely.
    """

    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "circuit-extract" / "schematic_images"
    )
    max_items: int | None = None
    data_dir: Path | None = None
    items: list[DatasetItem] = field(default_factory=list, repr=False)

    def load(self) -> SchematicDataset:
        """Download (if needed), extract, and pair images with netlists."""
        if self.data_dir is not None:
            images_dir, spice_dir = self._load_local()
        else:
            images_dir, spice_dir = self._load_remote()

        image_map = _find_files(images_dir, ".png") | _find_files(images_dir, ".jpg")
        spice_map = _find_files(spice_dir, ".sp")

        paired_stems = sorted(set(image_map) & set(spice_map))
        if self.max_items is not None:
            paired_stems = paired_stems[: self.max_items]

        self.items = [
            DatasetItem(
                image_path=image_map[stem],
                spice_path=spice_map[stem],
                stem=stem,
            )
            for stem in paired_stems
        ]
        return self

    def _load_local(self) -> tuple[Path, Path]:
        assert self.data_dir is not None
        images_dir = self.data_dir / "images"
        spice_dir = self.data_dir / "sp"
        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Expected 'images/' subdirectory in {self.data_dir}, not found."
            )
        if not spice_dir.is_dir():
            raise FileNotFoundError(f"Expected 'sp/' subdirectory in {self.data_dir}, not found.")
        return images_dir, spice_dir

    def _load_remote(self) -> tuple[Path, Path]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        images_dir = _download_and_extract("images.zip", self.cache_dir)
        spice_dir = _download_and_extract("sp.zip", self.cache_dir)
        return images_dir, spice_dir

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> DatasetItem:
        return self.items[idx]

    def __iter__(self):  # type: ignore[override]
        return iter(self.items)
