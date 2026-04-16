"""Tests for the dataset loader using a local data_dir."""

from __future__ import annotations

from pathlib import Path

import pytest

from circuit_extract.datasets.loader import SchematicDataset

SAMPLE_SP = """\

.subckt test_circuit
r0 net0 net1 r
r1 net1 gnd r
v2 net0 gnd v
.ends
"""


def _make_dataset(tmp_path: Path, n: int = 3) -> Path:
    """Create a minimal local dataset directory."""
    images_dir = tmp_path / "images"
    sp_dir = tmp_path / "sp"
    images_dir.mkdir()
    sp_dir.mkdir()

    for i in range(n):
        stem = f"test_{i:04d}"
        # Minimal valid PNG (1x1 white pixel) — enough for the loader to pair
        (images_dir / f"{stem}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        (sp_dir / f"{stem}.sp").write_text(SAMPLE_SP)

    return tmp_path


def test_load_local_data_dir(tmp_path: Path) -> None:
    data_dir = _make_dataset(tmp_path, n=5)
    ds = SchematicDataset(data_dir=data_dir).load()
    assert len(ds) == 5
    assert all(item.image_path.exists() for item in ds)
    assert all(item.spice_path.exists() for item in ds)


def test_load_with_max_items(tmp_path: Path) -> None:
    data_dir = _make_dataset(tmp_path, n=10)
    ds = SchematicDataset(data_dir=data_dir, max_items=3).load()
    assert len(ds) == 3


def test_ground_truth_parses(tmp_path: Path) -> None:
    data_dir = _make_dataset(tmp_path, n=1)
    ds = SchematicDataset(data_dir=data_dir).load()
    gt = ds[0].ground_truth
    assert len(gt.components) == 3
    types = {c.id: c.type for c in gt.components}
    assert types["r0"] == "resistor"
    assert types["v2"] == "voltage_source"


def test_missing_images_dir(tmp_path: Path) -> None:
    (tmp_path / "sp").mkdir()
    with pytest.raises(FileNotFoundError, match="images"):
        SchematicDataset(data_dir=tmp_path).load()


def test_missing_sp_dir(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError, match="sp"):
        SchematicDataset(data_dir=tmp_path).load()


def test_iteration(tmp_path: Path) -> None:
    data_dir = _make_dataset(tmp_path, n=3)
    ds = SchematicDataset(data_dir=data_dir).load()
    stems = [item.stem for item in ds]
    assert stems == ["test_0000", "test_0001", "test_0002"]
