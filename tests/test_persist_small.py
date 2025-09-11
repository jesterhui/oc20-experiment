import os
from pathlib import Path

import pytest
import torch

from oc20_exp.data.s2ef.persist import (
    save_all_to_pytorch,
    save_all_to_lmdb,
    save_all_to_hdf5,
)


def _make_sample(max_atoms: int = 4):
    return {
        "lattice": torch.zeros(3, 3, dtype=torch.float32),
        "atomic_numbers": torch.zeros(max_atoms, dtype=torch.long),
        "fractional_coords": torch.zeros(max_atoms, 3, dtype=torch.float32),
        "mask": torch.zeros(max_atoms, dtype=torch.bool),
        "metadata": {
            "system_id": 1,
            "frame_number": 0,
            "reference_energy": -1.0,
            "file_index": 0,
            "structure_index": 0,
            "n_atoms": 0,
            "formula": "X",
            "volume": 0.0,
        },
    }


def _chunks(n=1, max_atoms: int = 4):
    chunk = [_make_sample(max_atoms) for _ in range(n)]
    yield chunk


def test_save_all_to_pytorch(tmp_path: Path):
    out = tmp_path / "ds.pt"
    save_all_to_pytorch(_chunks(), out, save_stats=True)
    assert out.exists()
    assert (tmp_path / "ds.json").exists()


def test_save_all_to_lmdb(tmp_path: Path):
    lmdb = pytest.importorskip("lmdb")
    out = tmp_path / "ds.lmdb"
    save_all_to_lmdb(_chunks(), out)
    # LMDB creates a directory on some platforms or a file on others
    assert out.exists() or out.is_dir() or os.path.isdir(out)


def test_save_all_to_hdf5(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    out = tmp_path / "ds.h5"
    save_all_to_hdf5(_chunks(), out, max_atoms=4)
    assert out.exists()
