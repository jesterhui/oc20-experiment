"""Persistence utilities for saving S2EF data in multiple formats.

This module provides functions to save processed S2EF data to PyTorch,
LMDB, and HDF5 formats, along with dataset statistics.
"""

import io as _io
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch


def save_all_to_pytorch(
    chunks: Iterator[list[dict]],
    output_path: Path,
    *,
    save_stats: bool = True,
):
    """Save all chunks to a single PyTorch file.

    Args:
        chunks: Iterator of data chunks (each chunk is a list of dicts).
        output_path: Path where the PyTorch file will be saved.
        save_stats: If True, save dataset statistics to a JSON file.
    """
    all_data: list[dict] = []
    for chunk in chunks:
        all_data.extend(chunk)
    torch.save(all_data, output_path)
    if save_stats:
        save_dataset_stats(all_data, output_path.with_suffix(".json"))


def save_all_to_lmdb(
    chunks: Iterator[list[dict]],
    output_path: Path,
    *,
    map_size_bytes: int = 50 * 1024**3,
):
    """Save all chunks to an LMDB database for fast random access.

    Args:
        chunks: Iterator of data chunks (each chunk is a list of dicts).
        output_path: Path where the LMDB database will be created.
        map_size_bytes: Maximum size of the database in bytes (default 50GB).
    """
    try:
        import lmdb
    except ImportError as e:
        raise ImportError("LMDB not available. Install with: pip install lmdb") from e

    env = lmdb.open(str(output_path), map_size=map_size_bytes)
    total_count = 0
    with env.begin(write=True) as txn:
        for chunk in chunks:
            for i, data in enumerate(chunk):
                key = f"{total_count + i}".encode()
                buf = _io.BytesIO()
                torch.save(data, buf)
                value = buf.getvalue()
                txn.put(key, value)
            total_count += len(chunk)
    env.close()


def save_all_to_hdf5(
    chunks: Iterable[list[dict]],
    output_path: Path,
    *,
    max_atoms: int,
):
    """Save all chunks to an HDF5 file with resizable datasets.

    Args:
        chunks: Iterable of data chunks (each chunk is a list of dicts).
        output_path: Path where the HDF5 file will be saved.
        max_atoms: Maximum number of atoms per structure for padding.
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError("HDF5 not available. Install with: pip install h5py") from e

    it = iter(chunks)
    first_chunk = next(it)
    sample = first_chunk[0]

    estimated_total = 5000
    lattice_shape = (estimated_total,) + tuple(sample["lattice"].shape)
    coords_shape = (estimated_total,) + tuple(sample["fractional_coords"].shape)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("lattice", lattice_shape, dtype=np.float32)
        f.create_dataset("atomic_numbers", (estimated_total, max_atoms), dtype=np.int64)
        f.create_dataset("fractional_coords", coords_shape, dtype=np.float32)
        f.create_dataset("mask", (estimated_total, max_atoms), dtype=bool)
        f.create_dataset("system_ids", (estimated_total,), dtype=np.int64)
        f.create_dataset("reference_energies", (estimated_total,), dtype=np.float32)

        idx = 0

        for data in first_chunk:
            f["lattice"][idx] = data["lattice"].numpy()
            f["atomic_numbers"][idx] = data["atomic_numbers"].numpy()
            f["fractional_coords"][idx] = data["fractional_coords"].numpy()
            f["mask"][idx] = data["mask"].numpy()
            f["system_ids"][idx] = data["metadata"]["system_id"]
            f["reference_energies"][idx] = data["metadata"]["reference_energy"]
            idx += 1

        for chunk in it:
            for data in chunk:
                if idx >= estimated_total:
                    break
                f["lattice"][idx] = data["lattice"].numpy()
                f["atomic_numbers"][idx] = data["atomic_numbers"].numpy()
                f["fractional_coords"][idx] = data["fractional_coords"].numpy()
                f["mask"][idx] = data["mask"].numpy()
                f["system_ids"][idx] = data["metadata"]["system_id"]
                f["reference_energies"][idx] = data["metadata"]["reference_energy"]
                idx += 1

        f["lattice"].resize((idx,) + tuple(sample["lattice"].shape))
        f["atomic_numbers"].resize((idx, max_atoms))
        f["fractional_coords"].resize((idx,) + tuple(sample["fractional_coords"].shape))
        f["mask"].resize((idx, max_atoms))
        f["system_ids"].resize((idx,))
        f["reference_energies"].resize((idx,))


def save_dataset_stats(data: list[dict], stats_file: Path):
    """Save dataset statistics to a JSON file.

    Args:
        data: List of data dictionaries containing metadata.
        stats_file: Path where the JSON statistics file will be saved.
    """
    stats = {
        "total_structures": len(data),
        "max_atoms_per_structure": int(max(d["metadata"]["n_atoms"] for d in data) if data else 0),
        "unique_systems": len({d["metadata"]["system_id"] for d in data}),
        "formulas": sorted({d["metadata"]["formula"] for d in data})[:100],
        "avg_atoms_per_structure": (
            float(np.mean([d["metadata"]["n_atoms"] for d in data])) if data else 0.0
        ),
        "avg_volume": (float(np.mean([d["metadata"]["volume"] for d in data])) if data else 0.0),
    }
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
