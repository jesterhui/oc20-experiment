from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from oc20_exp.utils.logging import get_logger

from .convert import atoms_to_pst_format
from .io import find_data_files, load_metadata, load_structures
from .persist import save_all_to_hdf5, save_all_to_lmdb, save_all_to_pytorch
from .types import S2EFMetadata


class S2EFDataIngestion:
    """Pipeline for ingesting S2EF OC20 data and exporting in multiple formats."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str = "./processed_s2ef",
        max_atoms: int = 200,
        max_workers: int = 4,
        chunk_size: int = 1000,
        transform_lattice: str = "matrix",
        max_files: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_atoms = max_atoms
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.transform_lattice = transform_lattice

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__, self.output_dir)

        self.data_files: list[tuple[Path, Path]] = find_data_files(self.data_dir, self.logger)
        if max_files is not None:
            self.data_files = self.data_files[:max_files]
            self.logger.info(f"Limited to {max_files} data file pairs (test mode)")
        self.logger.info(f"Found {len(self.data_files)} data file pairs")

    def _atoms_to_pst_format(self, atoms, metadata: S2EFMetadata) -> dict:
        return atoms_to_pst_format(
            atoms,
            metadata,
            max_atoms=self.max_atoms,
            transform_lattice=self.transform_lattice,
        )

    def process_file_pair(self, extxyz_file: Path, txt_file: Path) -> list[dict]:
        self.logger.info(f"Processing {extxyz_file.name} and {txt_file.name}")
        metadata_list = load_metadata(txt_file)
        structures = load_structures(extxyz_file, self.output_dir, self.logger)

        if len(metadata_list) != len(structures):
            self.logger.warning(
                f"Metadata count ({len(metadata_list)}) != structure count ({len(structures)}) for {extxyz_file.name}"
            )
            min_count = min(len(metadata_list), len(structures))
            metadata_list = metadata_list[:min_count]
            structures = structures[:min_count]

        processed: list[dict] = []
        for atoms, meta in zip(structures, metadata_list):
            try:
                pst = self._atoms_to_pst_format(atoms, meta)
                processed.append(pst)
            except Exception as e:
                self.logger.error(f"Error processing structure {meta.structure_index}: {e}")
        return processed

    def process_all_files(self) -> Iterator[list[dict]]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_files = {
                executor.submit(self.process_file_pair, extxyz, txt): (extxyz, txt)
                for extxyz, txt in self.data_files
            }
            for future in as_completed(future_to_files):
                try:
                    structures = future.result()
                    for i in range(0, len(structures), self.chunk_size):
                        yield structures[i : i + self.chunk_size]
                except Exception as e:
                    extxyz_file, _ = future_to_files[future]
                    self.logger.error(f"Error processing {extxyz_file.name}: {e}")

    def save_to_pytorch(self, output_file: str = "s2ef_dataset.pt"):
        out = self.output_dir / output_file
        self.logger.info("Saving data to PyTorch format...")
        save_all_to_pytorch(self.process_all_files(), out, save_stats=True)
        self.logger.info(f"Saved dataset to {out}")

    def save_to_lmdb(self, output_file: str = "s2ef_dataset.lmdb"):
        out = self.output_dir / output_file
        self.logger.info("Saving data to LMDB format...")
        save_all_to_lmdb(self.process_all_files(), out)
        self.logger.info(f"Saved dataset to {out}")

    def save_to_hdf5(self, output_file: str = "s2ef_dataset.h5"):
        out = self.output_dir / output_file
        self.logger.info("Saving data to HDF5 format...")
        save_all_to_hdf5(self.process_all_files(), out, max_atoms=self.max_atoms)
        self.logger.info(f"Saved dataset to {out}")
