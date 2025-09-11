"""S2EF data ingestion pipeline using type-safe data classes.

Demonstrates integration of structured data classes with the S2EF pipeline
for better type safety and cleaner code organization.
"""

import os
import lzma
import torch
import numpy as np
from pathlib import Path
from typing import List, Iterator, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

try:
    import ase
    from ase import Atoms
    from ase.io import read

    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    raise ImportError("ASE is required. Install with: pip install ase")

# Import our new data structures
from data_structures import (
    CrystalStructure,
    LatticeData,
    AtomicStructure,
    S2EFTarget,
    S2EFSample,
    BatchedS2EFSamples,
    ModelPrediction,
    collate_s2ef_samples,
    LatticeFormat,
    CoordinateSystem,
)


class TypeSafeS2EFDataIngestion:
    """
    Type-safe S2EF data ingestion pipeline using structured data classes.

    This updated version provides:
    - Better type safety with structured data classes
    - Cleaner separation of concerns
    - Built-in validation and conversion methods
    - Easier serialization and batch handling
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str = "./processed_s2ef",
        max_atoms: int = 200,
        max_workers: int = 4,
        chunk_size: int = 1000,
        lattice_format: LatticeFormat = LatticeFormat.PARAMS_6D,
    ):
        """
        Initialize type-safe S2EF data ingestion pipeline.

        Args:
            data_dir: Directory containing s2ef data files
            output_dir: Directory to save processed data
            max_atoms: Maximum number of atoms to pad to
            max_workers: Number of parallel workers for processing
            chunk_size: Number of structures to process in each chunk
            lattice_format: Lattice representation format
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_atoms = max_atoms
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.lattice_format = lattice_format

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Find all data files
        self.data_files = self._find_data_files()
        self.logger.info(f"Found {len(self.data_files)} data file pairs")

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.output_dir / "ingestion.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _find_data_files(self) -> List[tuple[Path, Path]]:
        """Find all matching .extxyz.xz and .txt.xz file pairs."""
        extxyz_files = list(self.data_dir.glob("*.extxyz.xz"))
        data_pairs = []

        for extxyz_file in sorted(extxyz_files):
            base_name = extxyz_file.stem.replace(".extxyz", "")
            txt_file = self.data_dir / f"{base_name}.txt.xz"

            if txt_file.exists():
                data_pairs.append((extxyz_file, txt_file))
            else:
                self.logger.warning(f"No metadata file found for {extxyz_file}")

        return data_pairs

    def _load_metadata(self, txt_file: Path) -> List[dict]:
        """Load metadata from compressed text file."""
        metadata_list = []
        file_index = int(txt_file.stem.replace(".txt", ""))

        with lzma.open(txt_file, "rt") as f:
            for structure_index, line in enumerate(f):
                parts = line.strip().split(",")
                if len(parts) == 3:
                    system_id, frame_number, reference_energy = parts
                    metadata_list.append(
                        {
                            "system_id": int(system_id),
                            "frame_number": int(frame_number),
                            "reference_energy": float(reference_energy),
                            "file_index": file_index,
                            "structure_index": structure_index,
                        }
                    )

        return metadata_list

    def _load_structures(self, extxyz_file: Path) -> List[Atoms]:
        """Load structures from compressed trajectory file."""
        temp_file = self.output_dir / f"temp_{extxyz_file.stem}"

        with lzma.open(extxyz_file, "rb") as compressed:
            with open(temp_file, "wb") as decompressed:
                decompressed.write(compressed.read())

        try:
            structures = read(str(temp_file), ":")
            self.logger.info(
                f"Loaded {len(structures)} structures from {extxyz_file.name}"
            )
            return structures
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def _ase_atoms_to_crystal_structure(
        self, atoms: Atoms, metadata: dict
    ) -> CrystalStructure:
        """
        Convert ASE Atoms to CrystalStructure with proper type safety.

        Args:
            atoms: ASE Atoms object
            metadata: Structure metadata dictionary

        Returns:
            Type-safe CrystalStructure object
        """
        # Get basic properties
        n_atoms = len(atoms)
        atomic_numbers = atoms.get_atomic_numbers()

        # Get unit cell and convert to desired format
        cell = atoms.get_cell()

        if self.lattice_format == LatticeFormat.MATRIX_3X3:
            lattice_data = torch.tensor(cell.array, dtype=torch.float32)
        elif self.lattice_format == LatticeFormat.PARAMS_6D:
            cellpar = cell.cellpar()  # [a, b, c, alpha_deg, beta_deg, gamma_deg]
            lattice_data = torch.tensor(cellpar, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown lattice format: {self.lattice_format}")

        lattice = LatticeData(data=lattice_data, format=self.lattice_format)

        # Get fractional coordinates
        fractional_positions = atoms.get_scaled_positions(wrap=True)

        # Pad arrays to max_atoms
        padded_atomic_numbers = np.zeros(self.max_atoms, dtype=np.int64)
        padded_fractional_coords = np.zeros((self.max_atoms, 3), dtype=np.float32)
        mask = np.zeros(self.max_atoms, dtype=bool)

        # Fill valid entries
        n_atoms_actual = min(n_atoms, self.max_atoms)
        padded_atomic_numbers[:n_atoms_actual] = atomic_numbers[:n_atoms_actual]
        padded_fractional_coords[:n_atoms_actual] = fractional_positions[
            :n_atoms_actual
        ]
        mask[:n_atoms_actual] = True

        # Create atomic structure
        atomic_structure = AtomicStructure(
            atomic_numbers=torch.tensor(padded_atomic_numbers, dtype=torch.long),
            coordinates=torch.tensor(padded_fractional_coords, dtype=torch.float32),
            coordinate_system=CoordinateSystem.FRACTIONAL,
            mask=torch.tensor(mask, dtype=torch.bool),
        )

        # Create crystal structure with metadata
        structure_metadata = {
            "system_id": metadata["system_id"],
            "frame_number": metadata["frame_number"],
            "reference_energy": metadata["reference_energy"],
            "file_index": metadata["file_index"],
            "structure_index": metadata["structure_index"],
            "n_atoms": n_atoms,
            "formula": atoms.get_chemical_formula(),
            "volume": atoms.get_volume(),
        }

        return CrystalStructure(
            lattice=lattice, atoms=atomic_structure, metadata=structure_metadata
        )

    def _create_s2ef_sample(self, atoms: Atoms, metadata: dict) -> S2EFSample:
        """
        Create a complete S2EF sample from ASE atoms and metadata.

        Args:
            atoms: ASE Atoms object
            metadata: Structure metadata

        Returns:
            Complete S2EF sample
        """
        # Convert to crystal structure
        structure = self._ase_atoms_to_crystal_structure(atoms, metadata)

        # Extract energy and forces if available
        energy = None
        forces = None

        if atoms.calc is not None:
            try:
                energy = torch.tensor(atoms.get_potential_energy(), dtype=torch.float32)
            except:
                pass

            try:
                atom_forces = atoms.get_forces()
                # Pad forces to max_atoms
                padded_forces = np.zeros((self.max_atoms, 3), dtype=np.float32)
                n_atoms_actual = min(len(atom_forces), self.max_atoms)
                padded_forces[:n_atoms_actual] = atom_forces[:n_atoms_actual]
                forces = torch.tensor(padded_forces, dtype=torch.float32)
            except:
                pass

        # Create targets
        targets = S2EFTarget(
            energy=energy, forces=forces, reference_energy=metadata["reference_energy"]
        )

        # Create sample
        sample_id = f"{metadata['file_index']}_{metadata['structure_index']}"

        return S2EFSample(
            structure=structure,
            targets=targets,
            sample_id=sample_id,
            system_id=metadata["system_id"],
            frame_number=metadata["frame_number"],
        )

    def process_file_pair(self, extxyz_file: Path, txt_file: Path) -> List[S2EFSample]:
        """
        Process a single file pair and return type-safe S2EF samples.

        Args:
            extxyz_file: Path to trajectory file
            txt_file: Path to metadata file

        Returns:
            List of S2EF samples
        """
        self.logger.info(f"Processing {extxyz_file.name} and {txt_file.name}")

        # Load metadata and structures
        metadata_list = self._load_metadata(txt_file)
        structures = self._load_structures(extxyz_file)

        # Verify counts match
        if len(metadata_list) != len(structures):
            self.logger.warning(
                f"Metadata count ({len(metadata_list)}) != structure count ({len(structures)}) "
                f"for {extxyz_file.name}"
            )
            min_count = min(len(metadata_list), len(structures))
            metadata_list = metadata_list[:min_count]
            structures = structures[:min_count]

        # Convert to S2EF samples
        samples = []
        for atoms, metadata in zip(structures, metadata_list):
            try:
                sample = self._create_s2ef_sample(atoms, metadata)
                samples.append(sample)
            except Exception as e:
                self.logger.error(
                    f"Error processing structure {metadata['structure_index']}: {e}"
                )
                continue

        return samples

    def process_all_files(self) -> Iterator[List[S2EFSample]]:
        """
        Process all data files and yield chunks of S2EF samples.

        Yields:
            Chunks of S2EF samples
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_files = {
                executor.submit(self.process_file_pair, extxyz, txt): (extxyz, txt)
                for extxyz, txt in self.data_files
            }

            for future in tqdm(
                as_completed(future_to_files), total=len(self.data_files)
            ):
                extxyz_file, txt_file = future_to_files[future]
                try:
                    samples = future.result()

                    # Yield in chunks
                    for i in range(0, len(samples), self.chunk_size):
                        chunk = samples[i : i + self.chunk_size]
                        yield chunk

                except Exception as e:
                    self.logger.error(f"Error processing {extxyz_file.name}: {e}")

    def create_batched_dataset(
        self, batch_size: int = 32
    ) -> Iterator[BatchedS2EFSamples]:
        """
        Create batched dataset from processed samples.

        Args:
            batch_size: Size of each batch

        Yields:
            Batched S2EF samples ready for model training
        """
        sample_buffer = []

        for chunk in self.process_all_files():
            sample_buffer.extend(chunk)

            # Yield complete batches
            while len(sample_buffer) >= batch_size:
                batch_samples = sample_buffer[:batch_size]
                sample_buffer = sample_buffer[batch_size:]

                batched_samples = collate_s2ef_samples(batch_samples)
                yield batched_samples

        # Yield remaining samples if any
        if sample_buffer:
            batched_samples = collate_s2ef_samples(sample_buffer)
            yield batched_samples

    def save_processed_samples(self, output_file: str = "s2ef_samples.pt"):
        """
        Save all processed samples to file.

        Args:
            output_file: Output filename
        """
        self.logger.info("Processing and saving S2EF samples...")

        all_samples = []
        for chunk in self.process_all_files():
            all_samples.extend(chunk)

        output_path = self.output_dir / output_file
        torch.save(all_samples, output_path)
        self.logger.info(f"Saved {len(all_samples)} S2EF samples to {output_path}")

        # Save summary statistics
        self._save_sample_stats(all_samples, output_path.with_suffix(".json"))

    def _save_sample_stats(self, samples: List[S2EFSample], stats_file: Path):
        """Save dataset statistics."""
        import json

        stats = {
            "total_samples": len(samples),
            "max_atoms_per_structure": self.max_atoms,
            "lattice_format": self.lattice_format.value,
            "unique_systems": len(set(s.system_id for s in samples)),
            "samples_with_energy": sum(1 for s in samples if s.targets.has_energy),
            "samples_with_forces": sum(1 for s in samples if s.targets.has_forces),
            "avg_atoms_per_structure": np.mean(
                [s.structure.metadata["n_atoms"] for s in samples]
            ),
            "avg_volume": np.mean([s.structure.metadata["volume"] for s in samples]),
        }

        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Sample statistics saved to {stats_file}")


# Example usage and testing
def demo_type_safe_pipeline():
    """Demonstrate the type-safe data pipeline."""

    # Configuration
    data_dir = "s2ef_train_200K/s2ef_train_200K"
    output_dir = "processed_s2ef_typed"

    # Initialize pipeline
    ingestion = TypeSafeS2EFDataIngestion(
        data_dir=data_dir,
        output_dir=output_dir,
        max_atoms=200,
        max_workers=2,  # Reduced for demo
        chunk_size=100,
        lattice_format=LatticeFormat.PARAMS_6D,
    )

    print("=== Type-Safe S2EF Data Pipeline Demo ===")

    # Process a few samples for demonstration
    sample_count = 0
    for chunk in ingestion.process_all_files():
        if sample_count >= 5:  # Just process a few for demo
            break

        print(f"\nProcessed chunk with {len(chunk)} samples")

        # Show first sample details
        if chunk:
            sample = chunk[0]
            print(f"Sample ID: {sample.sample_id}")
            print(f"System ID: {sample.system_id}")
            print(f"Structure: {sample.structure.metadata['formula']}")
            print(f"Lattice format: {sample.structure.lattice.format}")
            print(f"Coordinate system: {sample.structure.atoms.coordinate_system}")
            print(f"Has energy: {sample.targets.has_energy}")
            print(f"Has forces: {sample.targets.has_forces}")

            # Convert to PST input format
            pst_input = sample.to_pst_input()
            print(f"PST input keys: {list(pst_input.keys())}")
            print(
                f"PST input shapes: {[(k, v.shape) for k, v in pst_input.items() if hasattr(v, 'shape')]}"
            )

        sample_count += len(chunk)

    print(f"\nProcessed {sample_count} samples total")

    # Demonstrate batching
    print("\n=== Batching Demo ===")
    batch_count = 0
    for batched_samples in ingestion.create_batched_dataset(batch_size=4):
        if batch_count >= 2:  # Just show a couple batches
            break

        print(f"Batch {batch_count + 1}:")
        print(f"  Batch size: {batched_samples.batch_size}")
        print(f"  Device: {batched_samples.device}")

        # Convert to PST input
        pst_batch = batched_samples.to_pst_input()
        print(f"  PST batch shapes: {[(k, v.shape) for k, v in pst_batch.items()]}")

        batch_count += 1


if __name__ == "__main__":
    demo_type_safe_pipeline()
