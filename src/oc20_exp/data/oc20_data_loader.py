"""
OC20 Data Loader for Periodic Set Transformer

This module provides utilities to load OC20 (Open Catalyst 2020) data
and convert it to the format required by the Periodic Set Transformer:
- lattice: 3x3 lattice matrix or 6D (a,b,c,α,β,γ)
- atomic_numbers: integer atomic numbers
- fractional_coords: fractional coordinates in [0,1)
"""

import warnings
from typing import Optional, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import ase  # noqa: F401
    from ase import Atoms
    from ase.io import read

    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    warnings.warn("ASE not installed. Install with: pip install ase", stacklevel=2)

try:
    import lmdb  # noqa: F401

    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False
    warnings.warn("LMDB not installed. Install with: pip install lmdb", stacklevel=2)

try:
    from fairchem.core.common.data_utils import collate_fn  # noqa: F401
    from fairchem.core.datasets import LmdbDataset

    HAS_FAIRCHEM = True
except ImportError:
    HAS_FAIRCHEM = False
    warnings.warn("FairChem not installed. Install with: pip install fairchem-core", stacklevel=2)


class OC20ToPST(Dataset):
    """
    Dataset class that loads OC20 data and converts it to Periodic Set Transformer format.

    Supports multiple input formats:
    1. LMDB files (using FairChem)
    2. ASE trajectory files
    3. Raw ASE Atoms objects
    """

    def __init__(
        self,
        data_path: str,
        max_atoms: int = 100,
        data_format: str = "lmdb",
        transform_lattice: str = "params",  # "matrix" or "params"
    ):
        """
        Initialize OC20 to PST dataset.

        Args:
            data_path: Path to data file/directory
            max_atoms: Maximum number of atoms to pad to
            data_format: "lmdb", "traj", or "atoms"
            transform_lattice: "matrix" for 3x3 or "params" for 6D (a,b,c,α,β,γ)
        """
        self.data_path = data_path
        self.max_atoms = max_atoms
        self.data_format = data_format.lower()
        self.transform_lattice = transform_lattice

        # Load data based on format
        if self.data_format == "lmdb":
            if not HAS_FAIRCHEM:
                raise ImportError("FairChem required for LMDB format")
            self.dataset = LmdbDataset({"src": data_path})
        elif self.data_format == "traj":
            if not HAS_ASE:
                raise ImportError("ASE required for trajectory format")
            self.atoms_list = read(data_path, ":")
        elif self.data_format == "atoms":
            # Assume data_path is a list of Atoms objects
            self.atoms_list = data_path
        else:
            raise ValueError(f"Unsupported data_format: {data_format}")

    def __len__(self) -> int:
        if self.data_format == "lmdb":
            return len(self.dataset)
        else:
            return len(self.atoms_list)

    def __getitem__(self, idx: int) -> dict[str, object]:
        """
        Get item converted to PST format.

        Returns:
            Dict containing:
            - lattice: (3, 3) or (6,) tensor
            - atomic_numbers: (max_atoms,) tensor
            - fractional_coords: (max_atoms, 3) tensor
            - mask: (max_atoms,) boolean tensor
            - metadata: dict with additional info
        """
        # Get atoms object
        if self.data_format == "lmdb":
            data_obj = self.dataset[idx]
            atoms = self._data_obj_to_atoms(data_obj)
        else:
            atoms = self.atoms_list[idx]

        # Convert to PST format
        return self.atoms_to_pst_format(atoms)

    def _data_obj_to_atoms(self, data_obj) -> Atoms:
        """Convert FairChem data object to ASE Atoms."""
        if not HAS_ASE:
            raise ImportError("ASE required for atoms conversion")

        # Extract positions, atomic numbers, and cell from data object
        pos = data_obj.pos.numpy()
        atomic_numbers = data_obj.atomic_numbers.numpy()

        # Handle cell information
        if hasattr(data_obj, "cell"):
            cell = data_obj.cell.numpy().reshape(3, 3)
        else:
            # Default to identity if no cell info
            cell = np.eye(3) * 10.0

        # Create ASE Atoms object
        atoms = Atoms(
            numbers=atomic_numbers,
            positions=pos,
            cell=cell,
            pbc=True,  # Periodic boundary conditions
        )

        return atoms

    def atoms_to_pst_format(self, atoms: Atoms) -> dict[str, object]:
        """
        Convert ASE Atoms to Periodic Set Transformer format.

        Args:
            atoms: ASE Atoms object

        Returns:
            Dictionary with lattice, atomic_numbers, fractional_coords, mask
        """
        if not HAS_ASE:
            raise ImportError("ASE required for atoms conversion")

        # Get basic properties
        n_atoms = len(atoms)
        atomic_numbers = atoms.get_atomic_numbers()

        # Get unit cell
        cell = atoms.get_cell()

        # Convert to lattice format
        if self.transform_lattice == "matrix":
            # 3x3 lattice matrix
            lattice = torch.tensor(cell.array, dtype=torch.float32)
        elif self.transform_lattice == "params":
            # 6D parameters (a, b, c, alpha, beta, gamma)
            cellpar = cell.cellpar()  # [a, b, c, alpha_deg, beta_deg, gamma_deg]
            lattice = torch.tensor(cellpar, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown transform_lattice: {self.transform_lattice}")

        # Get fractional coordinates
        fractional_positions = atoms.get_scaled_positions(wrap=True)

        # Pad arrays to max_atoms
        padded_atomic_numbers = np.zeros(self.max_atoms, dtype=np.int64)
        padded_fractional_coords = np.zeros((self.max_atoms, 3), dtype=np.float32)
        mask = np.zeros(self.max_atoms, dtype=bool)

        # Fill valid entries
        n_atoms_actual = min(n_atoms, self.max_atoms)
        padded_atomic_numbers[:n_atoms_actual] = atomic_numbers[:n_atoms_actual]
        padded_fractional_coords[:n_atoms_actual] = fractional_positions[:n_atoms_actual]
        mask[:n_atoms_actual] = True

        # Convert to tensors
        result: dict[str, object] = {
            "lattice": lattice,
            "atomic_numbers": torch.tensor(padded_atomic_numbers, dtype=torch.long),
            "fractional_coords": torch.tensor(padded_fractional_coords, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "metadata": {
                "n_atoms": n_atoms,
                "formula": atoms.get_chemical_formula(),
                "volume": atoms.get_volume(),
            },
        }

        return result


def collate_pst_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    """
    Collate function for batching PST format data.

    Args:
        batch: List of PST format dictionaries

    Returns:
        Batched dictionary with batch dimension added
    """
    keys = ["lattice", "atomic_numbers", "fractional_coords", "mask"]

    result: dict[str, object] = {}
    for key in keys:
        result[key] = torch.stack(
            [cast(torch.Tensor, item[key]) for item in batch],
            dim=0,
        )

    # Handle metadata separately
    result["metadata"] = [cast(dict[str, object], item["metadata"]) for item in batch]

    return result


class OC20DataModule:
    """
    PyTorch Lightning-style data module for OC20 data.
    Handles train/val/test splits and creates data loaders.
    """

    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        max_atoms: int = 100,
        batch_size: int = 32,
        num_workers: int = 4,
        data_format: str = "lmdb",
        transform_lattice: str = "matrix",
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.max_atoms = max_atoms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_format = data_format
        self.transform_lattice = transform_lattice

    def train_dataloader(self) -> DataLoader:
        dataset = OC20ToPST(
            self.train_path,
            max_atoms=self.max_atoms,
            data_format=self.data_format,
            transform_lattice=self.transform_lattice,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_pst_batch,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_path is None:
            return None
        dataset = OC20ToPST(
            self.val_path,
            max_atoms=self.max_atoms,
            data_format=self.data_format,
            transform_lattice=self.transform_lattice,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pst_batch,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_path is None:
            return None
        dataset = OC20ToPST(
            self.test_path,
            max_atoms=self.max_atoms,
            data_format=self.data_format,
            transform_lattice=self.transform_lattice,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pst_batch,
        )


def create_example_from_cif(cif_path: str) -> dict[str, object]:
    """
    Create PST format data from a CIF file (useful for testing).

    Args:
        cif_path: Path to CIF file

    Returns:
        PST format dictionary
    """
    if not HAS_ASE:
        raise ImportError("ASE required for CIF reading")

    atoms = read(cif_path)
    dataset = OC20ToPST("dummy", max_atoms=200, data_format="atoms")
    return dataset.atoms_to_pst_format(atoms)


def download_oc20_sample():
    """
    Instructions for downloading OC20 sample data.
    """
    instructions = """
    To download OC20 sample data:

    1. Using FairChem download script:
       ```bash
       python -c "from fairchem.core.scripts.download_data import download_data; download_data('s2ef', '200k')"
       ```

    2. Manual download from https://fair-chem.github.io/catalysts/datasets/oc20.html
       - Training sets: 200K, 2M, 20M, All
       - Validation sets: id, ood_ads, ood_cat, ood_both
       - Test sets: id, ood_ads, ood_cat, ood_both

    3. Data will be in LMDB format, ready for use with this loader
    """
    print(instructions)


if __name__ == "__main__":
    print("OC20 Data Loader for Periodic Set Transformer")
    print("=" * 50)

    # Show download instructions
    download_oc20_sample()

    # Example usage (uncomment when you have data)
    # data_module = OC20DataModule(
    #     train_path="path/to/train.lmdb",
    #     val_path="path/to/val.lmdb",
    #     max_atoms=100,
    #     batch_size=16
    # )
    #
    # train_loader = data_module.train_dataloader()
    # for batch in train_loader:
    #     print(f"Batch shapes:")
    #     print(f"  Lattice: {batch['lattice'].shape}")
    #     print(f"  Atomic numbers: {batch['atomic_numbers'].shape}")
    #     print(f"  Fractional coords: {batch['fractional_coords'].shape}")
    #     print(f"  Mask: {batch['mask'].shape}")
    #     break
