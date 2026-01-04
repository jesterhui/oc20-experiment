from typing import Union

import numpy as np
import torch

try:
    from ase import Atoms
except ImportError as e:
    raise ImportError("ASE is required. Install with: pip install ase") from e

from .types import S2EFMetadata


def atoms_to_pst_format(
    atoms: Atoms,
    metadata: S2EFMetadata,
    *,
    max_atoms: int,
    transform_lattice: str = "matrix",
) -> dict[str, Union[torch.Tensor, dict]]:
    """Convert ASE Atoms to Periodic Set Transformer (PST) sample dict."""
    n_atoms = len(atoms)
    atomic_numbers = atoms.get_atomic_numbers()

    cell = atoms.get_cell()
    if transform_lattice == "matrix":
        lattice = torch.tensor(cell.array, dtype=torch.float32)
    elif transform_lattice == "params":
        lattice = torch.tensor(cell.cellpar(), dtype=torch.float32)
    else:
        raise ValueError(f"Unknown transform_lattice: {transform_lattice}")

    fractional_positions = atoms.get_scaled_positions(wrap=True)

    padded_atomic_numbers = np.zeros(max_atoms, dtype=np.int64)
    padded_fractional_coords = np.zeros((max_atoms, 3), dtype=np.float32)
    mask = np.zeros(max_atoms, dtype=bool)

    n_atoms_actual = min(n_atoms, max_atoms)
    padded_atomic_numbers[:n_atoms_actual] = atomic_numbers[:n_atoms_actual]
    padded_fractional_coords[:n_atoms_actual] = fractional_positions[:n_atoms_actual]
    mask[:n_atoms_actual] = True

    energy = atoms.get_potential_energy() if atoms.calc is not None else None
    forces = atoms.get_forces() if atoms.calc is not None else None

    return {
        "lattice": lattice,
        "atomic_numbers": torch.tensor(padded_atomic_numbers, dtype=torch.long),
        "fractional_coords": torch.tensor(padded_fractional_coords, dtype=torch.float32),
        "mask": torch.tensor(mask, dtype=torch.bool),
        "energy": (torch.tensor(energy, dtype=torch.float32) if energy is not None else None),
        "forces": (torch.tensor(forces, dtype=torch.float32) if forces is not None else None),
        "metadata": {
            "system_id": metadata.system_id,
            "frame_number": metadata.frame_number,
            "reference_energy": metadata.reference_energy,
            "file_index": metadata.file_index,
            "structure_index": metadata.structure_index,
            "n_atoms": n_atoms,
            "formula": atoms.get_chemical_formula(),
            "volume": atoms.get_volume(),
        },
    }
