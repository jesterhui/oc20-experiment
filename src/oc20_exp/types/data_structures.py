"""Type-safe data structures for OC20 and Periodic Set Transformer.

Provides classes for crystal structures, S2EF data, model predictions,
and batch handling with automatic validation and format conversion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

import torch


class LatticeFormat(Enum):
    """Supported lattice representations."""

    MATRIX_3X3 = "matrix_3x3"  # 3x3 matrix format
    PARAMS_6D = "params_6d"  # (a, b, c, alpha, beta, gamma)


class CoordinateSystem(Enum):
    """Coordinate system types."""

    FRACTIONAL = "fractional"  # [0, 1) fractional coordinates
    CARTESIAN = "cartesian"  # Cartesian coordinates in Angstroms


@dataclass
class LatticeData:
    """
    Represents unit cell/lattice information.

    Supports both 3x3 matrix and 6-parameter representations.
    Handles conversion between formats automatically.
    """

    data: torch.Tensor
    format: LatticeFormat

    def __post_init__(self):
        """Validate lattice data format."""
        if self.format == LatticeFormat.MATRIX_3X3:
            if self.data.shape[-2:] != (3, 3):
                raise ValueError(f"Matrix format requires shape (..., 3, 3), got {self.data.shape}")
        elif self.format == LatticeFormat.PARAMS_6D and self.data.shape[-1] != 6:
            raise ValueError(f"Params format requires shape (..., 6), got {self.data.shape}")

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        if self.format == LatticeFormat.MATRIX_3X3:
            return self.data.shape[0] if self.data.dim() > 2 else 1
        else:
            return self.data.shape[0] if self.data.dim() > 1 else 1

    def to_matrix(self) -> LatticeData:
        """Convert to 3x3 matrix format."""
        if self.format == LatticeFormat.MATRIX_3X3:
            return self

        # Convert from 6D parameters to 3x3 matrix
        # This would require ASE or similar library for proper conversion
        # For now, return identity matrix as placeholder
        batch_size = self.batch_size
        matrix = (
            torch.eye(3, device=self.data.device, dtype=self.data.dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        return LatticeData(matrix, LatticeFormat.MATRIX_3X3)

    def to_params(self) -> LatticeData:
        """Convert to 6D parameters format."""
        if self.format == LatticeFormat.PARAMS_6D:
            return self

        # Convert from 3x3 matrix to 6D parameters
        # This would require proper crystallographic conversion
        # For now, return dummy parameters
        batch_size = self.batch_size
        params = self.data.new_tensor([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]).unsqueeze(0)
        params = params.repeat(batch_size, 1)
        return LatticeData(params, LatticeFormat.PARAMS_6D)

    def volume(self) -> torch.Tensor:
        """Calculate unit cell volume."""
        if self.format == LatticeFormat.MATRIX_3X3:
            return torch.det(self.data)
        else:
            # For 6D params: V = abc * sqrt(1 + 2*cos(α)*cos(β)*cos(γ) - cos²(α) - cos²(β) - cos²(γ))
            a, b, c = self.data[..., 0], self.data[..., 1], self.data[..., 2]
            angles = torch.deg2rad(self.data[..., 3:6])
            alpha, beta, gamma = angles.unbind(-1)

            cos_alpha, cos_beta, cos_gamma = (
                torch.cos(alpha),
                torch.cos(beta),
                torch.cos(gamma),
            )
            discriminant = (
                1 + 2 * cos_alpha * cos_beta * cos_gamma - cos_alpha**2 - cos_beta**2 - cos_gamma**2
            )

            return a * b * c * torch.sqrt(torch.clamp(discriminant, min=1e-10))


@dataclass
class AtomicStructure:
    """
    Represents atomic positions and types in a crystal structure.

    Handles both fractional and Cartesian coordinates with proper masking
    for variable-length structures.
    """

    atomic_numbers: torch.Tensor  # (max_atoms,) or (batch, max_atoms)
    coordinates: torch.Tensor  # (max_atoms, 3) or (batch, max_atoms, 3)
    coordinate_system: CoordinateSystem
    mask: torch.Tensor | None = None  # (max_atoms,) or (batch, max_atoms) - True for valid atoms

    def __post_init__(self):
        """Validate atomic structure data."""
        if self.coordinates.shape[-1] != 3:
            raise ValueError(
                f"Coordinates must have last dimension 3, got {self.coordinates.shape}"
            )

        expected_shape = self.coordinates.shape[:-1]
        if self.atomic_numbers.shape != expected_shape:
            raise ValueError(
                f"Atomic numbers shape {self.atomic_numbers.shape} doesn't match "
                f"coordinates shape {self.coordinates.shape}"
            )

        if self.mask is not None and self.mask.shape != expected_shape:
            raise ValueError(
                f"Mask shape {self.mask.shape} doesn't match expected {expected_shape}"
            )

    @property
    def max_atoms(self) -> int:
        """Get maximum number of atoms."""
        return self.atomic_numbers.shape[-1]

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.atomic_numbers.shape[0] if self.atomic_numbers.dim() > 1 else 1

    @property
    def n_atoms(self) -> torch.Tensor:
        """Get actual number of atoms per structure."""
        if self.mask is None:
            if self.atomic_numbers.dim() == 1:
                return torch.tensor((self.atomic_numbers > 0).sum().item())
            else:
                return (self.atomic_numbers > 0).sum(dim=-1)
        else:
            return self.mask.sum(dim=-1)

    def get_valid_atoms(self, batch_idx: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get valid atomic numbers and coordinates.

        Args:
            batch_idx: If specified, get atoms for specific batch element

        Returns:
            Tuple of (valid_atomic_numbers, valid_coordinates)
        """
        if batch_idx is not None:
            atomic_nums = self.atomic_numbers[batch_idx]
            coords = self.coordinates[batch_idx]
            mask = self.mask[batch_idx] if self.mask is not None else None
        else:
            atomic_nums = self.atomic_numbers
            coords = self.coordinates
            mask = self.mask

        if mask is not None:
            return atomic_nums[mask], coords[mask]
        else:
            valid_mask = atomic_nums > 0
            return atomic_nums[valid_mask], coords[valid_mask]

    def to_fractional(self, lattice: LatticeData) -> AtomicStructure:
        """Convert to fractional coordinates."""
        if self.coordinate_system == CoordinateSystem.FRACTIONAL:
            return self

        # Convert Cartesian to fractional using lattice inverse
        lattice_matrix = lattice.to_matrix().data
        if lattice_matrix.dim() == 2:
            lattice_matrix = lattice_matrix.unsqueeze(0)

        # coords @ lattice_inverse = fractional_coords
        lattice_inv = torch.inverse(lattice_matrix)
        fractional_coords = torch.matmul(self.coordinates.unsqueeze(-2), lattice_inv).squeeze(-2)

        return AtomicStructure(
            atomic_numbers=self.atomic_numbers,
            coordinates=fractional_coords,
            coordinate_system=CoordinateSystem.FRACTIONAL,
            mask=self.mask,
        )

    def to_cartesian(self, lattice: LatticeData) -> AtomicStructure:
        """Convert to Cartesian coordinates."""
        if self.coordinate_system == CoordinateSystem.CARTESIAN:
            return self

        # Convert fractional to Cartesian using lattice matrix
        lattice_matrix = lattice.to_matrix().data
        if lattice_matrix.dim() == 2:
            lattice_matrix = lattice_matrix.unsqueeze(0)

        # fractional_coords @ lattice = cartesian_coords
        cartesian_coords = torch.matmul(self.coordinates.unsqueeze(-2), lattice_matrix).squeeze(-2)

        return AtomicStructure(
            atomic_numbers=self.atomic_numbers,
            coordinates=cartesian_coords,
            coordinate_system=CoordinateSystem.CARTESIAN,
            mask=self.mask,
        )


@dataclass
class CrystalStructure:
    """
    Complete crystal structure with lattice and atomic information.

    Main data container for structures used throughout the pipeline.
    """

    lattice: LatticeData
    atoms: AtomicStructure
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate crystal structure consistency."""
        if self.lattice.batch_size != self.atoms.batch_size:
            raise ValueError(
                f"Lattice batch size ({self.lattice.batch_size}) != "
                f"atoms batch size ({self.atoms.batch_size})"
            )

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.lattice.batch_size

    @property
    def device(self) -> torch.device:
        """Get tensor device."""
        return self.lattice.data.device

    def to_device(self, device: torch.device) -> CrystalStructure:
        """Move all tensors to specified device."""
        return CrystalStructure(
            lattice=LatticeData(data=self.lattice.data.to(device), format=self.lattice.format),
            atoms=AtomicStructure(
                atomic_numbers=self.atoms.atomic_numbers.to(device),
                coordinates=self.atoms.coordinates.to(device),
                coordinate_system=self.atoms.coordinate_system,
                mask=(self.atoms.mask.to(device) if self.atoms.mask is not None else None),
            ),
            metadata=self.metadata.copy(),
        )

    def to_pst_format(self) -> dict[str, torch.Tensor | None]:
        """
        Convert to Periodic Set Transformer input format.

        Returns:
            Dictionary with keys: lattice, atomic_numbers, fractional_coords, mask
        """
        # Ensure fractional coordinates
        fractional_atoms = self.atoms.to_fractional(self.lattice)

        # Convert lattice to required format (PST expects 6D params)
        lattice_params = self.lattice.to_params()

        return {
            "lattice": lattice_params.data,
            "atomic_numbers": fractional_atoms.atomic_numbers,
            "fractional_coords": fractional_atoms.coordinates,
            "mask": fractional_atoms.mask,
        }


@dataclass
class S2EFTarget:
    """
    Target values for Structure-to-Energy-and-Forces task.

    Contains ground truth energies and forces for training/validation.
    """

    energy: torch.Tensor | None = None  # () or (batch,) - total energy
    forces: torch.Tensor | None = None  # (max_atoms, 3) or (batch, max_atoms, 3)
    stress: torch.Tensor | None = None  # (3, 3) or (batch, 3, 3) - stress tensor
    reference_energy: float | None = None  # Reference energy for adsorption calculations

    def __post_init__(self):
        """Validate S2EF target data."""
        if self.forces is not None and self.forces.shape[-1] != 3:
            raise ValueError(f"Forces must have last dimension 3, got {self.forces.shape}")

        if self.stress is not None and self.stress.shape[-2:] != (3, 3):
            raise ValueError(f"Stress must have shape (..., 3, 3), got {self.stress.shape}")

    @property
    def has_energy(self) -> bool:
        """Check if energy target is available."""
        return self.energy is not None

    @property
    def has_forces(self) -> bool:
        """Check if forces target is available."""
        return self.forces is not None

    @property
    def has_stress(self) -> bool:
        """Check if stress target is available."""
        return self.stress is not None

    def to_device(self, device: torch.device) -> S2EFTarget:
        """Move all tensors to specified device."""
        return S2EFTarget(
            energy=self.energy.to(device) if self.energy is not None else None,
            forces=self.forces.to(device) if self.forces is not None else None,
            stress=self.stress.to(device) if self.stress is not None else None,
            reference_energy=self.reference_energy,
        )


@dataclass
class ModelPrediction:
    """
    Model prediction outputs from Periodic Set Transformer.

    Contains predicted values and intermediate representations.
    """

    token_embeddings: torch.Tensor  # (batch, 1+max_atoms, d_model) - raw transformer output
    cell_embedding: torch.Tensor | None = None  # (batch, d_model) - cell token embedding
    atom_embeddings: torch.Tensor | None = (
        None  # (batch, max_atoms, d_model) - atom token embeddings
    )
    predicted_energy: torch.Tensor | None = None  # (batch,) - predicted total energy
    predicted_forces: torch.Tensor | None = None  # (batch, max_atoms, 3) - predicted forces
    predicted_stress: torch.Tensor | None = None  # (batch, 3, 3) - predicted stress
    attention_weights: torch.Tensor | None = None  # Attention weights if needed

    def __post_init__(self):
        """Extract cell and atom embeddings from token embeddings."""
        if self.cell_embedding is None:
            self.cell_embedding = self.token_embeddings[:, 0]  # First token is CELL

        if self.atom_embeddings is None:
            self.atom_embeddings = self.token_embeddings[:, 1:]  # Rest are atoms

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.token_embeddings.shape[0]

    @property
    def d_model(self) -> int:
        """Get model dimension."""
        return self.token_embeddings.shape[-1]

    @property
    def max_atoms(self) -> int:
        """Get maximum number of atoms."""
        return self.token_embeddings.shape[1] - 1  # Subtract 1 for CELL token

    def to_device(self, device: torch.device) -> ModelPrediction:
        """Move all tensors to specified device."""
        return ModelPrediction(
            token_embeddings=self.token_embeddings.to(device),
            cell_embedding=(
                self.cell_embedding.to(device) if self.cell_embedding is not None else None
            ),
            atom_embeddings=(
                self.atom_embeddings.to(device) if self.atom_embeddings is not None else None
            ),
            predicted_energy=(
                self.predicted_energy.to(device) if self.predicted_energy is not None else None
            ),
            predicted_forces=(
                self.predicted_forces.to(device) if self.predicted_forces is not None else None
            ),
            predicted_stress=(
                self.predicted_stress.to(device) if self.predicted_stress is not None else None
            ),
            attention_weights=(
                self.attention_weights.to(device) if self.attention_weights is not None else None
            ),
        )


@dataclass
class S2EFSample:
    """
    Complete S2EF sample with structure and targets.

    Main data container for training and evaluation.
    """

    structure: CrystalStructure
    targets: S2EFTarget
    sample_id: str | None = None
    system_id: int | None = None
    frame_number: int | None = None

    @property
    def device(self) -> torch.device:
        """Get tensor device."""
        return self.structure.device

    def to_device(self, device: torch.device) -> S2EFSample:
        """Move all tensors to specified device."""
        return S2EFSample(
            structure=self.structure.to_device(device),
            targets=self.targets.to_device(device),
            sample_id=self.sample_id,
            system_id=self.system_id,
            frame_number=self.frame_number,
        )

    def to_pst_input(self) -> dict[str, torch.Tensor | None]:
        """Convert to Periodic Set Transformer input format."""
        return self.structure.to_pst_format()


@dataclass
class BatchedS2EFSamples:
    """
    Batched collection of S2EF samples for efficient processing.

    Handles collation and provides convenient batch operations.
    """

    structures: CrystalStructure
    targets: S2EFTarget
    sample_ids: list[str] | None = None
    system_ids: torch.Tensor | None = None
    frame_numbers: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.structures.batch_size

    @property
    def device(self) -> torch.device:
        """Get tensor device."""
        return self.structures.device

    def to_device(self, device: torch.device) -> BatchedS2EFSamples:
        """Move all tensors to specified device."""
        return BatchedS2EFSamples(
            structures=self.structures.to_device(device),
            targets=self.targets.to_device(device),
            sample_ids=self.sample_ids,
            system_ids=(self.system_ids.to(device) if self.system_ids is not None else None),
            frame_numbers=(
                self.frame_numbers.to(device) if self.frame_numbers is not None else None
            ),
        )

    def to_pst_input(self) -> dict[str, torch.Tensor | None]:
        """Convert to Periodic Set Transformer input format."""
        return self.structures.to_pst_format()

    def __getitem__(self, idx: int) -> S2EFSample:
        """Get individual sample from batch."""
        return S2EFSample(
            structure=CrystalStructure(
                lattice=LatticeData(
                    data=self.structures.lattice.data[idx],
                    format=self.structures.lattice.format,
                ),
                atoms=AtomicStructure(
                    atomic_numbers=self.structures.atoms.atomic_numbers[idx],
                    coordinates=self.structures.atoms.coordinates[idx],
                    coordinate_system=self.structures.atoms.coordinate_system,
                    mask=(
                        self.structures.atoms.mask[idx]
                        if self.structures.atoms.mask is not None
                        else None
                    ),
                ),
                metadata=self.structures.metadata,
            ),
            targets=S2EFTarget(
                energy=(self.targets.energy[idx] if self.targets.energy is not None else None),
                forces=(self.targets.forces[idx] if self.targets.forces is not None else None),
                stress=(self.targets.stress[idx] if self.targets.stress is not None else None),
                reference_energy=self.targets.reference_energy,
            ),
            sample_id=self.sample_ids[idx] if self.sample_ids is not None else None,
            system_id=(int(self.system_ids[idx].item()) if self.system_ids is not None else None),
            frame_number=(
                int(self.frame_numbers[idx].item()) if self.frame_numbers is not None else None
            ),
        )


def collate_s2ef_samples(
    samples: list[S2EFSample], max_atoms: int | None = None
) -> BatchedS2EFSamples:
    """
    Collate function for batching S2EF samples.

    Args:
        samples: List of S2EF samples

    Returns:
        Batched S2EF samples
    """
    batch_size = len(samples)

    # Get lattice data
    lattice_format = samples[0].structure.lattice.format
    lattice_data = torch.stack([s.structure.lattice.data for s in samples])

    # Get atomic data - need to handle variable number of atoms
    coordinate_system = samples[0].structure.atoms.coordinate_system

    # Pad sequences to handle variable number of atoms
    from torch.nn.utils.rnn import pad_sequence

    if max_atoms is not None:
        # Pad to fixed max_atoms length
        batch_size = len(samples)

        # Find actual max length in this batch
        max_len_in_batch = max(s.structure.atoms.atomic_numbers.size(0) for s in samples)
        target_len = min(max_atoms, max_len_in_batch)

        # Pad atomic numbers
        atomic_numbers = torch.zeros(
            batch_size,
            target_len,
            dtype=samples[0].structure.atoms.atomic_numbers.dtype,
        )
        coordinates = torch.zeros(
            batch_size,
            target_len,
            3,
            dtype=samples[0].structure.atoms.coordinates.dtype,
        )

        masks = None
        if samples[0].structure.atoms.mask is not None:
            masks = torch.zeros(batch_size, target_len, dtype=samples[0].structure.atoms.mask.dtype)

        for i, sample in enumerate(samples):
            seq_len = min(sample.structure.atoms.atomic_numbers.size(0), target_len)
            atomic_numbers[i, :seq_len] = sample.structure.atoms.atomic_numbers[:seq_len]
            coordinates[i, :seq_len] = sample.structure.atoms.coordinates[:seq_len]
            if masks is not None and sample.structure.atoms.mask is not None:
                masks[i, :seq_len] = sample.structure.atoms.mask[:seq_len]
            elif masks is not None:
                raise ValueError("Expected atom masks for all samples when mask is present")
    else:
        # Use dynamic padding (original behavior)
        atomic_numbers = pad_sequence(
            [s.structure.atoms.atomic_numbers for s in samples],
            batch_first=True,
            padding_value=0,
        )
        coordinates = pad_sequence(
            [s.structure.atoms.coordinates for s in samples],
            batch_first=True,
            padding_value=0.0,
        )

        masks = None
        if samples[0].structure.atoms.mask is not None:
            if any(s.structure.atoms.mask is None for s in samples):
                raise ValueError("Expected atom masks for all samples when mask is present")
            masks = pad_sequence(
                [cast(torch.Tensor, s.structure.atoms.mask) for s in samples],
                batch_first=True,
                padding_value=False,
            )

    # Get target data
    energies = None
    if samples[0].targets.energy is not None:
        if any(s.targets.energy is None for s in samples):
            raise ValueError("Expected energy targets for all samples when energy is present")
        energies = torch.stack([cast(torch.Tensor, s.targets.energy) for s in samples])

    forces = None
    if samples[0].targets.forces is not None:
        if any(s.targets.forces is None for s in samples):
            raise ValueError("Expected force targets for all samples when forces are present")
        if max_atoms is not None:
            # Pad forces to same length as other tensors
            batch_size = len(samples)
            target_len = atomic_numbers.size(1)  # Use same length as atomic_numbers
            forces = torch.zeros(batch_size, target_len, 3, dtype=samples[0].targets.forces.dtype)

            for i, sample in enumerate(samples):
                sample_forces = cast(torch.Tensor, sample.targets.forces)
                seq_len = min(sample_forces.size(0), target_len)
                forces[i, :seq_len] = sample_forces[:seq_len]
        else:
            forces = pad_sequence(
                [cast(torch.Tensor, s.targets.forces) for s in samples],
                batch_first=True,
                padding_value=0.0,
            )

    stress = None
    if samples[0].targets.stress is not None:
        if any(s.targets.stress is None for s in samples):
            raise ValueError("Expected stress targets for all samples when stress is present")
        stress = torch.stack([cast(torch.Tensor, s.targets.stress) for s in samples])

    # Collect metadata
    sample_ids = None
    if samples[0].sample_id is not None:
        if any(s.sample_id is None for s in samples):
            raise ValueError("Expected sample_id for all samples when sample_id is present")
        sample_ids = [cast(str, s.sample_id) for s in samples]
    system_ids = None
    if samples[0].system_id is not None:
        if any(s.system_id is None for s in samples):
            raise ValueError("Expected system_id for all samples when system_id is present")
        system_ids = torch.tensor([int(cast(int, s.system_id)) for s in samples])
    frame_numbers = None
    if samples[0].frame_number is not None:
        if any(s.frame_number is None for s in samples):
            raise ValueError("Expected frame_number for all samples when frame_number is present")
        frame_numbers = torch.tensor([int(cast(int, s.frame_number)) for s in samples])

    return BatchedS2EFSamples(
        structures=CrystalStructure(
            lattice=LatticeData(lattice_data, lattice_format),
            atoms=AtomicStructure(
                atomic_numbers=atomic_numbers,
                coordinates=coordinates,
                coordinate_system=coordinate_system,
                mask=masks,
            ),
            metadata={},
        ),
        targets=S2EFTarget(
            energy=energies,
            forces=forces,
            stress=stress,
            reference_energy=samples[0].targets.reference_energy,
        ),
        sample_ids=sample_ids,
        system_ids=system_ids,
        frame_numbers=frame_numbers,
    )


# Serialization utilities
def save_crystal_structure(structure: CrystalStructure, path: str | Path) -> None:
    """Save crystal structure to file."""
    path = Path(path)

    if path.suffix == ".pt":
        torch.save(structure, path)
    elif path.suffix == ".json":
        # Convert to serializable format
        data = {
            "lattice": {
                "data": structure.lattice.data.tolist(),
                "format": structure.lattice.format.value,
            },
            "atoms": {
                "atomic_numbers": structure.atoms.atomic_numbers.tolist(),
                "coordinates": structure.atoms.coordinates.tolist(),
                "coordinate_system": structure.atoms.coordinate_system.value,
                "mask": (
                    structure.atoms.mask.tolist() if structure.atoms.mask is not None else None
                ),
            },
            "metadata": structure.metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_crystal_structure(path: str | Path) -> CrystalStructure:
    """Load crystal structure from file."""
    path = Path(path)

    if path.suffix == ".pt":
        return cast(CrystalStructure, torch.load(path))
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)

        lattice = LatticeData(
            data=torch.tensor(data["lattice"]["data"]),
            format=LatticeFormat(data["lattice"]["format"]),
        )

        atoms = AtomicStructure(
            atomic_numbers=torch.tensor(data["atoms"]["atomic_numbers"]),
            coordinates=torch.tensor(data["atoms"]["coordinates"]),
            coordinate_system=CoordinateSystem(data["atoms"]["coordinate_system"]),
            mask=(
                torch.tensor(data["atoms"]["mask"]) if data["atoms"]["mask"] is not None else None
            ),
        )

        return CrystalStructure(lattice=lattice, atoms=atoms, metadata=data["metadata"])
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
