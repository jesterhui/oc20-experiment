import pytest

ase = pytest.importorskip("ase")
from ase import Atoms

import torch

from oc20_exp.data.s2ef.types import S2EFMetadata
from oc20_exp.data.s2ef.convert import atoms_to_pst_format


def test_atoms_to_pst_format_shapes():
    atoms = Atoms("H2", cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], pbc=True)
    atoms.set_scaled_positions([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    meta = S2EFMetadata(
        system_id=1,
        frame_number=0,
        reference_energy=-1.23,
        file_index=0,
        structure_index=0,
    )

    sample = atoms_to_pst_format(atoms, meta, max_atoms=4, transform_lattice="matrix")

    assert isinstance(sample["lattice"], torch.Tensor)
    assert sample["atomic_numbers"].shape == (4,)
    assert sample["fractional_coords"].shape == (4, 3)
    assert sample["mask"].shape == (4,)
    assert sample["metadata"]["n_atoms"] == 2
