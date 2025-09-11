import pytest
import torch

from oc20_exp.models import PeriodicSetTransformer


def test_periodic_set_transformer_forward_shapes():
    d_model = 64
    batch = 2
    max_atoms = 5

    model = PeriodicSetTransformer(d_model=d_model, nhead=4, num_layers=2)

    lattice = torch.randn(batch, 6)
    atomic_numbers = torch.randint(1, 10, (batch, max_atoms))
    fractional_coords = torch.rand(batch, max_atoms, 3)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)

    out = model(
        lattice=lattice,
        atomic_numbers=atomic_numbers,
        fractional_coords=fractional_coords,
        mask=mask,
    )

    assert out.shape == (batch, 1 + max_atoms, d_model)
