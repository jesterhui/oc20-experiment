import torch

from periodic_set_transformer import PeriodicSetTransformer


def run_with_lengths_angles():
    batch, max_atoms = 2, 4
    model = PeriodicSetTransformer(
        d_model=128, nhead=4, num_layers=2, dim_feedforward=256
    )

    # Lattice as (a, b, c, alpha_deg, beta_deg, gamma_deg)
    lengths_angles = torch.tensor(
        [
            [3.0, 3.0, 3.0, 90.0, 90.0, 90.0],
            [4.5, 5.0, 6.0, 80.0, 95.0, 110.0],
        ],
        dtype=torch.float32,
    )

    atomic_numbers = torch.randint(low=1, high=15, size=(batch, max_atoms))
    fractional_coords = torch.rand(batch, max_atoms, 3)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)

    out = model(lengths_angles, atomic_numbers, fractional_coords, mask)
    print("Output (lengths/angles lattice) shape:", out.shape)


if __name__ == "__main__":
    run_with_lengths_angles()
