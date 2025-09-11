"""
Training script for Periodic Set Transformer with OC20 data

This script demonstrates how to:
1. Load OC20 data using the custom data loader
2. Train the Periodic Set Transformer
3. Handle different data formats (LMDB, trajectory files)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, Any, Optional

from periodic_set_transformer import PeriodicSetTransformer
from oc20_data_loader import OC20ToPST, OC20DataModule, collate_pst_batch

# Try to import wandb for logging
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("wandb not available. Install with: pip install wandb")


class OC20Trainer:
    """
    Trainer class for Periodic Set Transformer on OC20 data.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "auto",
        learning_rate: float = 1e-4,
        use_wandb: bool = False,
        project_name: str = "pst-oc20",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_wandb = use_wandb

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Setup loss function (example: energy prediction)
        self.criterion = nn.MSELoss()

        # Initialize wandb if requested
        if self.use_wandb and HAS_WANDB:
            wandb.init(project=project_name)
            wandb.config.update(
                {
                    "model": "PeriodicSetTransformer",
                    "learning_rate": learning_rate,
                    "device": str(self.device),
                }
            )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            self.optimizer.zero_grad()

            output = self.model(
                lattice=batch["lattice"],
                atomic_numbers=batch["atomic_numbers"],
                fractional_coords=batch["fractional_coords"],
                mask=batch["mask"],
            )

            # Example loss: predict energy from [CELL] token
            cell_token = output[:, 0, :]  # First token is [CELL]

            # For demonstration, create dummy target energies
            # In practice, you'd get these from your data
            batch_size = cell_token.shape[0]
            target_energy = torch.randn(batch_size, 1, device=self.device)

            # Simple energy prediction head
            if not hasattr(self, "energy_head"):
                self.energy_head = nn.Linear(self.model.d_model, 1).to(self.device)

            predicted_energy = self.energy_head(cell_token)
            loss = self.criterion(predicted_energy, target_energy)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            if self.use_wandb and HAS_WANDB:
                wandb.log({"train_loss_step": loss.item()})

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    def validate(self) -> Optional[Dict[str, float]]:
        """Validate the model."""
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                output = self.model(
                    lattice=batch["lattice"],
                    atomic_numbers=batch["atomic_numbers"],
                    fractional_coords=batch["fractional_coords"],
                    mask=batch["mask"],
                )

                # Example validation loss
                cell_token = output[:, 0, :]
                batch_size = cell_token.shape[0]
                target_energy = torch.randn(batch_size, 1, device=self.device)

                predicted_energy = self.energy_head(cell_token)
                loss = self.criterion(predicted_energy, target_energy)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {"val_loss": avg_loss}

    def train(self, num_epochs: int):
        """Train the model for multiple epochs."""
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Log metrics
            print(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}")
            if val_metrics:
                print(f"Epoch {epoch}: Val Loss: {val_metrics['val_loss']:.4f}")

            if self.use_wandb and HAS_WANDB:
                log_dict = train_metrics
                if val_metrics:
                    log_dict.update(val_metrics)
                wandb.log(log_dict)


def create_synthetic_oc20_data(n_samples: int = 100, max_atoms: int = 50):
    """
    Create synthetic data that mimics OC20 format for testing.
    """
    print(f"Creating synthetic OC20-style data with {n_samples} samples...")

    synthetic_data = []

    for i in range(n_samples):
        # Random lattice parameters (cubic to orthorhombic)
        a = np.random.uniform(4.0, 8.0)
        b = np.random.uniform(4.0, 8.0)
        c = np.random.uniform(4.0, 8.0)
        alpha = beta = gamma = 90.0  # Orthorhombic

        lattice = torch.tensor([a, b, c, alpha, beta, gamma], dtype=torch.float32)

        # Random number of atoms
        n_atoms = np.random.randint(10, max_atoms)

        # Random atomic numbers (common elements in catalysis)
        element_choices = [
            1,
            6,
            7,
            8,
            26,
            28,
            29,
            47,
            79,
        ]  # H, C, N, O, Fe, Ni, Cu, Ag, Au
        atomic_numbers = np.random.choice(element_choices, n_atoms)

        # Pad to max_atoms
        padded_numbers = np.zeros(max_atoms, dtype=np.int64)
        padded_numbers[:n_atoms] = atomic_numbers

        # Random fractional coordinates
        fractional_coords = np.random.uniform(0, 1, (max_atoms, 3)).astype(np.float32)

        # Mask
        mask = np.zeros(max_atoms, dtype=bool)
        mask[:n_atoms] = True

        sample = {
            "lattice": lattice,
            "atomic_numbers": torch.tensor(padded_numbers, dtype=torch.long),
            "fractional_coords": torch.tensor(fractional_coords, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "metadata": {"n_atoms": n_atoms, "formula": f"synthetic_{i}"},
        }

        synthetic_data.append(sample)

    return synthetic_data


def main():
    parser = argparse.ArgumentParser(description="Train PST on OC20 data")
    parser.add_argument("--data_path", type=str, help="Path to OC20 LMDB data")
    parser.add_argument(
        "--data_format",
        choices=["lmdb", "traj", "synthetic"],
        default="synthetic",
        help="Data format",
    )
    parser.add_argument(
        "--max_atoms", type=int, default=100, help="Max atoms per structure"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb logging")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Periodic Set Transformer - OC20 Training")

    # Create model
    model = PeriodicSetTransformer(
        d_model=args.d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=args.d_model * 2,
        fourier_features_dim=32,  # latent positional features
    )

    # Create data loaders
    if args.data_format == "synthetic":
        logger.info("Using synthetic data for demonstration...")
        train_data = create_synthetic_oc20_data(n_samples=200, max_atoms=args.max_atoms)
        val_data = create_synthetic_oc20_data(n_samples=50, max_atoms=args.max_atoms)

        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_pst_batch,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_pst_batch,
        )

    else:
        if args.data_path is None:
            raise ValueError("--data_path required for real data")

        # Real OC20 data
        data_module = OC20DataModule(
            train_path=args.data_path,
            max_atoms=args.max_atoms,
            batch_size=args.batch_size,
            data_format=args.data_format,
            transform_lattice="params",  # Use 6D parameters
        )

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

    # Test data loading
    logger.info("Testing data loading...")
    for batch in train_loader:
        print(f"Batch shapes:")
        print(f"  Lattice: {batch['lattice'].shape}")
        print(f"  Atomic numbers: {batch['atomic_numbers'].shape}")
        print(f"  Fractional coords: {batch['fractional_coords'].shape}")
        print(f"  Mask: {batch['mask'].shape}")
        break

    # Create trainer and train
    trainer = OC20Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        use_wandb=args.use_wandb,
    )

    trainer.train(num_epochs=args.epochs)

    print("Training completed!")


if __name__ == "__main__":
    main()
