#!/usr/bin/env python3
"""Training script for Periodic Set Transformer on S2EF data.

This script provides a complete training pipeline for the PST model on
Structure-to-Energy-and-Forces tasks using the OC20 dataset.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src and parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from oc20_exp.models import PeriodicSetTransformer
from oc20_exp.data.s2ef import S2EFDataIngestion
from oc20_exp.types import (
    S2EFSample,
    BatchedS2EFSamples,
    collate_s2ef_samples,
    CrystalStructure,
    LatticeData,
    AtomicStructure,
    S2EFTarget,
    LatticeFormat,
    CoordinateSystem,
)


class S2EFDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for S2EF samples."""

    def __init__(self, samples_file: str):
        """Load S2EF samples from file."""
        self.samples = torch.load(samples_file, weights_only=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> S2EFSample:
        """Convert saved PST-style dicts back into typed S2EFSample objects."""
        item = self.samples[idx]

        # Lattice: detect format by shape
        lattice_tensor = item["lattice"]
        if lattice_tensor.ndim == 1 and lattice_tensor.shape[-1] == 6:
            lattice_format = LatticeFormat.PARAMS_6D
        elif lattice_tensor.ndim == 2 and lattice_tensor.shape[-2:] == (3, 3):
            lattice_format = LatticeFormat.MATRIX_3X3
        else:
            raise ValueError(f"Unrecognized lattice shape: {tuple(lattice_tensor.shape)}")

        lattice = LatticeData(data=lattice_tensor, format=lattice_format)

        # Atomic structure (PST stores fractional coords by convention)
        atomic_numbers = item["atomic_numbers"]
        fractional_coords = item["fractional_coords"]
        mask = item.get("mask")
        atoms = AtomicStructure(
            atomic_numbers=atomic_numbers,
            coordinates=fractional_coords,
            coordinate_system=CoordinateSystem.FRACTIONAL,
            mask=mask,
        )

        # Targets (optional)
        energy = item.get("energy", None)
        forces = item.get("forces", None)
        targets = S2EFTarget(energy=energy, forces=forces)

        structure = CrystalStructure(lattice=lattice, atoms=atoms, metadata=item.get("metadata", {}))

        return S2EFSample(structure=structure, targets=targets)


class S2EFModel(nn.Module):
    """S2EF model wrapper around Periodic Set Transformer."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_atoms: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pst = PeriodicSetTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.max_atoms = max_atoms

        # Energy prediction head
        self.energy_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )

        # Forces prediction head (per atom)
        self.forces_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 3),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for S2EF prediction."""
        # Get PST embeddings
        token_embeddings = self.pst(
            lattice=batch["lattice"],
            atomic_numbers=batch["atomic_numbers"],
            fractional_coords=batch["fractional_coords"],
            mask=batch["mask"],
        )

        # Extract cell and atom embeddings
        cell_embeddings = token_embeddings[:, 0]  # First token is CELL
        atom_embeddings = token_embeddings[:, 1:]  # Rest are atoms

        # Predict energy from cell embedding
        predicted_energy = self.energy_head(cell_embeddings).squeeze(-1)

        # Predict forces from atom embeddings
        predicted_forces = self.forces_head(atom_embeddings)

        return {
            "energy": predicted_energy,
            "forces": predicted_forces,
            "token_embeddings": token_embeddings,
        }


class S2EFTrainer:
    """Trainer for S2EF model."""

    def __init__(
        self,
        model: S2EFModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        device: str = "auto",
        energy_weight: float = 1.0,
        forces_weight: float = 100.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Optimizer and loss weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight

        # Loss functions
        self.energy_loss_fn = nn.MSELoss()
        self.forces_loss_fn = nn.MSELoss()

        print(f"Training on device: {self.device}")

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute S2EF losses."""
        losses = {}
        total_loss = 0.0

        # Energy loss
        if "energy" in predictions and "energy" in targets:
            energy_loss = self.energy_loss_fn(predictions["energy"], targets["energy"])
            losses["energy"] = energy_loss
            total_loss += self.energy_weight * energy_loss

        # Forces loss (only for valid atoms)
        if "forces" in predictions and "forces" in targets:
            # Apply mask to forces
            pred_forces = predictions["forces"]
            true_forces = targets["forces"]

            # Ensure mask matches the padded sequence length
            # If forces were padded, we need to adjust the mask accordingly
            if mask.size(1) != true_forces.size(1):
                if mask.size(1) > true_forces.size(1):
                    # Mask is longer than forces, truncate it
                    adjusted_mask = mask[:, :true_forces.size(1)]
                else:
                    # Forces are longer than mask, pad mask with False
                    adjusted_mask = torch.zeros(true_forces.size(0), true_forces.size(1), dtype=mask.dtype, device=mask.device)
                    adjusted_mask[:, :mask.size(1)] = mask
                forces_mask = adjusted_mask.unsqueeze(-1).expand_as(pred_forces)
            else:
                # Expand mask to cover xyz dimensions
                forces_mask = mask.unsqueeze(-1).expand_as(pred_forces)

            # Only compute loss for valid atoms
            masked_pred = pred_forces[forces_mask]
            masked_true = true_forces[forces_mask]

            if len(masked_pred) > 0:
                forces_loss = self.forces_loss_fn(masked_pred, masked_true)
                losses["forces"] = forces_loss
                total_loss += self.forces_weight * forces_loss

        losses["total"] = total_loss
        return losses

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {"total": 0.0, "energy": 0.0, "forces": 0.0}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move batch to device
            batch = batch.to_device(self.device)

            # Convert to model input format
            model_input = batch.to_pst_input()

            # Prepare targets
            targets = {}
            if batch.targets.energy is not None:
                targets["energy"] = batch.targets.energy
            if batch.targets.forces is not None:
                targets["forces"] = batch.targets.forces

            # Forward pass
            predictions = self.model(model_input)

            # Compute loss
            losses = self.compute_loss(predictions, targets, model_input["mask"])

            # Backward pass
            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()

            # Update metrics
            for key, loss in losses.items():
                epoch_losses[key] += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{losses['total'].item():.4f}",
                    "energy": (
                        f"{losses.get('energy', 0.0):.4f}"
                        if "energy" in losses
                        else "N/A"
                    ),
                    "forces": (
                        f"{losses.get('forces', 0.0):.4f}"
                        if "forces" in losses
                        else "N/A"
                    ),
                }
            )

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = {"total": 0.0, "energy": 0.0, "forces": 0.0}
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = batch.to_device(self.device)

                # Convert to model input format
                model_input = batch.to_pst_input()

                # Prepare targets
                targets = {}
                if batch.targets.energy is not None:
                    targets["energy"] = batch.targets.energy
                if batch.targets.forces is not None:
                    targets["forces"] = batch.targets.forces

                # Forward pass
                predictions = self.model(model_input)

                # Compute loss
                losses = self.compute_loss(predictions, targets, model_input["mask"])

                # Update metrics
                for key, loss in losses.items():
                    val_losses[key] += loss.item()
                num_batches += 1

        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches

        return val_losses

    def train(
        self, num_epochs: int, save_dir: str = "./checkpoints", save_every: int = 10
    ) -> Dict[str, Any]:
        """Main training loop."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        history = {"train": [], "val": []}
        best_val_loss = float("inf")

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_losses = self.train_epoch()
            history["train"].append(train_losses)

            # Validate
            val_losses = self.validate()
            if val_losses:
                history["val"].append(val_losses)

            epoch_time = time.time() - start_time

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
            print(
                f"Train - Total: {train_losses['total']:.4f}, "
                f"Energy: {train_losses.get('energy', 0.0):.4f}, "
                f"Forces: {train_losses.get('forces', 0.0):.4f}"
            )

            if val_losses:
                print(
                    f"Val   - Total: {val_losses['total']:.4f}, "
                    f"Energy: {val_losses.get('energy', 0.0):.4f}, "
                    f"Forces: {val_losses.get('forces', 0.0):.4f}"
                )

                # Save best model
                if val_losses["total"] < best_val_loss:
                    best_val_loss = val_losses["total"]
                    torch.save(self.model.state_dict(), save_dir / "best_model.pt")
                    print("→ Saved best model")

            # Periodic save
            if (epoch + 1) % save_every == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "history": history,
                    },
                    save_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                )
                print(f"→ Saved checkpoint at epoch {epoch + 1}")

        return history


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train Periodic Set Transformer on S2EF data"
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../s2ef_train_200K/s2ef_train_200K",
        help="Directory containing S2EF data files",
    )
    parser.add_argument(
        "--processed-data",
        type=str,
        help="Path to preprocessed S2EF data file (if available)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split fraction"
    )

    # Model arguments
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument(
        "--nhead", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num-layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--max-atoms", type=int, default=200, help="Maximum number of atoms"
    )

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--energy-weight", type=float, default=1.0, help="Weight for energy loss"
    )
    parser.add_argument(
        "--forces-weight", type=float, default=100.0, help="Weight for forces loss"
    )

    # Output arguments
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: process only one file pair"
    )

    args = parser.parse_args()

    print("S2EF Training Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    # Load or prepare data
    if args.processed_data and Path(args.processed_data).exists():
        print(f"Loading preprocessed data from {args.processed_data}")
        dataset = S2EFDataset(args.processed_data)
    else:
        print(f"Processing raw data from {args.data_dir}")
        # Process data using ingestion pipeline
        max_files = 4 if args.test else None
        ingestion = S2EFDataIngestion(
            data_dir=args.data_dir,
            output_dir="./temp_processed",
            max_atoms=args.max_atoms,
            max_workers=4,
            max_files=max_files,
        )

        # Save processed data
        processed_file = "temp_processed/temp_s2ef_data.pt"
        ingestion.save_to_pytorch("temp_s2ef_data.pt")
        dataset = S2EFDataset(processed_file)

    print(f"Dataset size: {len(dataset)} samples")

    # Create collate function with max_atoms parameter
    from functools import partial
    collate_fn_with_max_atoms = partial(collate_s2ef_samples, max_atoms=args.max_atoms)
    
    # Train/validation split
    if args.val_split > 0:
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_with_max_atoms,
        )
        print(f"Train: {train_size}, Val: {val_size}")
    else:
        train_dataset = dataset
        val_loader = None
        print(f"Train: {len(train_dataset)} (no validation)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_max_atoms,
    )

    # Create model
    model = S2EFModel(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_atoms=args.max_atoms,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = S2EFTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        device=args.device,
        energy_weight=args.energy_weight,
        forces_weight=args.forces_weight,
    )

    # Train
    history = trainer.train(
        num_epochs=args.epochs, save_dir=args.save_dir, save_every=args.save_every
    )

    # Save final results
    save_dir = Path(args.save_dir)
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), save_dir / "final_model.pt")

    print(f"\nTraining completed! Results saved to {save_dir}")


if __name__ == "__main__":
    main()
