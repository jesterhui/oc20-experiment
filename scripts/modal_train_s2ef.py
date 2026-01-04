#!/usr/bin/env python3
"""Modal GPU training script for Periodic Set Transformer on S2EF data.

This script deploys the S2EF training pipeline to Modal GPUs for accelerated training.
It handles data upload, GPU training, and result download automatically.
"""

import io
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

import modal


def safe_extract(tar, path):
    """Safely extract tar archive, preventing path traversal attacks."""
    for member in tar.getmembers():
        # Prevent path traversal
        member_path = Path(path) / member.name
        try:
            member_path.resolve().relative_to(Path(path).resolve())
        except ValueError as err:
            raise ValueError(f"Attempted path traversal in tar file: {member.name}") from err
    tar.extractall(path)


# Create Modal image from custom Dockerfile
image = modal.Image.from_dockerfile("dockerfiles/Dockerfile.s2ef")

app = modal.App("s2ef-training", image=image)

# Create a shared volume for data persistence
data_volume = modal.Volume.from_name("s2ef-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("s2ef-checkpoints", create_if_missing=True)


@app.function(
    gpu="T4:1",  # Single T4 GPU - can be changed to L4, A100, H100, etc.
    memory=16384,  # 16GB RAM
    timeout=86400,  # 24 hours max
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
)
@modal.concurrent(max_inputs=1)
def train_s2ef_on_gpu(
    training_config: dict[str, Any],
    source_code_tarball: bytes,
    data_tarball: Optional[bytes] = None,
) -> dict[str, Any]:
    """
    Main training function that runs on Modal GPU.

    Args:
        training_config: Dictionary with training parameters
        source_code_tarball: Compressed source code
        data_tarball: Optional compressed data files

    Returns:
        Training results and metrics
    """
    import json
    import logging
    import sys
    import tarfile
    import tempfile

    import torch

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger("modal-s2ef-training")

    logger.info(f"Starting S2EF training on {torch.cuda.get_device_name()}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Extract source code
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract source code
        with tarfile.open(fileobj=io.BytesIO(source_code_tarball), mode="r:gz") as tar:
            safe_extract(tar, temp_path)

        # Find the extracted project directory
        project_dirs = [d for d in temp_path.iterdir() if d.is_dir() and (d / "src").exists()]
        if not project_dirs:
            raise ValueError("Could not find project directory with src/ folder")

        project_dir = project_dirs[0]
        sys.path.insert(0, str(project_dir / "src"))
        sys.path.insert(0, str(project_dir))

        # Extract data if provided
        data_dir = Path("/data")
        if data_tarball:
            logger.info("Extracting training data...")
            with tarfile.open(fileobj=io.BytesIO(data_tarball), mode="r:gz") as tar:
                safe_extract(tar, data_dir)

        # Import required modules after adding to path
        from oc20_exp.data.s2ef import S2EFDataIngestion
        from oc20_exp.models import PeriodicSetTransformer
        from oc20_exp.types import (
            AtomicStructure,
            CoordinateSystem,
            CrystalStructure,
            LatticeData,
            LatticeFormat,
            S2EFSample,
            S2EFTarget,
            collate_s2ef_samples,
        )

        # Import training components (we'll inline them since they're in the script)
        sys.path.append(str(project_dir / "scripts"))

        # Define training classes inline (copied from train_s2ef.py)
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

                structure = CrystalStructure(
                    lattice=lattice, atoms=atoms, metadata=item.get("metadata", {})
                )

                return S2EFSample(structure=structure, targets=targets)

        class S2EFModel(torch.nn.Module):
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
                self.energy_head = torch.nn.Sequential(
                    torch.nn.Linear(d_model, dim_feedforward),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(dim_feedforward, 1),
                )

                # Forces prediction head (per atom)
                self.forces_head = torch.nn.Sequential(
                    torch.nn.Linear(d_model, dim_feedforward),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(dim_feedforward, 3),
                )

            def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
                train_loader,
                val_loader=None,
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
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                self.energy_weight = energy_weight
                self.forces_weight = forces_weight

                # Loss functions
                self.energy_loss_fn = torch.nn.MSELoss()
                self.forces_loss_fn = torch.nn.MSELoss()

                logger.info(f"Training on device: {self.device}")

            def compute_loss(self, predictions, targets, mask):
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
                    pred_forces = predictions["forces"]
                    true_forces = targets["forces"]

                    # Ensure mask matches the padded sequence length
                    if mask.size(1) != true_forces.size(1):
                        if mask.size(1) > true_forces.size(1):
                            adjusted_mask = mask[:, : true_forces.size(1)]
                        else:
                            adjusted_mask = torch.zeros(
                                true_forces.size(0),
                                true_forces.size(1),
                                dtype=mask.dtype,
                                device=mask.device,
                            )
                            adjusted_mask[:, : mask.size(1)] = mask
                        forces_mask = adjusted_mask.unsqueeze(-1).expand_as(pred_forces)
                    else:
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

            def train_epoch(self):
                """Train for one epoch."""
                from tqdm import tqdm

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
                                f"{losses.get('energy', 0.0):.4f}" if "energy" in losses else "N/A"
                            ),
                            "forces": (
                                f"{losses.get('forces', 0.0):.4f}" if "forces" in losses else "N/A"
                            ),
                        }
                    )

                # Average losses
                for key in epoch_losses:
                    epoch_losses[key] /= num_batches

                return epoch_losses

            def validate(self):
                """Validate the model."""
                if self.val_loader is None:
                    return {}

                from tqdm import tqdm

                self.model.eval()
                val_losses = {"total": 0.0, "energy": 0.0, "forces": 0.0}
                num_batches = 0

                with torch.no_grad():
                    for batch in tqdm(self.val_loader, desc="Validation"):
                        batch = batch.to_device(self.device)
                        model_input = batch.to_pst_input()

                        targets = {}
                        if batch.targets.energy is not None:
                            targets["energy"] = batch.targets.energy
                        if batch.targets.forces is not None:
                            targets["forces"] = batch.targets.forces

                        predictions = self.model(model_input)
                        losses = self.compute_loss(predictions, targets, model_input["mask"])

                        for key, loss in losses.items():
                            val_losses[key] += loss.item()
                        num_batches += 1

                # Average losses
                for key in val_losses:
                    val_losses[key] /= num_batches

                return val_losses

            def train(self, num_epochs: int, save_dir: Union[str, Path] = "/checkpoints"):
                """Main training loop."""
                import time

                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                history: dict[str, list[Any]] = {"train": [], "val": []}
                best_val_loss = float("inf")

                logger.info(f"Starting training for {num_epochs} epochs...")

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

                    # Log epoch summary
                    logger.info(f"Epoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
                    logger.info(
                        f"Train - Total: {train_losses['total']:.4f}, "
                        f"Energy: {train_losses.get('energy', 0.0):.4f}, "
                        f"Forces: {train_losses.get('forces', 0.0):.4f}"
                    )

                    if val_losses:
                        logger.info(
                            f"Val   - Total: {val_losses['total']:.4f}, "
                            f"Energy: {val_losses.get('energy', 0.0):.4f}, "
                            f"Forces: {val_losses.get('forces', 0.0):.4f}"
                        )

                        # Save best model
                        if val_losses["total"] < best_val_loss:
                            best_val_loss = val_losses["total"]
                            torch.save(self.model.state_dict(), save_dir / "best_model.pt")
                            logger.info("Saved best model")

                    # Save checkpoint every 5 epochs
                    if (epoch + 1) % 5 == 0:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "history": history,
                            },
                            save_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                        )
                        logger.info(f"Saved checkpoint at epoch {epoch + 1}")

                # Save final model and history
                torch.save(self.model.state_dict(), save_dir / "final_model.pt")
                with open(save_dir / "training_history.json", "w") as f:
                    json.dump(history, f, indent=2)

                return history

        # Now run the actual training
        config = training_config
        logger.info(f"Training configuration: {config}")

        # Handle data processing
        processed_data_path = config.get("processed_data")
        if processed_data_path and Path(processed_data_path).exists():
            logger.info(f"Loading preprocessed data from {processed_data_path}")
            dataset = S2EFDataset(processed_data_path)
        else:
            data_dir_path = config.get("data_dir", "/data")
            logger.info(f"Processing raw data from {data_dir_path}")

            # Create temporary directory for processing
            import tempfile

            temp_process_dir = tempfile.mkdtemp(prefix="s2ef_processed_")
            logger.info(f"Using temporary processing directory: {temp_process_dir}")

            # Create data ingestion
            ingestion = S2EFDataIngestion(
                data_dir=data_dir_path,
                output_dir=temp_process_dir,
                max_atoms=config.get("max_atoms", 200),
                max_workers=4,
                max_files=config.get("max_files", None),
            )

            # Process and save data
            processed_file = str(Path(temp_process_dir) / "processed_s2ef_data.pt")
            ingestion.save_to_pytorch("processed_s2ef_data.pt")
            dataset = S2EFDataset(processed_file)

        logger.info(f"Dataset size: {len(dataset)} samples")

        # Create data loaders
        from functools import partial

        from torch.utils.data import DataLoader

        collate_fn_with_max_atoms = partial(
            collate_s2ef_samples, max_atoms=config.get("max_atoms", 200)
        )

        # Train/validation split
        from torch.utils.data import Dataset

        train_dataset: Dataset
        val_split = config.get("val_split", 0.1)
        if val_split > 0:
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=config.get("batch_size", 32),
                shuffle=False,
                collate_fn=collate_fn_with_max_atoms,
            )
            logger.info(f"Train: {train_size}, Val: {val_size}")
        else:
            train_dataset = dataset
            val_loader = None
            logger.info(f"Train: {len(train_dataset)} (no validation)")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=True,
            collate_fn=collate_fn_with_max_atoms,
        )

        # Create model
        model = S2EFModel(
            d_model=config.get("d_model", 256),
            nhead=config.get("nhead", 8),
            num_layers=config.get("num_layers", 4),
            max_atoms=config.get("max_atoms", 200),
        )

        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create trainer and train
        trainer = S2EFTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=config.get("lr", 1e-4),
            device="cuda",
            energy_weight=config.get("energy_weight", 1.0),
            forces_weight=config.get("forces_weight", 100.0),
        )

        # Train model
        history = trainer.train(num_epochs=config.get("epochs", 10))

        logger.info("Training completed!")
        logger.info("Results saved to /checkpoints")

        # Return training summary
        return {
            "status": "completed",
            "epochs": config.get("epochs", 10),
            "dataset_size": len(dataset),
            "model_params": sum(p.numel() for p in model.parameters()),
            "final_train_loss": history["train"][-1]["total"] if history["train"] else None,
            "final_val_loss": history["val"][-1]["total"] if history["val"] else None,
        }


@app.local_entrypoint()
def main(
    data_dir: str = "../s2ef_train_200K/s2ef_train_200K",
    processed_data: str = "",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    max_atoms: int = 200,
    val_split: float = 0.1,
    energy_weight: float = 1.0,
    forces_weight: float = 100.0,
    max_files: Optional[int] = None,
    gpu_type: str = "T4:1",
):
    """
    Deploy S2EF training to Modal GPU.

    Args:
        data_dir: Directory containing S2EF data files
        processed_data: Path to preprocessed data (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        max_atoms: Maximum number of atoms
        val_split: Validation split fraction
        energy_weight: Weight for energy loss
        forces_weight: Weight for forces loss
        max_files: Maximum number of files to process (for testing)
        gpu_type: GPU type (T4:1, L4:1, A100:1, H100:1, etc.)
    """
    import tarfile
    import tempfile

    print(f"Deploying S2EF training to Modal GPU ({gpu_type})")

    # Update GPU configuration
    train_s2ef_on_gpu.gpu = gpu_type

    # Prepare training configuration
    config = {
        "data_dir": data_dir,
        "processed_data": processed_data,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "max_atoms": max_atoms,
        "val_split": val_split,
        "energy_weight": energy_weight,
        "forces_weight": forces_weight,
        "max_files": max_files,
    }

    # Package source code
    print("Packaging source code...")
    project_root = Path(__file__).parent.parent  # Go up from scripts/ to project root

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        with tarfile.open(tmp_file.name, "w:gz") as tar:
            # Add source code
            if (project_root / "src").exists():
                tar.add(project_root / "src", arcname="oc20_exp/src")
            if (project_root / "scripts").exists():
                tar.add(project_root / "scripts", arcname="oc20_exp/scripts")
            if (project_root / "pyproject.toml").exists():
                tar.add(project_root / "pyproject.toml", arcname="oc20_exp/pyproject.toml")

        with open(tmp_file.name, "rb") as f:
            source_tarball = f.read()

    # Package data if needed
    data_tarball = None
    if data_dir and Path(data_dir).exists():
        print(f"Packaging data from {data_dir}...")
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            with tarfile.open(tmp_file.name, "w:gz") as tar:
                tar.add(data_dir, arcname=".")

            with open(tmp_file.name, "rb") as f:
                data_tarball = f.read()
        print(f"Data package size: {len(data_tarball) / 1e6:.1f} MB")

    print(f"Source package size: {len(source_tarball) / 1e6:.1f} MB")
    if data_tarball:
        config["data_dir"] = "/data"

    # Deploy training
    print("Deploying to Modal...")

    try:
        result = train_s2ef_on_gpu.remote(
            training_config=config,
            source_code_tarball=source_tarball,
            data_tarball=data_tarball,
        )

        print("\nTraining Results:")
        print(f"Status: {result['status']}")
        print(f"Epochs: {result['epochs']}")
        print(f"Dataset size: {result['dataset_size']}")
        print(f"Model parameters: {result['model_params']:,}")
        if result["final_train_loss"]:
            print(f"Final train loss: {result['final_train_loss']:.4f}")
        if result["final_val_loss"]:
            print(f"Final validation loss: {result['final_val_loss']:.4f}")

        print("\nTo download checkpoints, run:")
        print("modal volume get s2ef-checkpoints ./modal_checkpoints")

        return result

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    import fire

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    modal_env_detected = any(
        os.environ.get(env_name)
        for env_name in ("MODAL_ENVIRONMENT", "MODAL_APP_ID", "MODAL_FUNCTION_ID")
    )

    if not modal_env_detected:
        logger.warning(
            "This script should be run with 'modal run modal_train_s2ef.py', not directly with python"
        )
        logger.info("Example: modal run modal_train_s2ef.py --processed-data ./data.pt --epochs 20")
        sys.exit("Modal runtime context not detected; aborting.")

    fire.Fire(main)
