"""Configuration management for S2EF training scripts."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml


@dataclass
class ModelConfig:
    """Configuration for PST model architecture."""

    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    max_atoms: int = 200
    dropout: float = 0.1
    max_atomic_number: int = 118
    fourier_features_dim: int = 32
    lattice_hidden_dim: int = 128
    cell_features_dim: int = 32


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    data_dir: str
    processed_data: Optional[str] = None
    val_split: float = 0.1
    batch_size: int = 32
    num_workers: int = 4
    max_atoms: int = 200
    lattice_format: str = "params_6d"  # "params_6d" or "matrix_3x3"


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    lr: float = 1e-4
    epochs: int = 100
    device: str = "auto"
    energy_weight: float = 1.0
    forces_weight: float = 100.0
    save_dir: str = "./checkpoints"
    save_every: int = 10
    grad_clip_norm: Optional[float] = None
    scheduler: Optional[str] = None  # "cosine", "step", etc.
    warmup_epochs: int = 0


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    name: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    seed: int = 42
    notes: str = ""

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)

        with open(config_path) as f:
            if config_path.suffix.lower() == ".yaml" or config_path.suffix.lower() == ".yml":
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

        return cls(
            name=config_dict.get("name", "experiment"),
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            seed=config_dict.get("seed", 42),
            notes=config_dict.get("notes", ""),
        )

    def to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_dict = asdict(self)

        with open(config_path, "w") as f:
            if config_path.suffix.lower() == ".yaml" or config_path.suffix.lower() == ".yml":
                yaml.dump(config_dict, f, indent=2)
            elif config_path.suffix.lower() == ".json":
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def create_default_config(name: str = "default_s2ef") -> ExperimentConfig:
    """Create a default configuration for S2EF training."""
    return ExperimentConfig(
        name=name,
        model=ModelConfig(),
        data=DataConfig(data_dir="s2ef_train_200K/s2ef_train_200K"),
        training=TrainingConfig(),
        seed=42,
        notes="Default S2EF training configuration",
    )


def validate_config(config: ExperimentConfig) -> None:
    """Validate configuration parameters."""
    # Model validation
    assert config.model.d_model > 0, "d_model must be positive"
    assert config.model.nhead > 0, "nhead must be positive"
    assert config.model.d_model % config.model.nhead == 0, "d_model must be divisible by nhead"
    assert config.model.num_layers > 0, "num_layers must be positive"
    assert config.model.max_atoms > 0, "max_atoms must be positive"
    assert 0 <= config.model.dropout <= 1, "dropout must be between 0 and 1"

    # Data validation
    assert Path(config.data.data_dir).exists(), f"Data directory not found: {config.data.data_dir}"
    assert 0 <= config.data.val_split <= 1, "val_split must be between 0 and 1"
    assert config.data.batch_size > 0, "batch_size must be positive"
    assert config.data.lattice_format in [
        "params_6d",
        "matrix_3x3",
    ], "Invalid lattice_format"

    # Training validation
    assert config.training.lr > 0, "learning rate must be positive"
    assert config.training.epochs > 0, "epochs must be positive"
    assert config.training.energy_weight >= 0, "energy_weight must be non-negative"
    assert config.training.forces_weight >= 0, "forces_weight must be non-negative"
    assert config.training.save_every > 0, "save_every must be positive"
