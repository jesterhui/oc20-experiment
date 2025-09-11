# Periodic Set Transformer for OC20 Dataset

This repository contains a **Periodic Set Transformer** implementation for working with crystal structures and the **OC20 (Open Catalyst 2020)** dataset.

## Overview

The Periodic Set Transformer processes crystal structures using:
- **[CELL] token**: Encodes the unit cell (lattice matrix) 
- **Atom tokens**: Combine element embeddings with periodic Fourier features of fractional coordinates
- **Standard Transformer**: Processes the token sequence with attention

### Key Features

-  Handles both 3x3 lattice matrices and 6D parameters (a,b,c,α,β,γ)
-  Periodic Fourier features for fractional coordinates
-  Variable number of atoms with masking
-  OC20 data loading with LMDB and trajectory file support
-  ASE (Atomic Simulation Environment) integration
-  PyTorch native implementation

## Installation

```bash
# Install dependencies
uv sync

# Optional: Install FairChem for full OC20 support
uv pip install fairchem-core

# Optional: Install wandb for experiment tracking  
uv pip install wandb
```

## Quick Start

### 1. Test the Model

```bash
# Run basic model test
uv run python example_usage.py

# Run comprehensive demo with OC20-style data
uv run python demo_oc20_loading.py
```

### 2. Load OC20 Data

#### Option A: Using LMDB Files (Fastest)

```python
from oc20_data_loader import OC20DataModule

# Create data module
data_module = OC20DataModule(
    train_path="path/to/s2ef/200k/train/",  
    val_path="path/to/s2ef/200k/val_id/",
    max_atoms=200,
    batch_size=32,
    data_format="lmdb"
)

# Get data loaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
```

#### Option B: Using ASE Trajectory Files

```python  
from oc20_data_loader import OC20ToPST
from torch.utils.data import DataLoader

dataset = OC20ToPST(
    "path/to/trajectory.traj",
    max_atoms=100, 
    data_format="traj",
    transform_lattice="matrix"  # or "params"
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

### 3. Train the Model

```bash
# Train with synthetic data (for testing)
uv run python train_with_oc20.py --data_format synthetic --epochs 10

# Train with real OC20 data  
uv run python train_with_oc20.py \
    --data_path /path/to/oc20/train.lmdb \
    --data_format lmdb \
    --epochs 50 \
    --batch_size 32 \
    --use_wandb
```

## Data Format

The Periodic Set Transformer expects batched data in this format:

```python
batch = {
    'lattice': torch.Tensor,        # (batch, 3, 3) or (batch, 6)
    'atomic_numbers': torch.Tensor, # (batch, max_atoms) - integer atomic numbers
    'fractional_coords': torch.Tensor, # (batch, max_atoms, 3) - [0,1) coordinates  
    'mask': torch.Tensor,          # (batch, max_atoms) - True for valid atoms
    'metadata': List[Dict]         # Additional structure information
}
```

### Example Data Conversion

```python
# Convert ASE Atoms to PST format
from ase.io import read
from oc20_data_loader import OC20ToPST

atoms = read("structure.cif")
converter = OC20ToPST("dummy", max_atoms=100, data_format="atoms")
pst_data = converter.atoms_to_pst_format(atoms)

# Use with model
from periodic_set_transformer import PeriodicSetTransformer

model = PeriodicSetTransformer(d_model=128, nhead=4, num_layers=2)
output = model(
    lattice=pst_data['lattice'].unsqueeze(0),
    atomic_numbers=pst_data['atomic_numbers'].unsqueeze(0), 
    fractional_coords=pst_data['fractional_coords'].unsqueeze(0),
    mask=pst_data['mask'].unsqueeze(0)
)
```

## OC20 Dataset Information

### 1. Standard Loading Method
- **Primary format**: LMDB files (Lightning Memory-Mapped Database)
- **Library**: FairChem (formerly OCP) 
- **Speed**: Optimized for very fast random access

### 2. Common Libraries Used
- **fairchem-core**: Main library for OC20 models and data loading
- **ASE**: Atomic structure manipulation and I/O
- **LMDB**: Fast database format for PyTorch training
- **pymatgen**: Materials science data analysis

### 3. Data Format Details
- **Storage**: PyTorch Geometric Data objects in LMDB
- **Raw format**: ASE trajectory files (.traj)
- **Content**: Atomic positions, forces, energies, system metadata
- **Splits**: Train (200K/2M/20M/All), Val/Test (id/ood_ads/ood_cat/ood_both)

### 4. Download OC20 Data

```bash
# Using FairChem download script
python -c "from fairchem.core.scripts.download_data import download_data; download_data('s2ef', '200k')"

# Manual download from: https://fair-chem.github.io/core/datasets/oc20.html
```

### 5. Convert OC20 → PST Format

The data loader automatically converts:
- **Cell vectors** → Lattice matrix (3x3) or parameters (6D)
- **Cartesian coordinates** → Fractional coordinates [0,1)  
- **ASE Atoms objects** → Padded tensors with masks
- **Variable structures** → Fixed-size batches

## Model Architecture

```
Input: [lattice, atomic_numbers, fractional_coords, mask]
         →
    [CELL token]     [Atom tokens]
    lattice → MLP    elements + sin/cos(2→→frac_coords) 
         →                →
    Token sequence: [CELL, atom1, atom2, ..., atomN]
         →
    Standard Transformer (multi-head attention)
         →
    Output: [CELL', atom1', atom2', ..., atomN']
```

### Key Components

1. **UnitCellEmbedding**: Encodes lattice matrix via MLP
2. **AtomEmbedding**: Elements + periodic Fourier positional encoding  
3. **PeriodicFourierFeatures**: sin/cos(2→→u) for fractional coordinates
4. **TransformerEncoder**: Standard multi-head attention

## Files

- **`periodic_set_transformer.py`**: Main model implementation
- **`oc20_data_loader.py`**: OC20 data loading utilities  
- **`train_with_oc20.py`**: Training script with OC20 data
- **`demo_oc20_loading.py`**: Comprehensive demo and examples
- **`example_usage.py`**: Basic model usage example

## Contributing & Ticketing

- See `CONTRIBUTING.md` for the lean MCP-focused workflow.
- Use GitHub issues with the Task (MCP) template or Bug template.
- Refer to `docs/labels.md` for the minimal label set and status flow.

## Citation

If you use this code, please cite the relevant papers:
- OC20 dataset: [Open Catalyst 2020 (OC20) Dataset](https://arxiv.org/abs/2010.09990)
- Set Transformers: [Set Transformer](https://arxiv.org/abs/1810.00825)

## Requirements

- Python e 3.12
- PyTorch e 2.8.0  
- ASE e 3.19.1
- einops e 0.6.0
- LMDB e 1.0.0

See `pyproject.toml` for complete dependency list.
