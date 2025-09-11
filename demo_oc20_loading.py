"""
Demo script showing OC20 data loading for Periodic Set Transformer

This script demonstrates:
1. Creating synthetic crystal structures in OC20-like format
2. Converting them to PST format
3. Running inference with the Periodic Set Transformer
4. Showing how to handle real OC20 data when available

Run with: python demo_oc20_loading.py
"""

import torch
import numpy as np
import logging
from pathlib import Path
import warnings

from periodic_set_transformer import PeriodicSetTransformer
from oc20_data_loader import OC20ToPST, collate_pst_batch

try:
    import ase
    from ase import Atoms
    from ase.build import bulk, surface, add_adsorbate
    from ase.io import write

    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    warnings.warn("ASE not installed. Limited functionality.")


def create_catalyst_like_structure():
    """
    Create a catalyst-like structure similar to what's in OC20.
    This mimics adsorbate + catalyst surface systems.
    """
    if not HAS_ASE:
        print("ASE required for creating realistic structures")
        return None

    print("Creating catalyst-like structures (mimicking OC20)...")

    structures = []

    # 1. Cu(111) surface with CO adsorbate
    cu_surface = surface("Cu", (1, 1, 1), 4, vacuum=10.0)
    cu_surface = cu_surface.repeat((2, 2, 1))  # Make it bigger

    # Add CO molecule on top
    from ase import Atom

    co_molecule = Atoms([Atom("C", (0, 0, 5)), Atom("O", (0, 0, 6.1))])
    cu_surface.extend(co_molecule)

    # Set periodic boundary conditions
    cu_surface.set_pbc([True, True, False])  # Slab geometry

    structures.append(("Cu111_CO", cu_surface))

    # 2. Pt(100) surface with H adsorbate
    pt_surface = surface("Pt", (1, 0, 0), 3, vacuum=8.0)
    pt_surface = pt_surface.repeat((2, 2, 1))

    # Add H atom
    h_atom = Atom(
        "H",
        (
            pt_surface.positions[-1, 0],
            pt_surface.positions[-1, 1],
            pt_surface.positions[-1, 2] + 1.5,
        ),
    )
    pt_surface.append(h_atom)
    pt_surface.set_pbc([True, True, False])

    structures.append(("Pt100_H", pt_surface))

    # 3. Simple bulk crystal
    au_bulk = bulk("Au", "fcc", a=4.08)
    au_bulk = au_bulk.repeat((2, 2, 2))
    structures.append(("Au_bulk", au_bulk))

    return structures


def demo_synthetic_data():
    """Demo with simple synthetic crystal data."""
    print("\n" + "=" * 60)
    print("DEMO 1: Synthetic Crystal Structures")
    print("=" * 60)

    # Create simple synthetic structures
    synthetic_structures = []

    for i in range(3):
        # Simple cubic structures with random elements
        a = 5.0 + i * 0.5
        cell = np.eye(3) * a

        # Random atoms
        n_atoms = 8 + i * 4
        positions = np.random.uniform(0, a, (n_atoms, 3))
        elements = np.random.choice([6, 8, 26, 29], n_atoms)  # C, O, Fe, Cu

        if HAS_ASE:
            atoms = Atoms(numbers=elements, positions=positions, cell=cell, pbc=True)
            synthetic_structures.append((f"synthetic_{i}", atoms))
        else:
            print("ASE not available, creating basic tensor data...")
            # Create basic data without ASE
            lattice = torch.tensor([a, a, a, 90.0, 90.0, 90.0], dtype=torch.float32)
            fractional_coords = positions / a  # Convert to fractional

            # Pad to max_atoms
            max_atoms = 20
            padded_numbers = np.zeros(max_atoms, dtype=np.int64)
            padded_coords = np.zeros((max_atoms, 3), dtype=np.float32)
            mask = np.zeros(max_atoms, dtype=bool)

            padded_numbers[:n_atoms] = elements
            padded_coords[:n_atoms] = fractional_coords
            mask[:n_atoms] = True

            sample = {
                "lattice": lattice,
                "atomic_numbers": torch.tensor(padded_numbers, dtype=torch.long),
                "fractional_coords": torch.tensor(padded_coords, dtype=torch.float32),
                "mask": torch.tensor(mask, dtype=torch.bool),
                "metadata": {"n_atoms": n_atoms, "formula": f"synthetic_{i}"},
            }

            return [sample]  # Return early for non-ASE case

    return synthetic_structures


def demo_ase_conversion(structures):
    """Demo converting ASE structures to PST format."""
    print("\n" + "=" * 60)
    print("DEMO 2: ASE to PST Format Conversion")
    print("=" * 60)

    if not HAS_ASE:
        print("ASE not available - skipping ASE conversion demo")
        return []

    # Create dataset converter
    converter = OC20ToPST(
        "dummy", max_atoms=100, data_format="atoms", transform_lattice="matrix"
    )

    pst_samples = []

    for name, atoms in structures:
        print(f"\nProcessing {name}:")
        print(f"  Formula: {atoms.get_chemical_formula()}")
        print(f"  Number of atoms: {len(atoms)}")
        print(f"  Cell: {atoms.get_cell().cellpar()}")

        # Convert to PST format
        pst_data = converter.atoms_to_pst_format(atoms)

        print(f"  PST lattice shape: {pst_data['lattice'].shape}")
        print(f"  PST atomic numbers shape: {pst_data['atomic_numbers'].shape}")
        print(f"  PST fractional coords shape: {pst_data['fractional_coords'].shape}")
        print(f"  Valid atoms: {pst_data['mask'].sum().item()}")

        pst_samples.append(pst_data)

    return pst_samples


def demo_transformer_inference(pst_samples):
    """Demo running PST inference on the samples."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("DEMO 3: Periodic Set Transformer Inference")

    if not pst_samples:
        print("No PST samples available")
        return

    # Create model
    model = PeriodicSetTransformer(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        fourier_features_dim=32,  # latent positional features
    )

    logger.info(
        "Created PST model with %s parameters",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    # Create batch from samples
    if len(pst_samples) > 1:
        batch = collate_pst_batch(pst_samples[:2])  # Use first 2 samples
    else:
        # Single sample to batch
        sample = pst_samples[0]
        batch = {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]
            for k, v in sample.items()
        }

    logger.info("Batch shapes:")
    for key in ["lattice", "atomic_numbers", "fractional_coords", "mask"]:
        logger.info("  %s: %s", key, tuple(batch[key].shape))

    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(
            lattice=batch["lattice"],
            atomic_numbers=batch["atomic_numbers"],
            fractional_coords=batch["fractional_coords"],
            mask=batch["mask"],
        )

    print(f"\nTransformer output shape: {output.shape}")
    print(f"  [CELL] token: {output[:, 0, :].shape}")
    print(f"  Atom tokens: {output[:, 1:, :].shape}")

    # Analyze attention to [CELL] vs atoms
    cell_norms = torch.norm(output[:, 0, :], dim=-1)
    atom_norms = torch.norm(output[:, 1:, :], dim=-1).mean(dim=-1)

    for i in range(output.shape[0]):
        print(f"\nSample {i}:")
        print(f"  CELL token norm: {cell_norms[i]:.3f}")
        print(f"  Average atom token norm: {atom_norms[i]:.3f}")


def demo_real_oc20_loading():
    """Show how to load real OC20 data (when available)."""
    print("\n" + "=" * 60)
    print("DEMO 4: Real OC20 Data Loading Instructions")
    print("=" * 60)

    print(
        """
To use real OC20 data:

1. Install FairChem:
   pip install fairchem-core

2. Download OC20 data:
   # Small dataset for testing
   python -c "from fairchem.core.scripts.download_data import download_data; download_data('s2ef', '200k')"
   
   # Or manually from: https://fair-chem.github.io/core/datasets/oc20.html

3. Use the data loader:
   from oc20_data_loader import OC20DataModule
   
   data_module = OC20DataModule(
       train_path="path/to/s2ef/200k/train/",
       val_path="path/to/s2ef/200k/val_id/",
       max_atoms=200,
       batch_size=32,
       data_format="lmdb"
   )
   
   train_loader = data_module.train_dataloader()

4. The data will be automatically converted to PST format:
   - lattice: (batch, 3, 3) or (batch, 6) 
   - atomic_numbers: (batch, max_atoms)
   - fractional_coords: (batch, max_atoms, 3)
   - mask: (batch, max_atoms) boolean

5. Alternative: Load from trajectory files:
   dataset = OC20ToPST("path/to/file.traj", data_format="traj")
"""
    )


def main():
    """Main demo function."""
    print("OC20 Data Loading Demo for Periodic Set Transformer")
    print("This demo shows how to work with crystal structures and OC20 data")

    # Demo 1: Create synthetic or realistic structures
    if HAS_ASE:
        structures = create_catalyst_like_structure()
        if not structures:
            structures = demo_synthetic_data()
    else:
        structures = demo_synthetic_data()

    # Demo 2: Convert to PST format
    if HAS_ASE and isinstance(structures[0], tuple):
        pst_samples = demo_ase_conversion(structures)
    else:
        pst_samples = structures  # Already in PST format

    # Demo 3: Run transformer inference
    demo_transformer_inference(pst_samples)

    # Demo 4: Instructions for real data
    demo_real_oc20_loading()

    print("\n" + "=" * 60)
    print("Demo completed! Key takeaways:")
    print("- OC20 data can be loaded via LMDB (fastest) or trajectory files")
    print("- ASE provides tools for crystal structure manipulation")
    print("- The data loader automatically converts to PST format")
    print("- Your PST model works with both lattice matrices and parameters")
    print("- Fractional coordinates are handled with periodic boundary conditions")
    print("=" * 60)


if __name__ == "__main__":
    main()
