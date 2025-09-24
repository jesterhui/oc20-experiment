## OC20 Experiments — Transformer for Periodic Structures

Lightweight Transformer experiments for periodic crystal structures with OC20 (S2EF). The model ingests a unit cell plus atoms, tokenizes them, and applies standard Transformer encoder layers for downstream tasks (energies/forces and related heads).

## Tokenization & Architecture

- CELL token: MLP over lattice as either 3×3 matrix or 6D parameters (a, b, c, α, β, γ).
- Atom tokens: element embedding + periodic Fourier features of fractional coordinates.
- Sequence: [CELL, atom1, …, atomN] with padding + mask for variable atom counts.
- Encoder: standard multi‑head attention Transformer encoder layers over the sequence.
- Heads: pooled representations (often CELL) or token‑level reads for predictions.

## Code Structure

```
oc20_exp/
├── scripts/
│   └── train_s2ef.py                    # training entrypoint (S2EF on OC20-style data)
├── src/
│   └── oc20_exp/
│       ├── models/
│       │   └── periodic_set_transformer.py    # model implementation
│       ├── data/
│       │   └── s2ef/                    # ingestion & pipeline utilities
│       │       ├── pipeline.py
│       │       ├── convert.py
│       │       ├── io.py
│       │       ├── persist.py
│       │       └── types.py
│       └── types/
│           └── data_structures.py           # typed structures, batching, masks
└── tests/                               # minimal tests where present
```

## Install

```bash
uv sync
```

## Quick Run

```bash
# Tiny smoke run on a small subset
uv run python scripts/train_s2ef.py --test --data-dir /path/to/s2ef_subset
```

## Train

```bash
uv run python scripts/train_s2ef.py \
  --data-dir /path/to/s2ef \
  --epochs 50 \
  --batch-size 32 \
  --save-dir ./checkpoints
```

## Data

- Use OC20 S2EF LMDBs or ASE trajectories; see OC20 docs for downloads and splits.
- Keep large datasets outside the repo; pass paths via CLI flags.

## Requirements

- Python 3.9.x (see `pyproject.toml` for exact deps)
