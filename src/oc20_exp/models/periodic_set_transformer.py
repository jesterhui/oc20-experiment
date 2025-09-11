import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)


class PeriodicSetTransformer(nn.Module):
    """
    Periodic Set Transformer for crystal structures.

    Inputs per crystal:
    - lattice: 6D vector (a, b, c, alpha_deg, beta_deg, gamma_deg)
    - atomic_numbers: integer atomic numbers per site
    - fractional_coords: fractional coordinates in [0, 1)

    Tokens:
    - [CELL]: MLP over unit cell (6 -> cell_features_dim -> d_model)
    - atoms: element embedding + positional embedding (sin/cos(2π·u) -> fourier_features_dim -> d_model)

    Output: Transformer-encoded tokens [CELL, atom1, …, atomN].
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_atomic_number: int = 118,
        fourier_features_dim: int = 32,
        lattice_hidden_dim: int = 128,
        cell_features_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.fourier_features_dim = fourier_features_dim
        self.max_atomic_number = max_atomic_number

        # Unit cell embedding module for [CELL] token
        self.unitcell = UnitCellEmbedding(
            hidden_dim=lattice_hidden_dim,
            output_dim=cell_features_dim,
        )
        self.cell_proj = nn.Linear(cell_features_dim, d_model)

        # Atom embedding module (element + positional)
        self.atom_embedding = AtomEmbedding(
            max_atomic_number=max_atomic_number,
            d_model=d_model,
            pos_dim=fourier_features_dim,
        )

        # Standard Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Special token for CELL
        self.cell_token_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.atom_token_type = nn.Parameter(torch.zeros(1, 1, d_model))

        logger.debug(
            "Initialized PST d_model=%d nhead=%d layers=%d ff=%d pos_dim=%d cell_dim=%d",
            d_model,
            nhead,
            num_layers,
            dim_feedforward,
            fourier_features_dim,
            cell_features_dim,
        )

    def forward(
        self,
        lattice: torch.Tensor,  # (batch, 6) or (batch, 3, 3)
        atomic_numbers: torch.Tensor,  # (batch, max_atoms)
        fractional_coords: torch.Tensor,  # (batch, max_atoms, 3) in [0,1)
        mask: Optional[torch.Tensor] = None,  # (batch, max_atoms), True/1 = valid
    ) -> torch.Tensor:
        """Forward pass. Returns (batch, 1+max_atoms, d_model)."""
        batch_size, max_atoms = atomic_numbers.shape
        device = lattice.device

        self._validate_inputs(lattice, atomic_numbers, fractional_coords, mask)

        # Convert lattice to 6D format if needed
        lattice_6d = self._ensure_lattice_6d(lattice)
        
        cell_token = self.create_cell_token(lattice_6d)

        atom_tokens = self.create_atom_tokens(atomic_numbers, fractional_coords, mask)

        all_tokens = torch.cat([cell_token, atom_tokens], dim=1)

        transformer_mask = self.create_transformer_mask(mask, device)
        output = self.transformer(all_tokens, src_key_padding_mask=transformer_mask)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Forward shapes: lattice=%s atoms=%s coords=%s mask=%s tokens=%s out=%s",
                tuple(lattice.shape),
                tuple(atomic_numbers.shape),
                tuple(fractional_coords.shape),
                None if mask is None else tuple(mask.shape),
                tuple(all_tokens.shape),
                tuple(output.shape),
            )

        return output

    def create_cell_token(self, lattice: torch.Tensor) -> torch.Tensor:
        """Create [CELL] token from 6D lattice parameters."""
        cell_embedding = self.unitcell(lattice)  # (batch, cell_features_dim)
        cell_embedding = self.cell_proj(cell_embedding)  # (batch, d_model)

        cell_token = rearrange(cell_embedding, "b d -> b 1 d") + self.cell_token_type

        return cell_token

    def create_atom_tokens(
        self,
        atomic_numbers: torch.Tensor,
        fractional_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Create atom tokens from atomic numbers and fractional coordinates."""
        atom_tokens = self.atom_embedding(atomic_numbers, fractional_coords)
        atom_tokens = atom_tokens + self.atom_token_type

        if mask is not None:
            mask_e = rearrange(mask, "b n -> b n 1").to(atom_tokens.dtype)
            atom_tokens = atom_tokens * mask_e

        return atom_tokens

    def create_transformer_mask(
        self, atom_mask: Optional[torch.Tensor], device: torch.device
    ) -> Optional[torch.Tensor]:
        """Create padding mask for transformer."""
        if atom_mask is None:
            return None

        batch_size, max_atoms = atom_mask.shape

        cell_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cell_mask, atom_mask.bool()], dim=1)

        transformer_mask = ~full_mask

        return transformer_mask

    def _ensure_lattice_6d(self, lattice: torch.Tensor) -> torch.Tensor:
        """Convert lattice matrix to 6D parameters if needed."""
        if lattice.dim() == 2 and lattice.shape[-1] == 6:
            # Already in 6D format
            return lattice
        elif lattice.dim() == 3 and lattice.shape[-2:] == (3, 3):
            # Convert 3x3 matrix to 6D parameters
            return self._matrix_to_params(lattice)
        else:
            raise ValueError(
                "lattice must be (batch, 6) or (batch, 3, 3), got shape: " + str(lattice.shape)
            )
    
    def _matrix_to_params(self, matrix: torch.Tensor) -> torch.Tensor:
        """Convert 3x3 lattice matrix to 6D parameters (a,b,c,α,β,γ)."""
        # Extract lattice vectors
        a_vec = matrix[..., 0, :]  # (batch, 3)
        b_vec = matrix[..., 1, :]  # (batch, 3)
        c_vec = matrix[..., 2, :]  # (batch, 3)
        
        # Calculate lengths
        a = torch.norm(a_vec, dim=-1)  # (batch,)
        b = torch.norm(b_vec, dim=-1)  # (batch,)
        c = torch.norm(c_vec, dim=-1)  # (batch,)
        
        # Calculate angles in degrees
        cos_alpha = torch.sum(b_vec * c_vec, dim=-1) / (b * c)
        cos_beta = torch.sum(a_vec * c_vec, dim=-1) / (a * c)
        cos_gamma = torch.sum(a_vec * b_vec, dim=-1) / (a * b)
        
        # Clamp to avoid numerical issues with acos
        cos_alpha = torch.clamp(cos_alpha, -1.0 + 1e-7, 1.0 - 1e-7)
        cos_beta = torch.clamp(cos_beta, -1.0 + 1e-7, 1.0 - 1e-7)
        cos_gamma = torch.clamp(cos_gamma, -1.0 + 1e-7, 1.0 - 1e-7)
        
        alpha = torch.acos(cos_alpha) * 180.0 / math.pi
        beta = torch.acos(cos_beta) * 180.0 / math.pi
        gamma = torch.acos(cos_gamma) * 180.0 / math.pi
        
        return torch.stack([a, b, c, alpha, beta, gamma], dim=-1)

    def _validate_inputs(
        self,
        lattice: torch.Tensor,
        atomic_numbers: torch.Tensor,
        fractional_coords: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> None:
        # Updated validation to handle both formats
        if lattice.dim() == 2 and lattice.shape[-1] == 6:
            # 6D format is valid
            pass
        elif lattice.dim() == 3 and lattice.shape[-2:] == (3, 3):
            # 3x3 matrix format is valid
            pass
        else:
            raise ValueError(
                "lattice must be (batch, 6) = (a,b,c,alpha_deg,beta_deg,gamma_deg) or (batch, 3, 3) matrix"
            )
        if fractional_coords.shape[-1] != 3:
            raise ValueError("fractional_coords must have last dim 3")
        if mask is not None and mask.shape != atomic_numbers.shape:
            raise ValueError(
                "mask shape must match atomic_numbers shape (batch, max_atoms)"
            )

        max_in = int(atomic_numbers.max().item())
        min_in = int(atomic_numbers.min().item())
        if min_in < 0 or max_in > self.max_atomic_number:
            raise ValueError(
                f"atomic_numbers out of range [0,{self.max_atomic_number}] (min={min_in}, max={max_in})"
            )

        if torch.any(fractional_coords < 0.0) or torch.any(fractional_coords >= 1.0):
            n_bad = int(
                ((fractional_coords < 0.0) | (fractional_coords >= 1.0)).sum().item()
            )
            logger.warning(
                "%d fractional coords outside [0,1); downstream may assume wrapping.",
                n_bad,
            )


class UnitCellEmbedding(nn.Module):
    """Encode unit cell (a,b,c,alpha,beta,gamma) via small MLP."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, lattice: torch.Tensor) -> torch.Tensor:
        if lattice.dim() != 2 or lattice.shape[-1] != 6:
            raise ValueError(
                "lattice must be shape (batch, 6) = (a,b,c,alpha_deg,beta_deg,gamma_deg)"
            )
        return self.mlp(lattice)


class PositionalEmbedding(nn.Module):
    """Sin/cos(2π·u) positional embedding with optional projection to latent size."""

    def __init__(self, input_dim: int = 3, output_dim: int = 32):
        super().__init__()
        base_dim = 2 * input_dim  # sin and cos per coordinate
        if output_dim < base_dim:
            raise ValueError(f"output_dim must be >= {base_dim}, got {output_dim}")
        self.output_dim = output_dim
        self.base_dim = base_dim
        self.proj = None if output_dim == base_dim else nn.Linear(base_dim, output_dim)

    def forward(self, fractional_coords: torch.Tensor) -> torch.Tensor:
        phases = 2 * math.pi * fractional_coords
        sin_feat = torch.sin(phases)
        cos_feat = torch.cos(phases)
        base = torch.cat([sin_feat, cos_feat], dim=-1)
        if self.proj is None:
            return base
        return self.proj(base)


class AtomEmbedding(nn.Module):
    """Element embedding + positional embedding to form atom tokens."""

    def __init__(self, max_atomic_number: int, d_model: int, pos_dim: int = 32):
        super().__init__()
        self.element_embedding = nn.Embedding(max_atomic_number + 1, d_model)
        self.positional = PositionalEmbedding(input_dim=3, output_dim=pos_dim)
        self.pos_proj = nn.Linear(pos_dim, d_model)

    def forward(
        self,
        atomic_numbers: torch.Tensor,  # (batch, atoms)
        fractional_coords: torch.Tensor,  # (batch, atoms, 3)
    ) -> torch.Tensor:
        elem = self.element_embedding(atomic_numbers)
        pos = self.positional(fractional_coords)
        pos = self.pos_proj(pos)
        return elem + pos
