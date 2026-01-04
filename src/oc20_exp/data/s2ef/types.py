from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

try:
    from ase import Atoms
except ImportError as e:
    raise ImportError("ASE is required. Install with: pip install ase") from e

if TYPE_CHECKING:
    import numpy as np


@dataclass
class S2EFMetadata:
    """Metadata for S2EF structures."""

    system_id: str
    frame_number: int
    reference_energy: float
    file_index: int
    structure_index: int


@dataclass
class S2EFStructure:
    """Complete S2EF structure with metadata."""

    atoms: Atoms
    metadata: S2EFMetadata
    energy: Optional[float] = None
    forces: Optional["np.ndarray"] = None
