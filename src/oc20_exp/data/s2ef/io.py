import lzma
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from ase import Atoms
    from ase.io import read
except ImportError as e:
    raise ImportError("ASE is required. Install with: pip install ase") from e

if TYPE_CHECKING:
    from .types import S2EFMetadata


def find_data_files(data_dir: Path, logger) -> list[tuple[Path, Path]]:
    """Find all matching .extxyz.xz and .txt.xz file pairs."""
    extxyz_files = list(data_dir.glob("*.extxyz.xz"))
    data_pairs: list[tuple[Path, Path]] = []
    for extxyz_file in sorted(extxyz_files):
        base_name = extxyz_file.stem.replace(".extxyz", "")
        txt_file = data_dir / f"{base_name}.txt.xz"
        if txt_file.exists():
            data_pairs.append((extxyz_file, txt_file))
        else:
            logger.warning(f"No metadata file found for {extxyz_file}")
    return data_pairs


def load_metadata(txt_file: Path) -> list["S2EFMetadata"]:
    """Load metadata from compressed text file (.txt.xz)."""
    from .types import S2EFMetadata

    metadata_list: list[S2EFMetadata] = []
    file_index = int(txt_file.stem.replace(".txt", ""))

    with lzma.open(txt_file, "rt") as f:
        for structure_index, line in enumerate(f):
            parts = line.strip().split(",")
            if len(parts) == 3:
                system_id, frame_number, reference_energy = parts
                # Extract numeric part from frame_number (e.g., "frame206" -> 206)
                frame_num = (
                    int(frame_number.replace("frame", ""))
                    if frame_number.startswith("frame")
                    else int(frame_number)
                )

                metadata_list.append(
                    S2EFMetadata(
                        system_id=system_id,
                        frame_number=frame_num,
                        reference_energy=float(reference_energy),
                        file_index=file_index,
                        structure_index=structure_index,
                    )
                )
    return metadata_list


def load_structures(extxyz_file: Path, temp_dir: Path, logger) -> list[Atoms]:
    """Load ASE Atoms list from a compressed .extxyz.xz, via temp decompression."""
    temp_file = temp_dir / f"temp_{extxyz_file.stem}"

    with lzma.open(extxyz_file, "rb") as compressed, open(temp_file, "wb") as decompressed:
        decompressed.write(compressed.read())

    try:
        structures = read(str(temp_file), ":")
        logger.info(f"Loaded {len(structures)} structures from {extxyz_file.name}")
        return structures
    finally:
        if temp_file.exists():
            temp_file.unlink()
