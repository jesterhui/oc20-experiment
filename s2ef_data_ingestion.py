"""
S2EF Data Ingestion Pipeline for OC20 Dataset (shim)

This module preserves the original import path and CLI entry by delegating
to the refactored implementation under `s2ef_ingestion/`.
"""

from oc20_exp.data.s2ef import S2EFDataIngestion  # re-export for backward compatibility


def main():
    """Example usage of S2EF data ingestion pipeline."""

    data_dir = "s2ef_train_200K/s2ef_train_200K"
    output_dir = "processed_s2ef_data"

    ingestion = S2EFDataIngestion(
        data_dir=data_dir,
        output_dir=output_dir,
        max_atoms=200,
        max_workers=4,
        chunk_size=1000,
        transform_lattice="matrix",
    )

    print("Starting S2EF data ingestion...")
    ingestion.save_to_pytorch("s2ef_train_200k.pt")
    # Optionally:
    # ingestion.save_to_lmdb("s2ef_train_200k.lmdb")
    # ingestion.save_to_hdf5("s2ef_train_200k.h5")
    print("Data ingestion completed!")


if __name__ == "__main__":
    main()
