# preprocess.py
"""
Robust preprocessing for the AML Detection Pipeline.

Responsibilities:
- Validate raw input files
- Convert CSV → Parquet with schema enforcement
- Create optional sample datasets
- Merge alerts and transactions WITHOUT feature aggregation
- Persist clean base dataset for downstream feature engineering
- Produce audit-ready manifest metadata
"""

from pathlib import Path
import argparse
import json
import logging
import hashlib
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ----------------------------
# Logging configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------------------
# Utility functions
# ----------------------------
def file_checksum(path: Path) -> str:
    """Compute SHA256 checksum for auditability."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"File is empty: {path}")


# ----------------------------
# CSV → Sample Parquet conversion (fast iteration)
# ----------------------------
def csv_to_sample_parquet(
    csv_path: Path,
    out_path: Path,
    sample_size: int,
):
    """
    Stream CSV and save first N rows to Parquet (sample only, for fast iteration).
    """
    logger.info(f"Creating sample from {csv_path.name} → {out_path.name} (first {sample_size:,} rows)")
    tmp_path = out_path.with_suffix(".tmp.parquet")

    df = pd.read_csv(csv_path, nrows=sample_size)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp_path, compression="snappy")
    tmp_path.rename(out_path)

    logger.info(f"Sample saved: {out_path.name} ({len(df):,} rows)")
    return len(df), table.schema


# ----------------------------
# Merge & label creation (NO aggregates, samples only)
# ----------------------------
def load_and_merge(transactions_pq: Path, alerts_pq: Path) -> pd.DataFrame:
    logger.info(f"Loading sample parquets for merge...")
    tx = pd.read_parquet(transactions_pq)
    al = pd.read_parquet(alerts_pq)

    logger.info(f"Merging: {tx.shape[0]:,} transactions × {al.shape[0]:,} alerts on AlertID")
    merged = tx.merge(al, on="AlertID", how="inner")

    if merged.empty:
        raise ValueError("Merged dataset is empty — check AlertID keys.")

    logger.info(f"Merged shape: {merged.shape}")

    merged["Label"] = merged["Outcome"].map({"Report": 1, "Dismiss": 0})

    if merged["Label"].isna().any():
        raise ValueError("Unmapped Outcome values detected.")

    logger.info(f"Label mapping complete. Label distribution:\n{merged['Label'].value_counts()}")
    return merged


# ----------------------------
# Main
# ----------------------------
def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tx_csv = data_dir / "synthetic_transactions.csv"
    al_csv = data_dir / "synthetic_alerts.csv"

    validate_file(tx_csv)
    validate_file(al_csv)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_files": {},
        "rows": {},
        "schema": {},
        "note": "Sample-only preprocessing for fast iteration. Samples use first N rows from CSV.",
    }

    # Create samples from CSV (no intermediate full parquets)
    tx_rows, tx_schema = csv_to_sample_parquet(
        tx_csv,
        out_dir / "synthetic_transactions_sample.parquet",
        args.sample_size,
    )
    al_rows, al_schema = csv_to_sample_parquet(
        al_csv,
        out_dir / "synthetic_alerts_sample.parquet",
        args.sample_size,
    )

    manifest["rows"]["transactions_sample"] = tx_rows
    manifest["rows"]["alerts_sample"] = al_rows
    manifest["schema"]["transactions_sample"] = str(tx_schema)
    manifest["schema"]["alerts_sample"] = str(al_schema)

    manifest["source_files"]["transactions"] = {
        "path": str(tx_csv),
        "checksum": file_checksum(tx_csv),
    }
    manifest["source_files"]["alerts"] = {
        "path": str(al_csv),
        "checksum": file_checksum(al_csv),
    }

    # Merge samples without feature leakage
    merged = load_and_merge(
        out_dir / "synthetic_transactions_sample.parquet",
        out_dir / "synthetic_alerts_sample.parquet",
    )

    merged.to_parquet(out_dir / "merged_base_sample.parquet", index=False)
    logger.info(f"Merged base dataset saved ({merged.shape[0]:,} rows)")

    # Save manifest
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Preprocessing complete. Ready for feature engineering.")


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Pipeline: CSV → Sample Parquet conversion")
    parser.add_argument("--data-dir", default="data", help="Directory with synthetic_*.csv files")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory for parquets & manifest")
    parser.add_argument("--sample-size", type=int, default=100_000, help="Number of rows to sample from each CSV")
    args = parser.parse_args()

    main(args)
    logger.info("Preprocessing complete. Ready for feature engineering.")