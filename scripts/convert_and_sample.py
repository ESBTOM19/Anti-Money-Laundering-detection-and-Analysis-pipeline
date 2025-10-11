import pandas as pd
from pathlib import Path
# Create directories if they don't exist
data_dir = Path("data")
processed_dir = data_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# File paths
transactions_csv = data_dir / "synthetic_transactions.csv"
alerts_csv = data_dir / "synthetic_alerts.csv"

# Function to convert CSV → Parquet and create a smaller sample
def convert_and_sample(csv_path, name, sample_size=100_000):
    print(f"Processing {csv_path.name} ...")

    # Read only first `sample_size` rows for sample
    df_sample = pd.read_csv(csv_path, nrows=sample_size)
    df_sample.to_parquet(processed_dir / f"{name}_sample.parquet", index=False, compression="snappy")

    # Convert full file to parquet (efficient)
    print(f"Converting full {csv_path.name} to Parquet (this may take a while)...")
    chunks = pd.read_csv(csv_path, chunksize=200_000)
    for i, chunk in enumerate(chunks):
        chunk.to_parquet(processed_dir / f"{name}_part{i}.parquet", index=False, compression="snappy")

    print(f"Finished processing {csv_path.name}")

# Run conversions
convert_and_sample(transactions_csv, "synthetic_transactions")
convert_and_sample(alerts_csv, "synthetic_alerts")

print("\nConversion complete! Parquet files are saved in 'data/processed/'")
