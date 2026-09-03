"""
Data Ingestion Module
=====================
Downloads the UCI Diabetes 130-US Hospitals dataset and loads it into
pandas DataFrames. Also provides SQLite ingestion for SQL demo queries.
"""
import sqlite3
from pathlib import Path

import pandas as pd

from src.utils.config import (
    RAW_DATA_DIR,
    RAW_CSV_NAME,
    SQLITE_DB_PATH,
    MISSING_MARKER,
)
from src.utils.helpers import get_logger, timer

logger = get_logger(__name__)


@timer
def download_dataset(force: bool = False) -> Path:
    """
    Download the diabetes dataset from UCI repository using ucimlrepo.
    """
    csv_path = RAW_DATA_DIR / RAW_CSV_NAME

    if csv_path.exists() and not force:
        logger.info(f"Dataset already exists at {csv_path}")
        return csv_path

    logger.info("Downloading dataset using ucimlrepo...")

    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=296)

    # Combine features and targets into one DataFrame
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    # Save to CSV
    df.to_csv(csv_path, index=False)
    logger.info(f"Dataset saved to {csv_path} ({len(df):,} rows)")

    return csv_path


@timer
def load_raw_data(filepath: Path = None) -> pd.DataFrame:
    """
    Load the raw diabetes dataset CSV.
    """
    if filepath is None:
        filepath = RAW_DATA_DIR / RAW_CSV_NAME

    if not filepath.exists():
        logger.info("Raw data not found. Downloading...")
        filepath = download_dataset()

    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath, na_values=[MISSING_MARKER], low_memory=False)

    logger.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


@timer
def load_to_sqlite(df: pd.DataFrame, table_name: str = "diabetes_encounters") -> Path:
    """
    Load a DataFrame into SQLite for SQL demo queries.
    """
    logger.info(f"Loading data into SQLite at {SQLITE_DB_PATH}")

    conn = sqlite3.connect(SQLITE_DB_PATH)
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table_name}", conn).iloc[0, 0]
    logger.info(f"Loaded {count:,} rows into '{table_name}' table")

    conn.close()
    return SQLITE_DB_PATH


def get_sql_connection():
    """Get a connection to the SQLite database."""
    if not SQLITE_DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {SQLITE_DB_PATH}. "
            "Run the data pipeline first."
        )
    return sqlite3.connect(SQLITE_DB_PATH)


if __name__ == "__main__":
    csv_path = download_dataset()
    df = load_raw_data(csv_path)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")