"""
Data Cleaning Module
====================
Cleans the raw diabetes dataset: handles missing values, removes
duplicates, maps ICD-9 codes, encodes the target variable, and
drops uninformative columns.
"""
import re
import numpy as np
import pandas as pd

from src.utils.config import (
    COLUMNS_TO_DROP,
    TARGET_COL,
    TARGET_MAPPING,
    ICD9_GROUPS,
    PROCESSED_DATA_DIR,
)
from src.utils.helpers import get_logger, timer, print_dataframe_summary

logger = get_logger(__name__)


def _map_icd9_code(code) -> str:
    """
    Map an ICD-9 code to its top-level category group.

    Parameters
    ----------
    code : str or float
        ICD-9 diagnosis code.

    Returns
    -------
    str
        Category name (e.g., 'Circulatory', 'Endocrine').
    """
    if pd.isna(code):
        return "Unknown"

    code = str(code).strip()

    # Handle E-codes (external causes of injury)
    if code.startswith("E"):
        return "Injury"

    # Handle V-codes (supplementary classification)
    if code.startswith("V"):
        return "Supplementary"

    # Try to extract numeric part
    try:
        numeric = float(re.match(r"(\d+\.?\d*)", code).group(1))
    except (AttributeError, ValueError):
        return "Unknown"

    # Map to ICD-9 group
    for group_name, (low, high) in ICD9_GROUPS.items():
        if low <= numeric <= high:
            return group_name

    return "Other"


def _map_age_to_numeric(age_str: str) -> int:
    """
    Convert age bracket string to numeric midpoint.
    e.g., '[50-60)' -> 55
    """
    if pd.isna(age_str):
        return np.nan
    match = re.match(r"\[(\d+)-(\d+)\)", str(age_str))
    if match:
        return (int(match.group(1)) + int(match.group(2))) // 2
    return np.nan


@timer
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for the diabetes dataset.

    Steps:
    1. Drop uninformative columns
    2. Remove duplicate patient encounters (keep first)
    3. Handle missing values
    4. Encode target variable (binary)
    5. Map ICD-9 diagnosis codes to groups
    6. Clean categorical variables
    7. Remove deceased/hospice patients (discharge_disposition_id in [11, 13, 14, 19, 20, 21])

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for feature engineering.
    """
    logger.info("Starting data cleaning pipeline...")
    df = df.copy()
    initial_rows = len(df)

    # ─── Step 1: Store patient ID before dropping ───
    patient_col = "patient_nbr" if "patient_nbr" in df.columns else None

    # ─── Step 2: Remove deceased / hospice patients ───
    # These patients cannot be readmitted — including them would be data leakage
    deceased_dispositions = [11, 13, 14, 19, 20, 21]
    mask = df["discharge_disposition_id"].isin(deceased_dispositions)
    df = df[~mask]
    logger.info(f"Removed {mask.sum():,} deceased/hospice encounters")

    # ─── Step 3: Handle duplicate encounters per patient ───
    if patient_col and patient_col in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=[patient_col], keep="first")
        logger.info(f"Removed {before - len(df):,} duplicate patient encounters (kept first)")

    # ─── Step 4: Drop uninformative columns ───
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}")

    # ─── Step 5: Encode target variable ───
    df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAPPING)
    if df[TARGET_COL].isna().any():
        logger.warning(f"Found {df[TARGET_COL].isna().sum()} unmapped target values — dropping")
        df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    logger.info(f"Target distribution:\n{df[TARGET_COL].value_counts().to_string()}")

    # ─── Step 6: Map ICD-9 codes ───
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        if diag_col in df.columns:
            group_col = f"{diag_col}_group"
            df[group_col] = df[diag_col].apply(_map_icd9_code)
            df = df.drop(columns=[diag_col])
            logger.info(f"Mapped {diag_col} → {group_col} ({df[group_col].nunique()} groups)")

    # ─── Step 7: Clean categorical variables ───
    # Gender: remove 'Unknown/Invalid'
    if "gender" in df.columns:
        invalid_mask = df["gender"] == "Unknown/Invalid"
        if invalid_mask.sum() > 0:
            df = df[~invalid_mask]
            logger.info(f"Removed {invalid_mask.sum()} rows with invalid gender")

    # Race: fill NaN with 'Unknown'
    if "race" in df.columns:
        df["race"] = df["race"].fillna("Unknown")

    # Age: keep as categorical bracket (useful for dashboard filters)
    # Also create a numeric version
    if "age" in df.columns:
        df["age_numeric"] = df["age"].apply(_map_age_to_numeric)

    # ─── Step 8: Convert ID columns to string type ───
    id_cols = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ─── Step 9: Handle remaining missing values ───
    # Numeric: fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled {col} NaN with median={median_val}")

    # Categorical: fill with 'Unknown'
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna("Unknown")

    # ─── Final summary ───
    logger.info(f"Cleaning complete: {initial_rows:,} → {len(df):,} rows "
                f"({initial_rows - len(df):,} removed)")

    return df.reset_index(drop=True)


@timer
def save_processed_data(df: pd.DataFrame, filename: str = "cleaned_data.csv") -> str:
    """Save cleaned data to processed directory."""
    filepath = PROCESSED_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    logger.info(f"Saved processed data to {filepath}")
    return str(filepath)


if __name__ == "__main__":
    from src.data.ingest import load_raw_data

    raw_df = load_raw_data()
    print_dataframe_summary(raw_df, "Raw Data")

    clean_df = clean_data(raw_df)
    print_dataframe_summary(clean_df, "Cleaned Data")

    save_processed_data(clean_df)
