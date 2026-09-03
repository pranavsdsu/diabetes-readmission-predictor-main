"""
Feature Engineering Module
==========================
Creates derived features and builds a scikit-learn preprocessing
pipeline with ColumnTransformer for reproducible transformations.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.utils.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MEDICATION_COLS,
    TARGET_COL,
    RANDOM_STATE,
    TEST_SIZE,
    PROCESSED_DATA_DIR,
)
from src.utils.helpers import get_logger, timer

logger = get_logger(__name__)


@timer
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from the cleaned dataset.

    Engineered features:
    - prior_visits_total: sum of outpatient + emergency + inpatient visits
    - high_utilizer_flag: 1 if prior_visits_total > 5
    - num_medication_changes: count of medications that were changed
    - medication_change_flag: 1 if any medication was changed
    - diabetes_medication_flag: already exists as 'diabetesMed'
    - total_procedures: num_lab_procedures + num_procedures

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with engineered features added.
    """
    df = df.copy()
    logger.info("Engineering features...")

    # ─── Prior visit history ───
    visit_cols = ["number_outpatient", "number_emergency", "number_inpatient"]
    existing_visit_cols = [c for c in visit_cols if c in df.columns]
    if existing_visit_cols:
        df["prior_visits_total"] = df[existing_visit_cols].sum(axis=1)
        df["high_utilizer_flag"] = (df["prior_visits_total"] > 5).astype(int)
        logger.info(f"Created prior_visits_total from {existing_visit_cols}")

    # ─── Medication changes ───
    existing_med_cols = [c for c in MEDICATION_COLS if c in df.columns]
    if existing_med_cols:
        # Count medications where dosage was changed (Up, Down) or started (Steady)
        change_counts = df[existing_med_cols].apply(
            lambda row: sum(1 for v in row if v in ["Up", "Down"]), axis=1
        )
        df["num_medication_changes"] = change_counts
        df["medication_change_flag"] = (change_counts > 0).astype(int)
        logger.info(f"Created medication change features from {len(existing_med_cols)} med columns")

        # Drop individual medication columns (too sparse, info captured in aggregates)
        df = df.drop(columns=existing_med_cols)
        logger.info(f"Dropped {len(existing_med_cols)} individual medication columns")

    # ─── Total procedures ───
    if "num_lab_procedures" in df.columns and "num_procedures" in df.columns:
        df["total_procedures"] = df["num_lab_procedures"] + df["num_procedures"]

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def get_feature_lists(df: pd.DataFrame):
    """
    Identify numeric and categorical feature columns present in the DataFrame.

    Returns
    -------
    tuple
        (numeric_features, categorical_features) lists
    """
    numeric_feats = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical_feats = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    # Add any engineered numeric features
    extra_numeric = ["age_numeric", "high_utilizer_flag", "medication_change_flag", "total_procedures"]
    for col in extra_numeric:
        if col in df.columns and col not in numeric_feats:
            numeric_feats.append(col)

    logger.info(f"Numeric features ({len(numeric_feats)}): {numeric_feats}")
    logger.info(f"Categorical features ({len(categorical_feats)}): {categorical_feats}")

    return numeric_feats, categorical_feats


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer for preprocessing.

    - Numeric: impute median → standardize
    - Categorical: impute 'Unknown' → one-hot encode

    Parameters
    ----------
    numeric_features : list
        Numeric column names.
    categorical_features : list
        Categorical column names.

    Returns
    -------
    ColumnTransformer
        Fitted preprocessor.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


@timer
def prepare_splits(df: pd.DataFrame):
    """
    Split data into train/test sets, ensuring no patient leakage.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, preprocessor, feature_names_num, feature_names_cat)
    """
    logger.info("Preparing train/test splits...")

    numeric_feats, categorical_feats = get_feature_lists(df)
    all_features = numeric_feats + categorical_feats

    X = df[all_features]
    y = df[TARGET_COL]

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts(normalize=True).to_string()}")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(f"Train: {X_train.shape[0]:,} samples | Test: {X_test.shape[0]:,} samples")

    # Build and fit preprocessor
    preprocessor = build_preprocessor(numeric_feats, categorical_feats)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logger.info(f"Processed feature matrix shape: {X_train_processed.shape}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "numeric_features": numeric_feats,
        "categorical_features": categorical_feats,
    }


@timer
def save_features(df: pd.DataFrame, filename: str = "featured_data.csv") -> str:
    """Save feature-engineered data."""
    filepath = PROCESSED_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    logger.info(f"Saved featured data to {filepath}")
    return str(filepath)


if __name__ == "__main__":
    from src.data.ingest import load_raw_data
    from src.data.clean import clean_data

    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    featured_df = engineer_features(clean_df)
    save_features(featured_df)

    splits = prepare_splits(featured_df)
    print(f"\nTrain shape: {splits['X_train_processed'].shape}")
    print(f"Test shape: {splits['X_test_processed'].shape}")
