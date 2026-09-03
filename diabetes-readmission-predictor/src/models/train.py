"""
Model Training Module
=====================
Trains and compares multiple classification models for readmission
prediction. Supports hyperparameter tuning via Optuna and stores
results for comparison.
"""
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils.config import (
    RANDOM_STATE,
    CV_FOLDS,
    OPTUNA_N_TRIALS,
    PRIMARY_METRIC,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.utils.helpers import get_logger, timer

logger = get_logger(__name__)

# Suppress Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


def get_base_models() -> dict:
    """
    Return a dictionary of base models with default hyperparameters.

    Returns
    -------
    dict
        {model_name: sklearn-compatible estimator}
    """
    return {
        "LogisticRegression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=200,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=1,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            random_state=RANDOM_STATE,
            n_estimators=200,
            class_weight="balanced",
            verbose=-1,
            n_jobs=-1,
        ),
    }


@timer
def train_baseline_models(X_train, y_train, X_test, y_test) -> dict:
    """
    Train all baseline models and evaluate with cross-validation.

    Parameters
    ----------
    X_train : array-like
        Preprocessed training features.
    y_train : array-like
        Training labels.
    X_test : array-like
        Preprocessed test features.
    y_test : array-like
        Test labels.

    Returns
    -------
    dict
        {model_name: {'model': fitted_model, 'cv_scores': array, 'cv_mean': float}}
    """
    models = get_base_models()
    results = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        logger.info(f"Training {name}...")

        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring=PRIMARY_METRIC,
            n_jobs=-1,
        )

        # Fit on full training set
        model.fit(X_train, y_train)

        results[name] = {
            "model": model,
            "cv_scores": cv_scores,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

        logger.info(
            f"  {name}: CV {PRIMARY_METRIC} = "
            f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

    return results


def _optuna_xgboost_objective(trial, X_train, y_train, cv):
    """Optuna objective for XGBoost hyperparameter tuning."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
    }

    model = XGBClassifier(
        **params,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0,
    )

    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=PRIMARY_METRIC, n_jobs=-1)
    return scores.mean()


def _optuna_lgbm_objective(trial, X_train, y_train, cv):
    """Optuna objective for LightGBM hyperparameter tuning."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = LGBMClassifier(
        **params,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        verbose=-1,
        n_jobs=-1,
    )

    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=PRIMARY_METRIC, n_jobs=-1)
    return scores.mean()


@timer
def tune_best_model(X_train, y_train, model_name: str = "XGBoost", n_trials: int = None) -> dict:
    """
    Hyperparameter tuning using Optuna for the best-performing model.

    Parameters
    ----------
    X_train : array-like
        Preprocessed training features.
    y_train : array-like
        Training labels.
    model_name : str
        Model to tune ('XGBoost' or 'LightGBM').
    n_trials : int, optional
        Number of Optuna trials. Defaults to config value.

    Returns
    -------
    dict
        {'model': fitted_model, 'best_params': dict, 'best_score': float}
    """
    if n_trials is None:
        n_trials = OPTUNA_N_TRIALS

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    logger.info(f"Tuning {model_name} with {n_trials} Optuna trials...")

    if model_name == "XGBoost":
        objective = lambda trial: _optuna_xgboost_objective(trial, X_train, y_train, cv)
    elif model_name == "LightGBM":
        objective = lambda trial: _optuna_lgbm_objective(trial, X_train, y_train, cv)
    else:
        raise ValueError(f"Tuning not implemented for {model_name}")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value

    logger.info(f"Best {PRIMARY_METRIC}: {best_score:.4f}")
    logger.info(f"Best params: {best_params}")

    # Train final model with best params
    if model_name == "XGBoost":
        best_model = XGBClassifier(
            **best_params,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        )
    else:
        best_model = LGBMClassifier(
            **best_params,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbose=-1,
            n_jobs=-1,
        )

    best_model.fit(X_train, y_train)

    return {
        "model": best_model,
        "best_params": best_params,
        "best_score": best_score,
        "study": study,
    }


@timer
def save_model(model, preprocessor, metadata: dict, filename: str = "best_model.pkl") -> str:
    """
    Save model, preprocessor, and metadata as a single artifact.

    Parameters
    ----------
    model : estimator
        Trained model.
    preprocessor : ColumnTransformer
        Fitted preprocessor.
    metadata : dict
        Model metadata (params, scores, features, etc.).
    filename : str
        Output filename.

    Returns
    -------
    str
        Path to saved model.
    """
    filepath = MODELS_DIR / filename
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "saved_at": datetime.now().isoformat(),
    }

    joblib.dump(artifact, filepath)
    logger.info(f"Model saved to {filepath}")

    # Also save metadata as JSON
    meta_path = MODELS_DIR / "model_metadata.json"
    json_metadata = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, list, dict))}
    with open(meta_path, "w") as f:
        json.dump(json_metadata, f, indent=2, default=str)

    return str(filepath)


def load_model(filename: str = "best_model.pkl") -> dict:
    """Load a saved model artifact."""
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found at {filepath}")
    artifact = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return artifact


@timer
def generate_comparison_table(results: dict) -> pd.DataFrame:
    """
    Generate a model comparison summary table.

    Parameters
    ----------
    results : dict
        Output from train_baseline_models().

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by mean CV score.
    """
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            f"CV {PRIMARY_METRIC} (mean)": f"{res['cv_mean']:.4f}",
            f"CV {PRIMARY_METRIC} (std)": f"{res['cv_std']:.4f}",
            "CV Scores": [f"{s:.4f}" for s in res['cv_scores']],
        })

    df = pd.DataFrame(rows).sort_values(
        f"CV {PRIMARY_METRIC} (mean)", ascending=False
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    print("Model training module loaded successfully.")
    print(f"Available models: {list(get_base_models().keys())}")
