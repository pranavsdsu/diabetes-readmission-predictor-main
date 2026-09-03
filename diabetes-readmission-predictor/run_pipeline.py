"""
Run Pipeline — End-to-End Orchestrator
========================================
Runs the complete ML pipeline from data ingestion through model
training, evaluation, and SHAP explainability.

Usage:
    python run_pipeline.py                  # Run all steps
    python run_pipeline.py --step data      # Only data pipeline
    python run_pipeline.py --step train     # Only training
    python run_pipeline.py --step evaluate  # Only evaluation
    python run_pipeline.py --step explain   # Only explainability
"""
import sys
import argparse
import json
import numpy as np
import joblib

from src.utils.helpers import get_logger, timer
from src.utils.config import MODELS_DIR, REPORTS_DIR

logger = get_logger("pipeline")


@timer
def run_data_pipeline():
    """Phase 1: Data ingestion, cleaning, and feature engineering."""
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA PIPELINE")
    logger.info("=" * 60)

    from src.data.ingest import download_dataset, load_raw_data, load_to_sqlite
    from src.data.clean import clean_data, save_processed_data
    from src.data.features import engineer_features, save_features
    from src.utils.helpers import print_dataframe_summary

    # Step 1: Download & load
    download_dataset()
    raw_df = load_raw_data()
    print_dataframe_summary(raw_df, "Raw Data")

    # Step 2: Clean
    clean_df = clean_data(raw_df)
    save_processed_data(clean_df, "cleaned_data.csv")
    print_dataframe_summary(clean_df, "Cleaned Data")

    # Step 3: Feature engineering
    featured_df = engineer_features(clean_df)
    save_features(featured_df, "featured_data.csv")
    print_dataframe_summary(featured_df, "Featured Data")

    # Step 4: SQLite ingestion (for SQL demos)
    load_to_sqlite(featured_df)

    logger.info("Phase 1 complete!")
    return featured_df


@timer
def run_training_pipeline(featured_df=None):
    """Phase 2: Model training and hyperparameter tuning."""
    logger.info("=" * 60)
    logger.info("PHASE 2: MODEL TRAINING")
    logger.info("=" * 60)

    import pandas as pd
    from src.data.features import engineer_features, prepare_splits
    from src.data.clean import clean_data
    from src.data.ingest import load_raw_data
    from src.models.train import (
        train_baseline_models,
        tune_best_model,
        save_model,
        generate_comparison_table,
    )
    from src.utils.config import PROCESSED_DATA_DIR

    # Load featured data if not provided
    if featured_df is None:
        featured_path = PROCESSED_DATA_DIR / "featured_data.csv"
        if featured_path.exists():
            featured_df = pd.read_csv(featured_path)
            logger.info(f"Loaded featured data from {featured_path}")
        else:
            logger.info("Featured data not found, running data pipeline first...")
            featured_df = run_data_pipeline()

    # Prepare train/test splits
    splits = prepare_splits(featured_df)

    X_train = splits["X_train_processed"]
    X_test = splits["X_test_processed"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]

    # Train baseline models
    baseline_results = train_baseline_models(X_train, y_train, X_test, y_test)

    # Print comparison
    comparison = generate_comparison_table(baseline_results)
    logger.info(f"\nModel Comparison:\n{comparison.to_string()}")

    # Find best baseline model
    best_name = max(baseline_results, key=lambda k: baseline_results[k]["cv_mean"])
    logger.info(f"\nBest baseline model: {best_name} (CV={baseline_results[best_name]['cv_mean']:.4f})")

    # Tune the best model
    if best_name in ("XGBoost", "LightGBM"):
        tuned = tune_best_model(X_train, y_train, model_name=best_name, n_trials=30)
        best_model = tuned["model"]
        best_params = tuned["best_params"]
        logger.info(f"Tuned {best_name}: {tuned['best_score']:.4f}")
    else:
        best_model = baseline_results[best_name]["model"]
        best_params = best_model.get_params()

    # Get test predictions for the best model
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    # Compute metrics
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
    from sklearn.metrics import precision_score, recall_score, accuracy_score

    best_metrics = {
        "auroc": round(roc_auc_score(y_test, y_proba), 4),
        "auprc": round(average_precision_score(y_test, y_proba), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "brier_score": round(brier_score_loss(y_test, y_proba), 4),
    }

    # Save all results in metadata
    all_model_results = {}
    for name, res in baseline_results.items():
        m = res["model"]
        proba = m.predict_proba(X_test)[:, 1]
        pred = m.predict(X_test)
        all_model_results[name] = {
            "auroc": round(roc_auc_score(y_test, proba), 4),
            "auprc": round(average_precision_score(y_test, proba), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        }

    metadata = {
        "best_model_name": best_name,
        "best_params": best_params,
        "best_metrics": best_metrics,
        "all_model_results": all_model_results,
        "y_test": y_test.tolist(),
        "y_proba": y_proba.tolist(),
        "numeric_features": splits["numeric_features"],
        "categorical_features": splits["categorical_features"],
    }

    # Save model artifact
    save_model(best_model, splits["preprocessor"], metadata)

    # Save comparison table
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    logger.info("Phase 2 complete!")
    return best_model, splits, metadata


@timer
def run_evaluation_pipeline(model=None, splits=None, metadata=None):
    """Phase 3: Comprehensive evaluation and fairness analysis."""
    logger.info("=" * 60)
    logger.info("PHASE 3: EVALUATION & FAIRNESS")
    logger.info("=" * 60)

    import pandas as pd
    from src.models.evaluate import (
        evaluate_model,
        plot_roc_curves,
        plot_precision_recall_curves,
        plot_confusion_matrix,
        plot_calibration_curve,
        fairness_analysis,
        generate_metrics_summary,
    )
    from src.models.train import load_model
    from src.utils.config import PROCESSED_DATA_DIR, FAIRNESS_ATTRIBUTES

    # Load if not provided
    if model is None:
        artifact = load_model()
        model = artifact["model"]
        metadata = artifact["metadata"]

    if splits is None:
        from src.data.features import prepare_splits
        featured_df = pd.read_csv(PROCESSED_DATA_DIR / "featured_data.csv")
        splits = prepare_splits(featured_df)

    X_test = splits["X_test_processed"]
    y_test = splits["y_test"]

    # Evaluate best model
    best_name = metadata.get("best_model_name", "Best Model")
    metrics = evaluate_model(model, X_test, y_test, model_name=best_name)
    metrics["y_test"] = y_test

    all_metrics = {best_name: metrics}

    # Plot evaluation charts
    try:
        plot_roc_curves(all_metrics)
        plot_precision_recall_curves(all_metrics)
        plot_confusion_matrix(y_test, metrics["y_pred"], model_name=best_name)
        plot_calibration_curve(all_metrics)
    except Exception as e:
        logger.warning(f"Some plots failed: {e}")

    # Fairness analysis
    X_test_df = splits["X_test"]
    for attr in FAIRNESS_ATTRIBUTES:
        if attr in X_test_df.columns:
            try:
                fairness_df = fairness_analysis(
                    model, X_test_df, y_test, metrics["y_proba"], attribute=attr
                )
                fairness_df.to_csv(REPORTS_DIR / f"fairness_{attr}.csv", index=False)
            except Exception as e:
                logger.warning(f"Fairness analysis for {attr} failed: {e}")

    # Summary
    summary_df = generate_metrics_summary(all_metrics)
    logger.info(f"\nMetrics Summary:\n{summary_df.to_string()}")

    logger.info("Phase 3 complete!")
    return metrics


@timer
def run_explainability_pipeline(model=None, splits=None, metadata=None):
    """Phase 3b: SHAP explainability analysis."""
    logger.info("=" * 60)
    logger.info("PHASE 3b: EXPLAINABILITY (SHAP)")
    logger.info("=" * 60)

    import pandas as pd
    from src.models.explain import (
        compute_shap_values,
        plot_shap_summary,
        plot_shap_bar,
        plot_shap_dependence,
        get_top_features,
        generate_plain_language_summary,
    )
    from src.models.train import load_model
    from src.utils.config import PROCESSED_DATA_DIR

    # Load if not provided
    if model is None:
        artifact = load_model()
        model = artifact["model"]
        metadata = artifact["metadata"]

    if splits is None:
        from src.data.features import prepare_splits
        featured_df = pd.read_csv(PROCESSED_DATA_DIR / "featured_data.csv")
        splits = prepare_splits(featured_df)

    X_test = splits["X_test_processed"]
    preprocessor = splits["preprocessor"]

    # Get feature names from preprocessor
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

    # Compute SHAP values
    shap_result = compute_shap_values(model, X_test, feature_names=feature_names)

    # Generate plots
    try:
        plot_shap_summary(shap_result, feature_names)
        plot_shap_bar(shap_result, feature_names)
    except Exception as e:
        logger.warning(f"SHAP plots failed: {e}")

    # Top features analysis
    top_features = get_top_features(shap_result, feature_names, top_n=10)
    logger.info(f"\nTop Features:\n{top_features.to_string()}")

    # Generate dependence plots for top 3 features
    for i in range(min(3, len(feature_names))):
        try:
            top_feat_name = top_features.iloc[i]["Feature"]
            feat_idx = feature_names.index(top_feat_name)
            plot_shap_dependence(shap_result, feat_idx, feature_names)
        except Exception as e:
            logger.warning(f"Dependence plot {i} failed: {e}")

    # Plain language summary
    summary = generate_plain_language_summary(top_features)
    summary_path = REPORTS_DIR / "shap_summary.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    logger.info(f"SHAP summary saved to {summary_path}")

    # Save SHAP data for dashboard
    shap_data = {
        "shap_values": shap_result["shap_values"],
        "feature_names": feature_names,
        "X_sample": shap_result["X_sample"],
    }
    joblib.dump(shap_data, MODELS_DIR / "shap_data.pkl")

    logger.info("Phase 3b complete!")
    return shap_result


def main():
    parser = argparse.ArgumentParser(description="Diabetes Readmission Prediction Pipeline")
    parser.add_argument(
        "--step",
        choices=["data", "train", "evaluate", "explain", "all"],
        default="all",
        help="Which pipeline step to run (default: all)",
    )
    args = parser.parse_args()

    logger.info("🏥 Diabetes Readmission Prediction Pipeline")
    logger.info("=" * 60)

    if args.step in ("data", "all"):
        featured_df = run_data_pipeline()

    if args.step in ("train", "all"):
        model, splits, metadata = run_training_pipeline(
            featured_df if args.step == "all" else None
        )

    if args.step in ("evaluate", "all"):
        run_evaluation_pipeline(
            model if args.step == "all" else None,
            splits if args.step == "all" else None,
            metadata if args.step == "all" else None,
        )

    if args.step in ("explain", "all"):
        run_explainability_pipeline(
            model if args.step == "all" else None,
            splits if args.step == "all" else None,
            metadata if args.step == "all" else None,
        )

    logger.info("=" * 60)
    logger.info("🎉 Pipeline complete!")
    logger.info("Run 'streamlit run app/streamlit_app.py' to launch the dashboard")


if __name__ == "__main__":
    main()
