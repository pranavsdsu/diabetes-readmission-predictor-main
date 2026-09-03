"""
Model Evaluation Module
=======================
Comprehensive evaluation metrics including AUROC, AUPRC, calibration,
confusion matrices, and demographic fairness analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.calibration import calibration_curve

from src.utils.config import FIGURES_DIR, FAIRNESS_ATTRIBUTES, TARGET_COL
from src.utils.helpers import get_logger, timer, save_figure, set_plot_style

logger = get_logger(__name__)


@timer
def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Run full evaluation suite on a model.

    Parameters
    ----------
    model : estimator
        Trained model with predict_proba().
    X_test : array-like
        Test features.
    y_test : array-like
        True labels.
    model_name : str
        Name for logging and file naming.

    Returns
    -------
    dict
        Dictionary of all evaluation metrics.
    """
    logger.info(f"Evaluating {model_name}...")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "model_name": model_name,
        "auroc": roc_auc_score(y_test, y_proba),
        "auprc": average_precision_score(y_test, y_proba),
        "brier_score": brier_score_loss(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "y_proba": y_proba,
        "y_pred": y_pred,
    }

    logger.info(
        f"  {model_name}: AUROC={metrics['auroc']:.4f} | "
        f"AUPRC={metrics['auprc']:.4f} | F1={metrics['f1']:.4f}"
    )

    return metrics


def find_optimal_threshold(y_true, y_proba, metric="f1") -> float:
    """
    Find the optimal classification threshold.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_proba : array-like
        Predicted probabilities.
    metric : str
        Metric to optimize ('f1', 'youden').

    Returns
    -------
    float
        Optimal threshold.
    """
    thresholds = np.arange(0.1, 0.9, 0.01)

    if metric == "f1":
        scores = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    elif metric == "youden":
        # Youden's J statistic: sensitivity + specificity - 1
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return roc_thresholds[best_idx]

    best_idx = np.argmax(scores)
    return thresholds[best_idx]


@timer
def plot_roc_curves(all_metrics: dict, save: bool = True):
    """
    Plot overlaid ROC curves for all models.

    Parameters
    ----------
    all_metrics : dict
        {model_name: metrics_dict from evaluate_model()}
    save : bool
        Whether to save the figure.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.Set2(np.linspace(0, 1, len(all_metrics)))

    for (name, metrics), color in zip(all_metrics.items(), colors):
        y_test = metrics["y_test"]
        y_proba = metrics["y_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auroc = metrics["auroc"]
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.4f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    if save:
        save_figure(fig, "roc_curves.png")
    return fig


@timer
def plot_precision_recall_curves(all_metrics: dict, save: bool = True):
    """Plot overlaid Precision-Recall curves for all models."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.Set2(np.linspace(0, 1, len(all_metrics)))

    for (name, metrics), color in zip(all_metrics.items(), colors):
        y_test = metrics["y_test"]
        y_proba = metrics["y_proba"]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        auprc = metrics["auprc"]
        ax.plot(recall, precision, label=f"{name} (AUPRC={auprc:.4f})", color=color, linewidth=2)

    # Baseline: prevalence
    baseline = all_metrics[list(all_metrics.keys())[0]]["y_test"]
    prevalence = np.mean(baseline)
    ax.axhline(y=prevalence, color="gray", linestyle="--", alpha=0.5, label=f"Baseline ({prevalence:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    if save:
        save_figure(fig, "precision_recall_curves.png")
    return fig


@timer
def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model",
                          threshold: float = 0.5, save: bool = True):
    """Plot confusion matrix heatmap."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Counts
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Not Readmitted", "Readmitted <30d"],
                yticklabels=["Not Readmitted", "Readmitted <30d"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title(f"{model_name} — Confusion Matrix (counts)")

    # Normalized
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues", ax=axes[1],
                xticklabels=["Not Readmitted", "Readmitted <30d"],
                yticklabels=["Not Readmitted", "Readmitted <30d"])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title(f"{model_name} — Confusion Matrix (normalized)")

    plt.tight_layout()

    if save:
        save_figure(fig, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    return fig


@timer
def plot_calibration_curve(all_metrics: dict, n_bins: int = 10, save: bool = True):
    """Plot calibration curves for all models."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.Set2(np.linspace(0, 1, len(all_metrics)))

    for (name, metrics), color in zip(all_metrics.items(), colors):
        y_test = metrics["y_test"]
        y_proba = metrics["y_proba"]
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins, strategy="uniform")
        ax.plot(prob_pred, prob_true, "s-", label=name, color=color, linewidth=2, markersize=6)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    if save:
        save_figure(fig, "calibration_curves.png")
    return fig


@timer
def fairness_analysis(model, X_test_df: pd.DataFrame, y_test, y_proba,
                      attribute: str, save: bool = True) -> pd.DataFrame:
    """
    Compute fairness metrics across demographic subgroups.

    Parameters
    ----------
    model : estimator
        Trained model.
    X_test_df : pd.DataFrame
        Original (non-processed) test features with demographic columns.
    y_test : array-like
        True labels.
    y_proba : array-like
        Predicted probabilities.
    attribute : str
        Demographic attribute column (e.g., 'race', 'gender').
    save : bool
        Whether to save the figure.

    Returns
    -------
    pd.DataFrame
        Fairness metrics by subgroup.
    """
    logger.info(f"Running fairness analysis on '{attribute}'...")

    if attribute not in X_test_df.columns:
        logger.warning(f"Attribute '{attribute}' not found in test data. Skipping.")
        return pd.DataFrame()

    results = []
    groups = X_test_df[attribute].unique()
    y_pred = (y_proba >= 0.5).astype(int)

    for group in sorted(groups):
        mask = X_test_df[attribute].values == group
        n = mask.sum()
        if n < 20:
            continue

        group_y_true = np.array(y_test)[mask]
        group_y_pred = y_pred[mask]
        group_y_proba = y_proba[mask]

        results.append({
            "Group": group,
            "N": n,
            "Prevalence": group_y_true.mean(),
            "Prediction Rate": group_y_pred.mean(),
            "AUROC": roc_auc_score(group_y_true, group_y_proba) if group_y_true.sum() > 0 else np.nan,
            "Precision": precision_score(group_y_true, group_y_pred, zero_division=0),
            "Recall": recall_score(group_y_true, group_y_pred, zero_division=0),
            "F1": f1_score(group_y_true, group_y_pred, zero_division=0),
        })

    fairness_df = pd.DataFrame(results)

    if save and len(fairness_df) > 0:
        set_plot_style()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, metric in enumerate(["AUROC", "Recall", "Prediction Rate"]):
            if metric in fairness_df.columns:
                ax = axes[idx]
                bars = ax.bar(fairness_df["Group"], fairness_df[metric], color=plt.cm.Set2(np.arange(len(fairness_df))))
                ax.set_title(f"{metric} by {attribute}")
                ax.set_ylabel(metric)
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True, alpha=0.3, axis="y")

                # Add value labels
                for bar, val in zip(bars, fairness_df[metric]):
                    if not np.isnan(val):
                        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        plt.suptitle(f"Fairness Analysis — {attribute}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_figure(fig, f"fairness_{attribute.lower()}.png")

    logger.info(f"Fairness analysis for '{attribute}':\n{fairness_df.to_string()}")
    return fairness_df


def generate_metrics_summary(all_metrics: dict) -> pd.DataFrame:
    """
    Create a summary DataFrame of all evaluation metrics across models.

    Parameters
    ----------
    all_metrics : dict
        {model_name: metrics_dict}

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Model": name,
            "AUROC": f"{m['auroc']:.4f}",
            "AUPRC": f"{m['auprc']:.4f}",
            "F1": f"{m['f1']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "Accuracy": f"{m['accuracy']:.4f}",
            "Brier Score": f"{m['brier_score']:.4f}",
        })

    return pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("Evaluation module loaded successfully.")
