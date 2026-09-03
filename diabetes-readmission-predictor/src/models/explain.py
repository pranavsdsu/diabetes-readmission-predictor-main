"""
Model Explainability Module
============================
SHAP-based model interpretability: global feature importance,
local explanations, and interaction analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.utils.config import FIGURES_DIR
from src.utils.helpers import get_logger, timer, save_figure, set_plot_style

logger = get_logger(__name__)


@timer
def compute_shap_values(model, X_test, feature_names: list = None, max_samples: int = 1000):
    """
    Compute SHAP values for model predictions.

    Parameters
    ----------
    model : estimator
        Trained tree-based model.
    X_test : array-like
        Test features (processed).
    feature_names : list, optional
        Feature names for the processed features.
    max_samples : int
        Maximum samples to compute SHAP for (speed).

    Returns
    -------
    dict
        {'shap_values': array, 'explainer': TreeExplainer, 'X_sample': array}
    """
    logger.info(f"Computing SHAP values (max {max_samples} samples)...")

    # Subsample if needed
    if X_test.shape[0] > max_samples:
        indices = np.random.RandomState(42).choice(X_test.shape[0], max_samples, replace=False)
        X_sample = X_test[indices] if hasattr(X_test, '__getitem__') else X_test.iloc[indices]
    else:
        X_sample = X_test
        indices = np.arange(X_test.shape[0])

    # Create explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        logger.info("TreeExplainer failed, falling back to KernelExplainer...")
        background = shap.sample(X_sample, min(100, len(X_sample)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample)

    # For binary classification, take the positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Create DataFrame for easier manipulation
    if feature_names is not None:
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
    else:
        shap_df = pd.DataFrame(shap_values)

    logger.info(f"SHAP values computed. Shape: {shap_values.shape}")

    return {
        "shap_values": shap_values,
        "shap_df": shap_df,
        "explainer": explainer,
        "X_sample": X_sample,
        "indices": indices,
    }


@timer
def plot_shap_summary(shap_result: dict, feature_names: list = None,
                      max_display: int = 20, save: bool = True):
    """
    Plot SHAP summary (beeswarm) plot showing global feature importance.

    Parameters
    ----------
    shap_result : dict
        Output from compute_shap_values().
    feature_names : list
        Feature names.
    max_display : int
        Max features to display.
    save : bool
        Whether to save.
    """
    logger.info("Generating SHAP summary plot...")

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_result["shap_values"],
        shap_result["X_sample"],
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Feature Importance (Global)", fontsize=14, pad=15)
    plt.tight_layout()

    if save:
        fig = plt.gcf()
        save_figure(fig, "shap_summary.png", dpi=150)
    else:
        plt.show()

    return fig


@timer
def plot_shap_bar(shap_result: dict, feature_names: list = None,
                  max_display: int = 20, save: bool = True):
    """
    Plot SHAP bar chart (mean absolute SHAP values).
    """
    logger.info("Generating SHAP bar plot...")

    mean_abs_shap = np.abs(shap_result["shap_values"]).mean(axis=0)

    if feature_names is not None:
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean |SHAP|": mean_abs_shap,
        }).sort_values("Mean |SHAP|", ascending=True).tail(max_display)
    else:
        importance_df = pd.DataFrame({
            "Feature": [f"Feature {i}" for i in range(len(mean_abs_shap))],
            "Mean |SHAP|": mean_abs_shap,
        }).sort_values("Mean |SHAP|", ascending=True).tail(max_display)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df["Feature"], importance_df["Mean |SHAP|"], color="#2196F3")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Top Feature Importance (SHAP)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    if save:
        save_figure(fig, "shap_bar.png")
    return fig


@timer
def plot_shap_force(shap_result: dict, idx: int = 0, feature_names: list = None,
                    save: bool = True):
    """
    Plot SHAP force plot for a single prediction.

    Parameters
    ----------
    shap_result : dict
        Output from compute_shap_values().
    idx : int
        Index in the sample to explain.
    feature_names : list
        Feature names.
    save : bool
        Whether to save.
    """
    logger.info(f"Generating SHAP force plot for sample index {idx}...")

    expected_value = shap_result["explainer"].expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

    force_plot = shap.force_plot(
        expected_value,
        shap_result["shap_values"][idx],
        shap_result["X_sample"][idx] if hasattr(shap_result["X_sample"], '__getitem__') else None,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )

    if save:
        fig = plt.gcf()
        save_figure(fig, f"shap_force_idx{idx}.png", dpi=150)

    return force_plot


@timer
def plot_shap_dependence(shap_result: dict, feature_idx: int, feature_names: list = None,
                         interaction_idx: str = "auto", save: bool = True):
    """
    Plot SHAP dependence plot for a specific feature.

    Parameters
    ----------
    shap_result : dict
        Output from compute_shap_values().
    feature_idx : int
        Index of the feature to plot.
    feature_names : list
        Feature names.
    interaction_idx : str or int
        Feature to color by for interaction.
    save : bool
        Whether to save.
    """
    feature_name = feature_names[feature_idx] if feature_names else f"Feature {feature_idx}"
    logger.info(f"Generating SHAP dependence plot for '{feature_name}'...")

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        feature_idx,
        shap_result["shap_values"],
        shap_result["X_sample"],
        feature_names=feature_names,
        interaction_index=interaction_idx,
        ax=ax,
        show=False,
    )
    plt.title(f"SHAP Dependence: {feature_name}", fontsize=14)
    plt.tight_layout()

    if save:
        safe_name = feature_name.replace(" ", "_").replace("/", "_")[:30]
        save_figure(fig, f"shap_dependence_{safe_name}.png")
    return fig


def get_top_features(shap_result: dict, feature_names: list, top_n: int = 10) -> pd.DataFrame:
    """
    Get the top N most important features by mean absolute SHAP value.

    Returns
    -------
    pd.DataFrame
        Feature importance ranking.
    """
    mean_abs = np.abs(shap_result["shap_values"]).mean(axis=0)

    df = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP|": mean_abs,
        "Rank": range(1, len(feature_names) + 1),
    }).sort_values("Mean |SHAP|", ascending=False).head(top_n)

    df["Rank"] = range(1, len(df) + 1)

    return df.reset_index(drop=True)


def generate_plain_language_summary(top_features_df: pd.DataFrame) -> str:
    """
    Generate a plain-English summary of the top risk factors.

    Parameters
    ----------
    top_features_df : pd.DataFrame
        Output from get_top_features().

    Returns
    -------
    str
        Human-readable summary.
    """
    lines = [
        "## Top Risk Factors for 30-Day Hospital Readmission",
        "",
        "Based on SHAP analysis of the best-performing model, the following",
        "factors have the greatest influence on predicting whether a diabetes",
        "patient will be readmitted within 30 days:",
        "",
    ]

    for _, row in top_features_df.iterrows():
        feature = row["Feature"]
        importance = row["Mean |SHAP|"]

        # Clean up feature names for readability
        clean_name = (
            feature
            .replace("num__", "")
            .replace("cat__", "")
            .replace("_", " ")
            .title()
        )

        lines.append(f"{int(row['Rank'])}. **{clean_name}** (importance: {importance:.4f})")

    lines.extend([
        "",
        "### Clinical Implications",
        "",
        "These features suggest that readmission risk is primarily driven by:",
        "- **Visit history** — patients with more prior inpatient/emergency visits are at higher risk",
        "- **Medication complexity** — number of medications and medication changes signal severity",
        "- **Length of stay** — longer hospitalizations correlate with higher readmission probability",
        "- **Diagnosis patterns** — certain primary diagnoses (circulatory, respiratory) increase risk",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    print("Explainability module loaded successfully.")
