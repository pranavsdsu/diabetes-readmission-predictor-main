"""
Model Performance Dashboard Page
=================================
Displays ROC curves, PR curves, calibration plots, confusion matrices,
and model comparison tables.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import joblib

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve


def load_model_artifact():
    """Load the saved model artifact."""
    model_path = Path(__file__).resolve().parent.parent.parent / "models" / "best_model.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def load_metadata():
    """Load model metadata JSON."""
    meta_path = Path(__file__).resolve().parent.parent.parent / "models" / "model_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r") as f:
        return json.load(f)


def show_model_performance():
    st.header("🤖 Model Performance")
    st.markdown("Compare model performance across multiple metrics and visualizations.")

    artifact = load_model_artifact()
    metadata = load_metadata()

    if artifact is None or metadata is None:
        st.warning("Model artifacts not found. Please run the training pipeline first.")
        st.code("make train  # or python -m src.models.train", language="bash")

        # Show placeholder with expected metrics
        st.subheader("Expected Model Comparison")
        st.markdown("""
        After training, this page will show:
        - **ROC Curves** for all 4 models (overlaid)
        - **Precision-Recall Curves** (critical for imbalanced data)
        - **Calibration Curves** (reliability of probabilities)
        - **Confusion Matrix** with adjustable threshold
        - **Full Metrics Table** (AUROC, AUPRC, F1, Precision, Recall, Brier Score)
        """)
        return

    # Extract data from metadata
    model = artifact["model"]
    all_results = metadata.get("all_model_results", {})
    best_model_name = metadata.get("best_model_name", "Best Model")

    # ─── Metrics Overview ───
    st.subheader("Performance Summary")

    if all_results:
        results_df = pd.DataFrame(all_results).T
        results_df.index.name = "Model"
        results_df = results_df.reset_index()

        # Highlight best model
        st.dataframe(
            results_df.style.highlight_max(
                subset=["auroc", "auprc", "f1"],
                color="#90EE90",
            ),
            use_container_width=True,
        )
    else:
        st.info("Detailed model comparison will be available after running the full training pipeline.")

    st.markdown("---")

    # ─── Metric Cards for Best Model ───
    st.subheader(f"Best Model: {best_model_name}")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    best_metrics = metadata.get("best_metrics", {})

    col1.metric("AUROC", f"{best_metrics.get('auroc', 'N/A')}")
    col2.metric("AUPRC", f"{best_metrics.get('auprc', 'N/A')}")
    col3.metric("F1 Score", f"{best_metrics.get('f1', 'N/A')}")
    col4.metric("Precision", f"{best_metrics.get('precision', 'N/A')}")
    col5.metric("Recall", f"{best_metrics.get('recall', 'N/A')}")
    col6.metric("Brier Score", f"{best_metrics.get('brier_score', 'N/A')}")

    st.markdown("---")

    # ─── Interactive Threshold Analysis ───
    st.subheader("🎚️ Threshold Analysis")
    st.markdown("Adjust the classification threshold to see its impact on predictions.")

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Probability threshold above which a patient is predicted as 'will be readmitted'",
    )

    # If we have test predictions stored
    y_test = metadata.get("y_test", None)
    y_proba = metadata.get("y_proba", None)

    if y_test is not None and y_proba is not None:
        y_test = np.array(y_test)
        y_proba = np.array(y_proba)
        y_pred_thresh = (y_proba >= threshold).astype(int)

        col1, col2, col3, col4 = st.columns(4)

        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        flagged = y_pred_thresh.mean() * 100

        col1.metric("Precision", f"{prec:.3f}")
        col2.metric("Recall", f"{rec:.3f}")
        col3.metric("F1 Score", f"{f1:.3f}")
        col4.metric("% Flagged", f"{flagged:.1f}%")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_thresh)
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["Not Readmitted", "Readmitted <30d"],
            y=["Not Readmitted", "Readmitted <30d"],
            color_continuous_scale="Blues",
            title=f"Confusion Matrix (threshold={threshold})",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ─── ROC Curve (if we have stored curves) ───
    st.subheader("ROC & Precision-Recall Curves")

    roc_data = metadata.get("roc_curves", None)
    pr_data = metadata.get("pr_curves", None)

    if y_test is not None and y_proba is not None:
        col_left, col_right = st.columns(2)

        with col_left:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auroc = roc_auc_score(y_test, y_proba)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{best_model_name} (AUROC={auroc:.4f})",
                                     line=dict(color="#2196F3", width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="Random", line=dict(color="gray", dash="dash")))
            fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                              height=450, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_proba)
            auprc = average_precision_score(y_test, y_proba)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recall_arr, y=precision_arr, mode="lines",
                                     name=f"{best_model_name} (AUPRC={auprc:.4f})",
                                     line=dict(color="#FF5722", width=3)))
            fig.add_hline(y=y_test.mean(), line_dash="dash", line_color="gray",
                          annotation_text=f"Baseline ({y_test.mean():.3f})")
            fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall",
                              yaxis_title="Precision", height=450, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    # ─── Calibration Curve ───
    st.subheader("Calibration Curve")
    if y_test is not None and y_proba is not None:
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers",
                                 name=best_model_name, marker=dict(size=8),
                                 line=dict(color="#4CAF50", width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 name="Perfect Calibration",
                                 line=dict(color="gray", dash="dash")))
        fig.update_layout(title="Calibration Curve", xaxis_title="Mean Predicted Probability",
                          yaxis_title="Fraction of Positives", height=450, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
