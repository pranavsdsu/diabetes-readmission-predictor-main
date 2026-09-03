"""
Explainability Dashboard Page
==============================
SHAP-based model interpretability visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import json


def load_shap_data():
    """Load precomputed SHAP data."""
    shap_path = Path(__file__).resolve().parent.parent.parent / "models" / "shap_data.pkl"
    if not shap_path.exists():
        return None
    return joblib.load(shap_path)


def load_metadata():
    """Load model metadata."""
    meta_path = Path(__file__).resolve().parent.parent.parent / "models" / "model_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r") as f:
        return json.load(f)


def show_explainability():
    st.header("🔍 Model Explainability")
    st.markdown(
        "Understand what drives the model's predictions using SHAP "
        "(SHapley Additive exPlanations) analysis."
    )

    shap_data = load_shap_data()
    metadata = load_metadata()

    if shap_data is None:
        st.warning("SHAP data not found. Please run the explainability pipeline first.")
        st.code("make explain  # or python run_pipeline.py --step explain", language="bash")

        st.subheader("What SHAP Will Show")
        st.markdown("""
        After running the explainability pipeline, this page will display:

        1. **Global Feature Importance** — Which features matter most overall
        2. **Individual Predictions** — Why the model flagged a specific patient
        3. **Feature Interactions** — How features interact to influence risk
        4. **Cohort Analysis** — Risk factor differences across demographics
        """)
        return

    shap_values = shap_data["shap_values"]
    feature_names = shap_data.get("feature_names", [f"Feature {i}" for i in range(shap_values.shape[1])])
    X_sample = shap_data.get("X_sample", None)

    # ─── Tab Layout ───
    tab1, tab2, tab3 = st.tabs(["Global Importance", "Individual Predictions", "Feature Deep-Dive"])

    # ─── Tab 1: Global Feature Importance ───
    with tab1:
        st.subheader("Global Feature Importance")
        st.markdown("Features ranked by their average impact on model predictions.")

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean |SHAP|": mean_abs_shap,
        }).sort_values("Mean |SHAP|", ascending=True)

        top_n = st.slider("Number of features to show", 5, min(30, len(feature_names)), 15)
        display_df = importance_df.tail(top_n)

        fig = px.bar(
            display_df,
            x="Mean |SHAP|",
            y="Feature",
            orientation="h",
            title=f"Top {top_n} Features by SHAP Importance",
            color="Mean |SHAP|",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=max(400, top_n * 30), showlegend=False, yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.markdown("### Feature Importance Table")
        table_df = importance_df.sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
        table_df.index = table_df.index + 1
        table_df.index.name = "Rank"
        table_df["Mean |SHAP|"] = table_df["Mean |SHAP|"].round(4)
        st.dataframe(table_df.head(20), use_container_width=True)

    # ─── Tab 2: Individual Predictions ───
    with tab2:
        st.subheader("Individual Prediction Explanations")
        st.markdown("Select a patient to see what drove the model's prediction for them.")

        n_samples = min(shap_values.shape[0], 100)
        patient_idx = st.number_input(
            "Patient Index (from test set)",
            min_value=0,
            max_value=n_samples - 1,
            value=0,
            step=1,
        )

        # Get SHAP values for selected patient
        patient_shap = shap_values[patient_idx]
        patient_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": patient_shap,
            "Impact": ["Increases Risk" if v > 0 else "Decreases Risk" for v in patient_shap],
        }).sort_values("SHAP Value", key=abs, ascending=False).head(15)

        # Show prediction context
        if metadata:
            y_proba = metadata.get("y_proba", None)
            y_test = metadata.get("y_test", None)
            if y_proba and patient_idx < len(y_proba):
                col1, col2, col3 = st.columns(3)
                prob = y_proba[patient_idx]
                actual = y_test[patient_idx] if y_test else "N/A"
                risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"

                col1.metric("Predicted Probability", f"{prob:.3f}")
                col2.metric("Risk Level", risk_level)
                col3.metric("Actual Outcome", "Readmitted" if actual == 1 else "Not Readmitted")

        # Waterfall-style chart
        fig = go.Figure()

        colors = ["#FF5722" if v > 0 else "#2196F3" for v in patient_df["SHAP Value"]]
        fig.add_trace(go.Bar(
            x=patient_df["SHAP Value"],
            y=patient_df["Feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in patient_df["SHAP Value"]],
            textposition="auto",
        ))

        fig.update_layout(
            title=f"SHAP Explanation — Patient #{patient_idx}",
            xaxis_title="SHAP Value (impact on readmission prediction)",
            height=500,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Reading this chart:**
        - 🔴 **Red bars** push the prediction toward readmission (higher risk)
        - 🔵 **Blue bars** push the prediction away from readmission (lower risk)
        - Longer bars = stronger influence on this specific prediction
        """)

    # ─── Tab 3: Feature Deep-Dive ───
    with tab3:
        st.subheader("Feature Deep-Dive")
        st.markdown("Explore how individual features relate to SHAP values across all patients.")

        selected_feature = st.selectbox("Select Feature", sorted(feature_names))

        if selected_feature and X_sample is not None:
            feat_idx = feature_names.index(selected_feature)
            feat_shap = shap_values[:, feat_idx]

            # Try to get feature values
            if hasattr(X_sample, 'shape') and X_sample.shape[1] > feat_idx:
                feat_values = X_sample[:, feat_idx] if not isinstance(X_sample, pd.DataFrame) else X_sample.iloc[:, feat_idx]

                fig = px.scatter(
                    x=feat_values,
                    y=feat_shap,
                    labels={"x": selected_feature, "y": "SHAP Value"},
                    title=f"SHAP Dependence: {selected_feature}",
                    trendline="lowess",
                    opacity=0.4,
                )
                fig.update_layout(height=450, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            # Distribution of SHAP values for this feature
            fig2 = px.histogram(
                x=feat_shap,
                nbins=50,
                labels={"x": "SHAP Value"},
                title=f"Distribution of SHAP Values for {selected_feature}",
                color_discrete_sequence=["#2196F3"],
            )
            fig2.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

    # ─── Plain Language Summary ───
    st.markdown("---")
    st.subheader("📝 Key Risk Factors (Plain Language)")

    top_features = pd.DataFrame({
        "Feature": feature_names,
        "Importance": np.abs(shap_values).mean(axis=0),
    }).nlargest(5, "Importance")

    st.markdown("""
    Based on SHAP analysis, the top factors predicting 30-day readmission are:
    """)

    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        clean_name = row["Feature"].replace("num__", "").replace("cat__", "").replace("_", " ").title()
        st.markdown(f"**{i}. {clean_name}** — importance score: {row['Importance']:.4f}")

    st.info(
        "💡 **Clinical Takeaway**: Readmission risk is primarily driven by "
        "visit history (prior inpatient/emergency visits), medication complexity, "
        "and length of hospital stay. Targeted discharge planning for patients "
        "with these risk factors could reduce readmission rates."
    )
