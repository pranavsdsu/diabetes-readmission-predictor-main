"""
EDA Dashboard Page
==================
Interactive exploratory data analysis with filters and dynamic plots.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def load_data():
    """Load processed data."""
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "featured_data.csv"
    if not data_path.exists():
        data_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "cleaned_data.csv"
    if not data_path.exists():
        st.error("Processed data not found. Please run the data pipeline first.")
        st.stop()
    return pd.read_csv(data_path)


def show_eda():
    st.header("📊 Exploratory Data Analysis")
    st.markdown("Explore the diabetes dataset interactively with filters and visualizations.")

    df = load_data()

    # ─── Sidebar Filters ───
    st.sidebar.markdown("### 🔧 EDA Filters")

    # Age filter
    if "age" in df.columns:
        age_options = sorted(df["age"].unique())
        selected_ages = st.sidebar.multiselect("Age Groups", age_options, default=age_options)
        df = df[df["age"].isin(selected_ages)]

    # Race filter
    if "race" in df.columns:
        race_options = sorted(df["race"].unique())
        selected_races = st.sidebar.multiselect("Race", race_options, default=race_options)
        df = df[df["race"].isin(selected_races)]

    # Gender filter
    if "gender" in df.columns:
        gender_options = sorted(df["gender"].unique())
        selected_genders = st.sidebar.multiselect("Gender", gender_options, default=gender_options)
        df = df[df["gender"].isin(selected_genders)]

    st.markdown(f"**Showing {len(df):,} records** after filters")

    # ─── Key Statistics ───
    st.subheader("Key Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)

    readmit_rate = df["readmitted"].mean() * 100 if "readmitted" in df.columns else 0
    avg_los = df["time_in_hospital"].mean() if "time_in_hospital" in df.columns else 0
    avg_meds = df["num_medications"].mean() if "num_medications" in df.columns else 0
    avg_diag = df["number_diagnoses"].mean() if "number_diagnoses" in df.columns else 0
    avg_procs = df["num_procedures"].mean() if "num_procedures" in df.columns else 0

    col1.metric("Readmission Rate", f"{readmit_rate:.1f}%")
    col2.metric("Avg Length of Stay", f"{avg_los:.1f} days")
    col3.metric("Avg Medications", f"{avg_meds:.1f}")
    col4.metric("Avg Diagnoses", f"{avg_diag:.1f}")
    col5.metric("Avg Procedures", f"{avg_procs:.1f}")

    st.markdown("---")

    # ─── Target Distribution ───
    st.subheader("Target Variable Distribution")
    col_left, col_right = st.columns(2)

    with col_left:
        if "readmitted" in df.columns:
            target_counts = df["readmitted"].value_counts()
            fig = px.pie(
                values=target_counts.values,
                names=["Not Readmitted (<30d)" if i == 0 else "Readmitted <30d" for i in target_counts.index],
                title="30-Day Readmission Distribution",
                color_discrete_sequence=["#2196F3", "#FF5722"],
                hole=0.4,
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        if "readmitted" in df.columns:
            st.markdown("### Class Balance")
            st.markdown(f"""
            - **Not Readmitted**: {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df)*100:.1f}%)
            - **Readmitted <30d**: {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df)*100:.1f}%)
            - **Imbalance Ratio**: {target_counts.get(0, 1) / max(target_counts.get(1, 1), 1):.1f}:1
            """)
            st.info("The dataset is imbalanced — we use SMOTE, class weights, and AUPRC for evaluation.")

    st.markdown("---")

    # ─── Feature Distributions ───
    st.subheader("Feature Distributions")

    tab1, tab2, tab3 = st.tabs(["Numeric Features", "Categorical Features", "Readmission Patterns"])

    with tab1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "readmitted"]

        selected_numeric = st.selectbox("Select Numeric Feature", numeric_cols)

        if selected_numeric:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Distribution", "By Readmission Status"])

            # Overall distribution
            fig.add_trace(
                go.Histogram(x=df[selected_numeric], name="All", marker_color="#2196F3", opacity=0.7),
                row=1, col=1,
            )

            # By readmission status
            if "readmitted" in df.columns:
                for status, color in [(0, "#2196F3"), (1, "#FF5722")]:
                    subset = df[df["readmitted"] == status]
                    label = "Not Readmitted" if status == 0 else "Readmitted <30d"
                    fig.add_trace(
                        go.Histogram(x=subset[selected_numeric], name=label, marker_color=color, opacity=0.6),
                        row=1, col=2,
                    )

            fig.update_layout(height=400, barmode="overlay")
            st.plotly_chart(fig, use_container_width=True)

            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{df[selected_numeric].mean():.2f}")
            col2.metric("Median", f"{df[selected_numeric].median():.2f}")
            col3.metric("Std Dev", f"{df[selected_numeric].std():.2f}")
            col4.metric("Missing", f"{df[selected_numeric].isna().sum()}")

    with tab2:
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if cat_cols:
            selected_cat = st.selectbox("Select Categorical Feature", cat_cols)

            if selected_cat:
                value_counts = df[selected_cat].value_counts().head(15)
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    labels={"x": selected_cat, "y": "Count"},
                    title=f"Distribution of {selected_cat}",
                    color=value_counts.values,
                    color_continuous_scale="Blues",
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Readmission Rates by Feature")

        analysis_col = st.selectbox(
            "Analyze readmission rates by:",
            ["age", "race", "gender", "admission_type_id", "discharge_disposition_id"],
        )

        if analysis_col in df.columns and "readmitted" in df.columns:
            rates = df.groupby(analysis_col)["readmitted"].agg(["mean", "count"]).reset_index()
            rates.columns = [analysis_col, "Readmission Rate", "Count"]
            rates = rates[rates["Count"] >= 20]  # Filter small groups
            rates["Readmission Rate"] = rates["Readmission Rate"] * 100

            fig = px.bar(
                rates,
                x=analysis_col,
                y="Readmission Rate",
                text="Count",
                title=f"Readmission Rate by {analysis_col}",
                color="Readmission Rate",
                color_continuous_scale="RdYlGn_r",
            )
            fig.update_traces(texttemplate="n=%{text:,}", textposition="outside")
            fig.update_layout(height=450, yaxis_title="Readmission Rate (%)")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ─── Correlation Heatmap ───
    st.subheader("Feature Correlations")
    numeric_for_corr = df.select_dtypes(include=[np.number]).columns.tolist()[:15]
    if len(numeric_for_corr) > 2:
        corr = df[numeric_for_corr].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix",
            aspect="auto",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    # ─── Length of Stay Analysis ───
    st.subheader("Length of Stay vs Readmission")
    if "time_in_hospital" in df.columns and "readmitted" in df.columns:
        los_by_readmit = df.groupby("time_in_hospital")["readmitted"].mean().reset_index()
        los_by_readmit.columns = ["Length of Stay (days)", "Readmission Rate"]

        fig = px.scatter(
            los_by_readmit,
            x="Length of Stay (days)",
            y="Readmission Rate",
            size=df.groupby("time_in_hospital").size().values,
            title="Length of Stay vs 30-Day Readmission Rate",
            trendline="lowess",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
