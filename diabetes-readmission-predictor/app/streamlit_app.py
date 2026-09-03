"""
Diabetes Readmission Predictor — Streamlit Dashboard
=====================================================
Multi-page dashboard for exploring data, model performance,
explainability, and live predictions.
"""
import streamlit as st

st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/hospital-3.png", width=60)
        st.title("Navigation")
        st.markdown("---")

        page = st.radio(
            "Go to",
            [
                "🏠 Overview",
                "📊 EDA Explorer",
                "🤖 Model Performance",
                "🔍 Explainability",
                "🎯 Live Predictor",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown(
            "**Built by [Prayag](https://github.com/)**\n\n"
            "Diabetes 130-US Hospitals Dataset\n"
            "UCI ML Repository"
        )

    # Route to pages
    if page == "🏠 Overview":
        show_overview()
    elif page == "📊 EDA Explorer":
        from app.pages.eda_dashboard import show_eda
        show_eda()
    elif page == "🤖 Model Performance":
        from app.pages.model_performance import show_model_performance
        show_model_performance()
    elif page == "🔍 Explainability":
        from app.pages.explainability import show_explainability
        show_explainability()
    elif page == "🎯 Live Predictor":
        from app.pages.predict import show_predictor
        show_predictor()


def show_overview():
    """Landing page with project overview and key metrics."""
    st.markdown('<p class="main-header">🏥 Diabetes Readmission Predictor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">'
        'Predicting 30-day hospital readmissions for diabetes patients using '
        'machine learning — an end-to-end analytics pipeline.'
        '</p>',
        unsafe_allow_html=True,
    )

    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">101,766</div>
            <div class="metric-label">Patient Encounters</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">50+</div>
            <div class="metric-label">Clinical Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">ML Models Compared</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">SHAP</div>
            <div class="metric-label">Explainability</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Problem statement
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Why Predict Readmissions?")
        st.markdown("""
        Hospital readmissions within 30 days are a critical quality metric in healthcare:

        - **CMS penalties**: The Hospital Readmissions Reduction Program penalizes hospitals
          with excess readmission rates, costing billions annually.
        - **Patient outcomes**: Readmissions indicate gaps in care transitions and can lead
          to worse patient outcomes.
        - **Cost reduction**: Each preventable readmission saves an estimated $15,000–$25,000.

        This project builds a machine learning pipeline to identify high-risk patients
        at discharge, enabling targeted interventions.
        """)

    with col_right:
        st.subheader("Project Pipeline")
        st.markdown("""
        ```
        📥 Data Ingestion
            ↓
        🧹 Cleaning & Validation
            ↓
        🔧 Feature Engineering
            ↓
        🤖 Model Training & Tuning
            ↓
        📊 Evaluation & Fairness
            ↓
        🔍 SHAP Explainability
            ↓
        🚀 Streamlit Dashboard
        ```
        """)

    st.markdown("---")

    # Dataset overview
    st.subheader("Dataset Overview")
    st.markdown("""
    The **Diabetes 130-US Hospitals (1999–2008)** dataset from the UCI ML Repository
    contains clinical records for diabetes patients across 130 US hospitals over 10 years.

    | Attribute | Detail |
    |---|---|
    | **Source** | UCI Machine Learning Repository |
    | **Records** | ~101,766 patient encounters |
    | **Features** | 50+ clinical, demographic, and medication features |
    | **Target** | Readmission within 30 days (binary) |
    | **Privacy** | Fully de-identified, HIPAA compliant |
    """)


if __name__ == "__main__":
    main()
