from setuptools import setup, find_packages

setup(
    name="diabetes-readmission-predictor",
    version="1.0.0",
    description="End-to-end ML pipeline for predicting 30-day hospital readmissions",
    author="Prayag",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "lightgbm>=4.0",
        "shap>=0.44",
        "streamlit>=1.30",
        "plotly>=5.18",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "optuna>=3.5",
        "joblib>=1.3",
        "requests>=2.31",
        "tqdm>=4.66",
    ],
)
