.PHONY: help setup data train evaluate explain all test lint app clean

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║  Diabetes Readmission Predictor — Makefile Commands     ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  make setup     — Install dependencies                  ║"
	@echo "║  make data      — Run data pipeline (download+clean)    ║"
	@echo "║  make train     — Train and tune models                 ║"
	@echo "║  make evaluate  — Run evaluation & fairness analysis    ║"
	@echo "║  make explain   — Run SHAP explainability               ║"
	@echo "║  make all       — Run complete pipeline end-to-end      ║"
	@echo "║  make test      — Run unit tests                        ║"
	@echo "║  make lint      — Run linter (flake8)                   ║"
	@echo "║  make app       — Launch Streamlit dashboard            ║"
	@echo "║  make clean     — Remove generated files                ║"
	@echo "╚══════════════════════════════════════════════════════════╝"

# Install dependencies
setup:
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Run data pipeline
data:
	python run_pipeline.py --step data
	@echo "✅ Data pipeline complete"

# Train models
train:
	python run_pipeline.py --step train
	@echo "✅ Model training complete"

# Evaluate models
evaluate:
	python run_pipeline.py --step evaluate
	@echo "✅ Evaluation complete"

# Run SHAP explainability
explain:
	python run_pipeline.py --step explain
	@echo "✅ Explainability analysis complete"

# Run complete pipeline
all:
	python run_pipeline.py --step all
	@echo "✅ Full pipeline complete"

# Run tests
test:
	python -m pytest tests/ -v --tb=short

# Lint code
lint:
	flake8 src/ --max-line-length=120 --ignore=E501,W503,E203

# Launch Streamlit dashboard
app:
	streamlit run app/streamlit_app.py --server.port 8501

# Clean generated files
clean:
	rm -rf data/raw/*.csv data/raw/*.zip
	rm -rf data/processed/*.csv
	rm -rf data/*.db
	rm -rf models/*.pkl models/*.json
	rm -rf reports/figures/*.png
	rm -rf __pycache__ src/__pycache__ src/**/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	@echo "🧹 Cleaned generated files"
