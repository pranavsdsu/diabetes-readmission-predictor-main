# 🚀 Deployment Guide — Streamlit Cloud

## Prerequisites

1. **GitHub account** with the repository pushed
2. **Streamlit Cloud account** (free) at [share.streamlit.io](https://share.streamlit.io)
3. Pipeline has been run locally (`make all`) so model artifacts exist

---

## Step-by-Step Deployment

### Step 1: Prepare Repository for Deployment

Since model files are gitignored (too large), we need to ensure the app
can generate or download them. Two approaches:

#### Option A: Include Small Model in Repo (Recommended for Portfolio)

```bash
# After running the pipeline, check model size
ls -lh models/best_model.pkl

# If under 100MB, force-add to git
git add -f models/best_model.pkl
git add -f models/model_metadata.json
git add -f models/shap_data.pkl
git add -f data/processed/featured_data.csv
git commit -m "Add model artifacts for deployment"
```

#### Option B: Auto-Download on App Start

Add this to the top of `app/streamlit_app.py`:

```python
@st.cache_resource
def ensure_model_exists():
    from pathlib import Path
    model_path = Path("models/best_model.pkl")
    if not model_path.exists():
        import subprocess
        subprocess.run(["python", "run_pipeline.py", "--step", "all"])
    return True
```

### Step 2: Push to GitHub

```bash
# Initialize git (if not already)
cd diabetes-readmission-predictor
git init
git add .
git commit -m "Initial commit: complete diabetes readmission predictor"

# Create GitHub repo and push
# Go to github.com → New Repository → "diabetes-readmission-predictor"
git remote add origin https://github.com/YOUR_USERNAME/diabetes-readmission-predictor.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Fill in:
   - **Repository**: `YOUR_USERNAME/diabetes-readmission-predictor`
   - **Branch**: `main`
   - **Main file path**: `app/streamlit_app.py`
4. Click **"Deploy!"**

### Step 4: Configure App Settings (Optional)

In the Streamlit Cloud dashboard:

- **Secrets**: Add any API keys if needed (not required for this project)
- **Python version**: 3.11 (recommended)
- **Advanced settings**: Set `PYTHONPATH=.` if imports fail

### Step 5: Verify Deployment

- Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`
- Test all 4 pages: Overview, EDA, Model Performance, Predictor
- Check that plots render correctly
- Test the live predictor with sample inputs

---

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError`:
```
# Add to .streamlit/config.toml or app entry:
import sys
sys.path.insert(0, ".")
```

### Memory Limits
Streamlit Cloud free tier has 1GB RAM. If the app crashes:
- Reduce SHAP sample size in `src/models/explain.py` (change `max_samples=500`)
- Use `@st.cache_data` and `@st.cache_resource` aggressively

### Model File Too Large
If `best_model.pkl` > 100MB:
- Use Git LFS: `git lfs track "models/*.pkl"`
- Or host on Google Drive and download on app start

### Slow Cold Starts
First load after inactivity is slow. To mitigate:
- Use `@st.cache_resource` for model loading
- Use `@st.cache_data` for data loading

---

## Post-Deployment Checklist

- [ ] App loads without errors
- [ ] All 4 navigation pages work
- [ ] EDA filters are responsive
- [ ] Model performance metrics display correctly
- [ ] SHAP plots render
- [ ] Live predictor returns results
- [ ] App URL is shareable
- [ ] README updated with live app link
- [ ] GitHub repo has proper description and topics

---

## Adding to Resume / Portfolio

Once deployed, add these to your resume:

**Project Line:**
> **Diabetes Readmission Predictor** | Python, scikit-learn, XGBoost, SHAP, Streamlit
> Built end-to-end ML pipeline predicting 30-day hospital readmissions (AUROC: X.XX)
> with SHAP explainability and fairness analysis across demographic groups.
> [Live Demo](https://your-app.streamlit.app) | [GitHub](https://github.com/you/repo)

**LinkedIn Featured Section:**
- Add the Streamlit app URL as a featured link
- Add the GitHub repo as a featured link
- Write a brief post about the project and tag #DataScience #MachineLearning
