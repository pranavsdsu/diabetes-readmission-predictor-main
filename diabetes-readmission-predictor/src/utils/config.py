"""
Centralized configuration for the Diabetes Readmission Predictor project.
All paths, parameters, and constants are defined here.
"""
import os
from pathlib import Path

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
DATASET_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130+us+hospitals+for+years+1999+2008.zip"
RAW_CSV_NAME = "diabetic_data.csv"
IDS_MAPPING_NAME = "IDs_mapping.csv"
SQLITE_DB_PATH = DATA_DIR / "diabetes.db"

# ──────────────────────────────────────────────
# DATA PROCESSING
# ──────────────────────────────────────────────
# Columns to drop (too many missing values or not useful)
COLUMNS_TO_DROP = [
    "encounter_id",
    "patient_nbr",
    "weight",           # ~97% missing
    "payer_code",       # ~40% missing, not clinically relevant
    "medical_specialty", # ~50% missing
    "citoglipton",      # near-zero variance
    "examide",          # near-zero variance
]

# Missing value marker in dataset
MISSING_MARKER = "?"

# Target variable
TARGET_COL = "readmitted"

# Binary target: 1 = readmitted within 30 days, 0 = otherwise
TARGET_MAPPING = {
    "<30": 1,
    ">30": 0,
    "NO": 0,
}

# ICD-9 code groupings (top-level categories)
ICD9_GROUPS = {
    "Infectious": (1, 139),
    "Neoplasms": (140, 239),
    "Endocrine": (240, 279),
    "Blood": (280, 289),
    "Mental": (290, 319),
    "Nervous": (320, 389),
    "Circulatory": (390, 459),
    "Respiratory": (460, 519),
    "Digestive": (520, 579),
    "Genitourinary": (580, 629),
    "Pregnancy": (630, 679),
    "Skin": (680, 709),
    "Musculoskeletal": (710, 739),
    "Congenital": (740, 759),
    "Perinatal": (760, 779),
    "Ill-defined": (780, 799),
    "Injury": (800, 999),
}

# ──────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────
NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "prior_visits_total",        # engineered
    "num_medication_changes",    # engineered
]

CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
    "diag_1_group",              # engineered
    "diag_2_group",              # engineered
    "diag_3_group",              # engineered
]

# Medication columns (used for feature engineering)
MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# ──────────────────────────────────────────────
# MODELING
# ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
OPTUNA_N_TRIALS = 50

# Model registry
MODEL_NAMES = [
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
    "LightGBM",
]

# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────
PRIMARY_METRIC = "roc_auc"
FAIRNESS_ATTRIBUTES = ["race", "gender"]
CLINICAL_THRESHOLD = 0.5  # default, will be optimized

# ──────────────────────────────────────────────
# DASHBOARD
# ──────────────────────────────────────────────
STREAMLIT_PAGE_TITLE = "Diabetes Readmission Predictor"
STREAMLIT_PAGE_ICON = "🏥"
STREAMLIT_LAYOUT = "wide"
