"""
Tests for the data pipeline.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.clean import _map_icd9_code, _map_age_to_numeric, clean_data
from src.data.features import engineer_features, get_feature_lists, build_preprocessor
from src.utils.config import TARGET_COL, TARGET_MAPPING


class TestICD9Mapping:
    """Tests for ICD-9 code mapping."""

    def test_circulatory_code(self):
        assert _map_icd9_code("428") == "Circulatory"

    def test_diabetes_code(self):
        assert _map_icd9_code("250") == "Endocrine"

    def test_respiratory_code(self):
        assert _map_icd9_code("486") == "Respiratory"

    def test_e_code(self):
        assert _map_icd9_code("E819") == "Injury"

    def test_v_code(self):
        assert _map_icd9_code("V58") == "Supplementary"

    def test_nan_code(self):
        assert _map_icd9_code(np.nan) == "Unknown"

    def test_invalid_code(self):
        assert _map_icd9_code("???") == "Unknown"


class TestAgeMapping:
    """Tests for age bracket to numeric mapping."""

    def test_middle_age(self):
        assert _map_age_to_numeric("[50-60)") == 55

    def test_young(self):
        assert _map_age_to_numeric("[0-10)") == 5

    def test_elderly(self):
        assert _map_age_to_numeric("[90-100)") == 95

    def test_nan(self):
        assert np.isnan(_map_age_to_numeric(np.nan))


class TestDataCleaning:
    """Tests for the data cleaning pipeline."""

    @pytest.fixture
    def sample_raw_data(self):
        """Create a minimal sample dataset for testing."""
        np.random.seed(42)
        n = 100
        data = {
            "encounter_id": range(n),
            "patient_nbr": range(n),
            "race": np.random.choice(["Caucasian", "AfricanAmerican", None], n),
            "gender": np.random.choice(["Male", "Female"], n),
            "age": np.random.choice(["[50-60)", "[60-70)", "[70-80)"], n),
            "weight": [None] * n,  # 100% missing
            "admission_type_id": np.random.choice([1, 2, 3], n),
            "discharge_disposition_id": np.random.choice([1, 2, 3, 6], n),
            "admission_source_id": np.random.choice([1, 7], n),
            "time_in_hospital": np.random.randint(1, 14, n),
            "payer_code": [None] * n,
            "medical_specialty": [None] * n,
            "num_lab_procedures": np.random.randint(1, 100, n),
            "num_procedures": np.random.randint(0, 6, n),
            "num_medications": np.random.randint(1, 40, n),
            "number_outpatient": np.random.randint(0, 5, n),
            "number_emergency": np.random.randint(0, 3, n),
            "number_inpatient": np.random.randint(0, 5, n),
            "number_diagnoses": np.random.randint(1, 16, n),
            "diag_1": np.random.choice(["250", "428", "486"], n),
            "diag_2": np.random.choice(["250", "401", None], n),
            "diag_3": np.random.choice(["250", "496", None], n),
            "max_glu_serum": np.random.choice(["None", ">200", ">300"], n),
            "A1Cresult": np.random.choice(["None", ">7", ">8"], n),
            "metformin": np.random.choice(["No", "Steady", "Up"], n),
            "insulin": np.random.choice(["No", "Steady", "Up", "Down"], n),
            "change": np.random.choice(["No", "Ch"], n),
            "diabetesMed": np.random.choice(["Yes", "No"], n),
            "citoglipton": ["No"] * n,
            "examide": ["No"] * n,
            "readmitted": np.random.choice(["<30", ">30", "NO"], n),
        }
        return pd.DataFrame(data)

    def test_clean_data_returns_dataframe(self, sample_raw_data):
        result = clean_data(sample_raw_data)
        assert isinstance(result, pd.DataFrame)

    def test_clean_data_has_target(self, sample_raw_data):
        result = clean_data(sample_raw_data)
        assert TARGET_COL in result.columns

    def test_target_is_binary(self, sample_raw_data):
        result = clean_data(sample_raw_data)
        assert set(result[TARGET_COL].unique()).issubset({0, 1})

    def test_dropped_columns_removed(self, sample_raw_data):
        result = clean_data(sample_raw_data)
        assert "encounter_id" not in result.columns
        assert "weight" not in result.columns

    def test_icd9_grouped(self, sample_raw_data):
        result = clean_data(sample_raw_data)
        assert "diag_1_group" in result.columns
        assert "diag_1" not in result.columns

    def test_no_missing_in_target(self, sample_raw_data):
        result = clean_data(sample_raw_data)
        assert result[TARGET_COL].isna().sum() == 0

    def test_fewer_rows_after_cleaning(self, sample_raw_data):
        result = clean_data(sample_raw_data)
        assert len(result) <= len(sample_raw_data)


class TestFeatureEngineering:
    """Tests for feature engineering."""

    @pytest.fixture
    def sample_cleaned_data(self):
        """Create sample cleaned data."""
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            "race": np.random.choice(["Caucasian", "AfricanAmerican"], n),
            "gender": np.random.choice(["Male", "Female"], n),
            "age": np.random.choice(["[50-60)", "[60-70)"], n),
            "admission_type_id": np.random.choice(["1", "2"], n).astype(str),
            "discharge_disposition_id": np.random.choice(["1", "2"], n).astype(str),
            "admission_source_id": np.random.choice(["1", "7"], n).astype(str),
            "time_in_hospital": np.random.randint(1, 14, n),
            "num_lab_procedures": np.random.randint(1, 100, n),
            "num_procedures": np.random.randint(0, 6, n),
            "num_medications": np.random.randint(1, 40, n),
            "number_outpatient": np.random.randint(0, 5, n),
            "number_emergency": np.random.randint(0, 3, n),
            "number_inpatient": np.random.randint(0, 5, n),
            "number_diagnoses": np.random.randint(1, 16, n),
            "max_glu_serum": np.random.choice(["None", ">200"], n),
            "A1Cresult": np.random.choice(["None", ">7"], n),
            "change": np.random.choice(["No", "Ch"], n),
            "diabetesMed": np.random.choice(["Yes", "No"], n),
            "metformin": np.random.choice(["No", "Steady", "Up"], n),
            "insulin": np.random.choice(["No", "Steady", "Up"], n),
            "diag_1_group": np.random.choice(["Circulatory", "Endocrine"], n),
            "diag_2_group": np.random.choice(["Circulatory", "Respiratory"], n),
            "diag_3_group": np.random.choice(["Endocrine", "Digestive"], n),
            "readmitted": np.random.choice([0, 1], n),
        })

    def test_engineer_creates_prior_visits(self, sample_cleaned_data):
        result = engineer_features(sample_cleaned_data)
        assert "prior_visits_total" in result.columns

    def test_engineer_creates_medication_changes(self, sample_cleaned_data):
        result = engineer_features(sample_cleaned_data)
        assert "num_medication_changes" in result.columns

    def test_engineer_drops_med_columns(self, sample_cleaned_data):
        result = engineer_features(sample_cleaned_data)
        assert "metformin" not in result.columns
        assert "insulin" not in result.columns

    def test_preprocessor_builds(self, sample_cleaned_data):
        result = engineer_features(sample_cleaned_data)
        num_feats, cat_feats = get_feature_lists(result)
        preprocessor = build_preprocessor(num_feats, cat_feats)
        assert preprocessor is not None


class TestModelPipeline:
    """Tests for model loading and prediction."""

    def test_model_metadata_keys(self):
        """Test that expected metadata keys are defined."""
        from src.utils.config import MODEL_NAMES, PRIMARY_METRIC
        assert len(MODEL_NAMES) > 0
        assert PRIMARY_METRIC == "roc_auc"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
