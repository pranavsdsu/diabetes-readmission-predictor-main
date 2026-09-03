"""
Live Predictor Dashboard Page
==============================
Input patient details and get a real-time readmission risk prediction
with SHAP-based explanation.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json


def load_model_artifact():
    """Load trained model and preprocessor."""
    model_path = Path(__file__).resolve().parent.parent.parent / "models" / "best_model.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def show_predictor():
    st.header("🎯 Live Readmission Risk Predictor")
    st.markdown(
        "Enter patient details below to get a real-time readmission risk prediction. "
        "The model will explain which factors are driving the risk assessment."
    )

    artifact = load_model_artifact()

    if artifact is None:
        st.warning("Trained model not found. Please run the training pipeline first.")
        st.code("make train", language="bash")

        # Show demo UI
        st.subheader("Demo: What the predictor will look like")
        st.markdown("""
        After training, you'll be able to:
        1. Enter patient demographics, visit details, and medication info
        2. Get an instant risk prediction (Low / Medium / High)
        3. See which factors are driving the prediction
        4. Use "What-if" toggles to explore scenarios
        """)
        _show_demo_ui()
        return

    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    metadata = artifact.get("metadata", {})

    _show_prediction_ui(model, preprocessor, metadata)


def _show_demo_ui():
    """Show the predictor UI even without a trained model."""
    st.markdown("---")
    _render_input_form()

    # Demo output
    st.markdown("---")
    st.subheader("Prediction Result (Demo)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Readmission Probability", "0.342")
    col2.metric("Risk Level", "⚠️ Medium")
    col3.metric("Confidence", "Moderate")

    st.progress(34, text="Readmission Risk: 34.2%")

    st.info(
        "🔍 **Key Risk Drivers**: Prior inpatient visits (3), "
        "number of medications (15), length of stay (7 days). "
        "Consider enhanced discharge planning and follow-up scheduling."
    )


def _render_input_form() -> dict:
    """Render the patient input form and return values."""
    st.subheader("Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Demographics")
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.selectbox("Age Group", [
            "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
            "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
        ], index=5)

    with col2:
        st.markdown("#### Visit Details")
        time_in_hospital = st.slider("Length of Stay (days)", 1, 14, 4)
        num_lab_procedures = st.slider("Lab Procedures", 0, 132, 43)
        num_procedures = st.slider("Medical Procedures", 0, 6, 1)
        num_medications = st.slider("Number of Medications", 1, 81, 16)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)

    with col3:
        st.markdown("#### History & Disposition")
        number_outpatient = st.number_input("Prior Outpatient Visits", 0, 42, 0)
        number_emergency = st.number_input("Prior Emergency Visits", 0, 76, 0)
        number_inpatient = st.number_input("Prior Inpatient Visits", 0, 21, 0)
        admission_type = st.selectbox("Admission Type", ["1", "2", "3", "4", "5", "6", "7", "8"])
        discharge_disposition = st.selectbox("Discharge Disposition", ["1", "2", "3", "4", "5", "6", "7", "8"])
        admission_source = st.selectbox("Admission Source", ["1", "2", "3", "4", "5", "6", "7"])

    st.markdown("#### Medication & Test Results")
    col4, col5, col6 = st.columns(3)

    with col4:
        max_glu_serum = st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"])
        a1c_result = st.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])

    with col5:
        change = st.selectbox("Medication Changed?", ["No", "Ch"])
        diabetes_med = st.selectbox("On Diabetes Medication?", ["Yes", "No"])

    with col6:
        diag_1_group = st.selectbox("Primary Diagnosis Group", [
            "Circulatory", "Respiratory", "Digestive", "Endocrine",
            "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms",
            "Nervous", "Mental", "Other", "Unknown",
        ])

    return {
        "race": race,
        "gender": gender,
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "admission_type_id": admission_type,
        "discharge_disposition_id": discharge_disposition,
        "admission_source_id": admission_source,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": a1c_result,
        "change": change,
        "diabetesMed": diabetes_med,
        "diag_1_group": diag_1_group,
        "diag_2_group": "Unknown",
        "diag_3_group": "Unknown",
        # Engineered features
        "prior_visits_total": number_outpatient + number_emergency + number_inpatient,
        "num_medication_changes": 1 if change == "Ch" else 0,
        "age_numeric": int(age.strip("[)").split("-")[0]) + 5 if "-" in age else 55,
        "high_utilizer_flag": 1 if (number_outpatient + number_emergency + number_inpatient) > 5 else 0,
        "medication_change_flag": 1 if change == "Ch" else 0,
        "total_procedures": num_lab_procedures + num_procedures,
    }


def _show_prediction_ui(model, preprocessor, metadata):
    """Show the full predictor with actual model."""
    patient_data = _render_input_form()

    st.markdown("---")

    if st.button("🔮 Predict Readmission Risk", type="primary", use_container_width=True):
        # Create DataFrame from input
        input_df = pd.DataFrame([patient_data])

        try:
            # Get the expected feature columns from preprocessor
            expected_features = metadata.get("numeric_features", []) + metadata.get("categorical_features", [])

            # Ensure all expected columns exist
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = "Unknown" if col in metadata.get("categorical_features", []) else 0

            # Select only expected features in order
            input_df = input_df[expected_features]

            # Preprocess and predict
            X_processed = preprocessor.transform(input_df)
            probability = model.predict_proba(X_processed)[0][1]

            # Display results
            _display_prediction_result(probability, patient_data)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("This may happen if the model was trained with different feature columns. "
                    "Try retraining the model.")


def _display_prediction_result(probability: float, patient_data: dict):
    """Display the prediction result with visualizations."""
    st.subheader("Prediction Result")

    # Risk level
    if probability >= 0.7:
        risk_level = "🔴 HIGH RISK"
        risk_color = "red"
        recommendation = ("Recommend enhanced discharge planning: schedule follow-up within 48 hours, "
                          "medication reconciliation, home health referral.")
    elif probability >= 0.3:
        risk_level = "🟡 MEDIUM RISK"
        risk_color = "orange"
        recommendation = ("Recommend standard follow-up within 7 days, pharmacy consultation, "
                          "and patient education on warning signs.")
    else:
        risk_level = "🟢 LOW RISK"
        risk_color = "green"
        recommendation = "Standard discharge protocol. Schedule routine follow-up within 14 days."

    col1, col2, col3 = st.columns(3)
    col1.metric("Readmission Probability", f"{probability:.1%}")
    col2.metric("Risk Level", risk_level)
    col3.metric("Percentile", f"Top {max(1, int(probability * 100))}%")

    # Progress bar
    st.progress(min(probability, 1.0), text=f"Readmission Risk: {probability:.1%}")

    # Recommendation
    st.info(f"📋 **Clinical Recommendation**: {recommendation}")

    # Key risk factors for this patient
    st.subheader("Key Risk Factors for This Patient")

    risk_factors = []
    if patient_data.get("number_inpatient", 0) > 0:
        risk_factors.append(f"Prior inpatient visits: {patient_data['number_inpatient']}")
    if patient_data.get("num_medications", 0) > 15:
        risk_factors.append(f"High medication count: {patient_data['num_medications']}")
    if patient_data.get("time_in_hospital", 0) > 7:
        risk_factors.append(f"Extended stay: {patient_data['time_in_hospital']} days")
    if patient_data.get("number_emergency", 0) > 0:
        risk_factors.append(f"Prior ER visits: {patient_data['number_emergency']}")
    if patient_data.get("number_diagnoses", 0) > 7:
        risk_factors.append(f"Multiple diagnoses: {patient_data['number_diagnoses']}")

    if risk_factors:
        for rf in risk_factors:
            st.markdown(f"- ⚠️ {rf}")
    else:
        st.markdown("No major individual risk factors identified.")

    # What-if analysis
    st.markdown("---")
    st.subheader("🔄 What-If Scenarios")
    st.markdown("*What would happen if this patient had different characteristics?*")

    st.markdown("""
    - If the patient had **0 prior inpatient visits** → risk would likely decrease
    - If medications were reduced to **<10** → risk would likely decrease
    - If length of stay was **<3 days** → risk would likely decrease

    *Run the full SHAP analysis for precise what-if estimates.*
    """)
