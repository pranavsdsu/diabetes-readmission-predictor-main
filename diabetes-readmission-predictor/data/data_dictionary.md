# Data Dictionary — Diabetes 130-US Hospitals Dataset

## Source
UCI Machine Learning Repository: [Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/)

**Citation**: Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J., & Clore, J.N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*.

---

## Original Features

| Column | Type | Description |
|---|---|---|
| encounter_id | int | Unique identifier for each encounter (dropped) |
| patient_nbr | int | Unique identifier for each patient (dropped after dedup) |
| race | categorical | Caucasian, AfricanAmerican, Hispanic, Asian, Other, Unknown |
| gender | categorical | Male, Female |
| age | categorical | Age brackets: [0-10), [10-20), ..., [90-100) |
| weight | categorical | Weight brackets (dropped — ~97% missing) |
| admission_type_id | int→str | 1=Emergency, 2=Urgent, 3=Elective, 4=Newborn, 5=Trauma, etc. |
| discharge_disposition_id | int→str | 1=Home, 2=Short-term hospital, 3=SNF, 6=Home health, etc. |
| admission_source_id | int→str | 1=Physician referral, 2=Clinic referral, 7=Emergency room, etc. |
| time_in_hospital | int | Days between admission and discharge (1–14) |
| payer_code | categorical | Insurance type (dropped — ~40% missing) |
| medical_specialty | categorical | Admitting physician specialty (dropped — ~50% missing) |
| num_lab_procedures | int | Number of lab tests performed (1–132) |
| num_procedures | int | Number of non-lab procedures (0–6) |
| num_medications | int | Number of distinct medications administered (1–81) |
| number_outpatient | int | Outpatient visits in prior year (0–42) |
| number_emergency | int | ER visits in prior year (0–76) |
| number_inpatient | int | Inpatient visits in prior year (0–21) |
| number_diagnoses | int | Number of diagnoses entered (1–16) |
| max_glu_serum | categorical | Glucose serum test: None, Norm, >200, >300 |
| A1Cresult | categorical | HbA1c test: None, Norm, >7, >8 |
| diag_1, diag_2, diag_3 | categorical | Primary, secondary, tertiary ICD-9 diagnosis codes |
| [24 medication columns] | categorical | Dosage status: No, Steady, Up, Down |
| change | categorical | Medication change: No, Ch |
| diabetesMed | categorical | Diabetes medication prescribed: Yes, No |
| **readmitted** | **categorical** | **Target: <30, >30, NO** |

---

## Engineered Features

| Column | Type | Description | How Computed |
|---|---|---|---|
| age_numeric | int | Age bracket midpoint | Extracted from age string |
| diag_1_group | categorical | ICD-9 top-level category for primary diagnosis | ICD-9 code mapping |
| diag_2_group | categorical | ICD-9 top-level category for secondary diagnosis | ICD-9 code mapping |
| diag_3_group | categorical | ICD-9 top-level category for tertiary diagnosis | ICD-9 code mapping |
| prior_visits_total | int | Total prior visits (outpatient + emergency + inpatient) | Sum of three visit columns |
| high_utilizer_flag | binary | 1 if prior_visits_total > 5 | Threshold on prior_visits_total |
| num_medication_changes | int | Count of medications with dosage Up or Down | Count across 24 med columns |
| medication_change_flag | binary | 1 if any medication was changed | Derived from num_medication_changes |
| total_procedures | int | Total procedures (lab + non-lab) | Sum of num_lab_procedures + num_procedures |

---

## Target Variable

**Original**: `readmitted` with 3 values: `<30`, `>30`, `NO`

**Binary encoding** (used in modeling):
- `1` = Readmitted within 30 days (`<30`)
- `0` = Not readmitted within 30 days (`>30` or `NO`)

---

## ICD-9 Code Groups

| Group | ICD-9 Range | Examples |
|---|---|---|
| Infectious | 001–139 | Tuberculosis, HIV, septicemia |
| Neoplasms | 140–239 | Cancer, benign tumors |
| Endocrine | 240–279 | Diabetes (250), thyroid disorders |
| Blood | 280–289 | Anemia, coagulation disorders |
| Mental | 290–319 | Depression, anxiety, dementia |
| Nervous | 320–389 | Epilepsy, Parkinson's, neuropathy |
| Circulatory | 390–459 | Heart failure (428), hypertension (401) |
| Respiratory | 460–519 | Pneumonia (486), COPD (496), asthma |
| Digestive | 520–579 | GI bleeding, liver disease |
| Genitourinary | 580–629 | Kidney disease, UTI |
| Musculoskeletal | 710–739 | Osteoarthritis, back disorders |
| Injury | 800–999 / E-codes | Fractures, adverse drug effects |
| Supplementary | V-codes | Aftercare, screening, health exams |

---

## Data Quality Notes

- **Weight**: ~97% missing → dropped
- **Payer code**: ~40% missing → dropped
- **Medical specialty**: ~50% missing → dropped
- **Race**: ~2% missing → filled with "Unknown"
- **Gender**: 3 records with "Unknown/Invalid" → removed
- **Deceased/hospice patients**: Removed (discharge_disposition_id in [11, 13, 14, 19, 20, 21]) — cannot be readmitted
- **Duplicate patients**: First encounter kept, subsequent encounters dropped
