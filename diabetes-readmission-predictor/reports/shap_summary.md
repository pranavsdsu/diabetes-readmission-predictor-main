## Top Risk Factors for 30-Day Hospital Readmission

Based on SHAP analysis of the best-performing model, the following
factors have the greatest influence on predicting whether a diabetes
patient will be readmitted within 30 days:

1. **Number Inpatient** (importance: 0.2344)
2. **Discharge Disposition Id 1** (importance: 0.1759)
3. **Prior Visits Total** (importance: 0.0830)
4. **Age Numeric** (importance: 0.0551)
5. **Number Diagnoses** (importance: 0.0541)
6. **Diag 1 Group Circulatory** (importance: 0.0521)
7. **Time In Hospital** (importance: 0.0452)
8. **Diabetesmed No** (importance: 0.0305)
9. **Num Medications** (importance: 0.0292)
10. **Num Lab Procedures** (importance: 0.0288)

### Clinical Implications

These features suggest that readmission risk is primarily driven by:
- **Visit history** — patients with more prior inpatient/emergency visits are at higher risk
- **Medication complexity** — number of medications and medication changes signal severity
- **Length of stay** — longer hospitalizations correlate with higher readmission probability
- **Diagnosis patterns** — certain primary diagnoses (circulatory, respiratory) increase risk