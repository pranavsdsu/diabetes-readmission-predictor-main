# Model Card — Diabetes Readmission Predictor

## Model Details

- **Model type**: Gradient Boosted Decision Trees (XGBoost / LightGBM)
- **Task**: Binary classification (30-day hospital readmission prediction)
- **Training data**: UCI Diabetes 130-US Hospitals (1999–2008)
- **Framework**: scikit-learn, XGBoost, LightGBM
- **Tuning**: Optuna Bayesian optimization
- **Author**: Prayag
- **Date**: 2026

## Intended Use

- **Primary use**: Portfolio demonstration of end-to-end ML pipeline
- **Users**: Hiring managers, technical reviewers, educational purposes
- **NOT intended for**: Clinical decision-making without proper validation

## Training Data

- 101,766 patient encounters from 130 US hospitals
- 50+ clinical, demographic, and medication features
- Binary target: readmitted within 30 days (yes/no)
- Data is from 1999–2008 (significant temporal limitation)

## Evaluation Metrics

| Metric | Value |
|---|---|
| AUROC | Run pipeline to populate |
| AUPRC | Run pipeline to populate |
| F1 Score | Run pipeline to populate |
| Brier Score | Run pipeline to populate |

## Ethical Considerations

- **Fairness**: Analysis conducted across race and gender subgroups
- **Privacy**: Dataset is fully de-identified, no HIPAA concerns
- **Bias**: Historical data may reflect systemic biases in healthcare
- **Limitations**: Model trained on data >15 years old; clinical practices have changed

## Caveats and Recommendations

1. This model should NOT be used for clinical decisions without external validation
2. The dataset's age means it may not reflect current readmission patterns
3. Important features (lab results, vitals, social determinants) are not available
4. Fairness metrics should be monitored and addressed before any deployment
