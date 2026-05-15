# Data Directory

This folder holds all dataset artefacts used by the self-healing ML pipeline.

## Files

| File | Rows | Description |
|------|------|-------------|
| `raw/pima_raw.csv` | 768 | Original Pima Indians Diabetes dataset (untouched) |
| `historical_data.csv` | 614 | Cleaned, imputed data used for **model training** |
| `current_data.csv` | 154 | Cleaned data with simulated drift used for **drift detection** |

## Dataset – Pima Indians Diabetes

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository (via J. Brownlee's mirror) |
| Task | Binary classification – predict diabetes onset (Outcome: 0/1) |
| Features | 8 numeric clinical measurements |
| Rows | 768 |
| Missing values | Physiological zeros (Glucose, BloodPressure, SkinThickness, Insulin, BMI) encoded as 0 in the original; replaced with column medians |

## Feature Glossary

| Feature | Unit | Notes |
|---------|------|-------|
| Pregnancies | count | Number of pregnancies |
| Glucose | mg/dL | 2-hour plasma glucose concentration |
| BloodPressure | mmHg | Diastolic blood pressure |
| SkinThickness | mm | Triceps skinfold thickness |
| Insulin | μIU/mL | 2-hour serum insulin |
| BMI | kg/m² | Body mass index |
| DiabetesPedigreeFunction | — | Genetic risk score |
| Age | years | Patient age |
| Outcome | 0 / 1 | **Target** – 0 = no diabetes, 1 = diabetes |

## Regenerating the Data

```bash
python scripts/prepare_data.py
```
