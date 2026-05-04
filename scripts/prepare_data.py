"""
Dataset Selection & EDA – Data Preparation Script
==================================================
Dataset : Pima Indians Diabetes Dataset (UCI / Kaggle)
Task    : Binary classification (Outcome: 0 = no diabetes, 1 = diabetes)
Outputs :
  data/historical_data.csv – 80 % of the cleaned data (used for model training)
  data/current_data.csv    – 20 % of the cleaned data with slight distribution
                             shifts introduced to simulate real-world data drift
"""

import os
import io
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 1.  Load the raw dataset
# ---------------------------------------------------------------------------
COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

RAW_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
)
RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "pima_raw.csv")

print("Loading dataset ...")
try:
    # Try network download first
    with urllib.request.urlopen(RAW_URL, timeout=15) as resp:
        raw_csv = resp.read().decode("utf-8")
    df_raw = pd.read_csv(io.StringIO(raw_csv), header=None, names=COLUMNS)
    print(f"  Downloaded from URL  ({len(df_raw)} rows).")
except Exception:
    # Fallback: use sklearn / synthetic data if network is unavailable
    print("  Network unavailable – generating synthetic data from sklearn.")
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=768,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        random_state=42,
        class_sep=0.8,
    )
    df_raw = pd.DataFrame(X, columns=COLUMNS[:-1])
    # Re-scale standardised features to realistic clinical ranges matching Pima:
    # form: (std_value * scale + mean).clip(clinical_min, clinical_max)
    df_raw["Pregnancies"] = (df_raw["Pregnancies"] * 1.5 + 4).clip(0, 17).round(0)
    df_raw["Glucose"] = (df_raw["Glucose"] * 30 + 120).clip(0, 199)         # 0–199 mg/dL
    df_raw["BloodPressure"] = (df_raw["BloodPressure"] * 12 + 70).clip(0, 122)  # 0–122 mmHg
    df_raw["SkinThickness"] = (df_raw["SkinThickness"] * 15 + 28).clip(0, 99)   # 0–99 mm
    df_raw["Insulin"] = (df_raw["Insulin"] * 80 + 100).clip(0, 846)         # 0–846 μIU/mL
    df_raw["BMI"] = (df_raw["BMI"] * 7 + 32).clip(0, 67)                    # 0–67 kg/m²
    df_raw["DiabetesPedigreeFunction"] = (df_raw["DiabetesPedigreeFunction"] * 0.3 + 0.47).clip(
        0.078, 2.42                                                          # 0.078–2.42 (score)
    )
    df_raw["Age"] = (df_raw["Age"] * 10 + 33).clip(21, 81).round(0)         # 21–81 years
    df_raw["Outcome"] = y
    print(f"  Generated synthetic dataset ({len(df_raw)} rows).")

# ---------------------------------------------------------------------------
# 2.  Persist the raw file so it can be inspected / re-processed later
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
df_raw.to_csv(RAW_PATH, index=False)
print(f"  Raw file saved → {RAW_PATH}")

# ---------------------------------------------------------------------------
# 3.  Identify and handle missing values
#     In the Pima dataset, physiological zeros are biologically impossible and
#     encode missing measurements:
#       Glucose, BloodPressure, SkinThickness, Insulin, BMI → 0 means NaN
# ---------------------------------------------------------------------------
ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df = df_raw.copy()
df[ZERO_AS_NAN_COLS] = df[ZERO_AS_NAN_COLS].replace(0, np.nan)

print("\nMissing value counts after encoding zeros as NaN:")
missing = df.isnull().sum()
print(missing[missing > 0].to_string())

# Impute with the column median (robust to outliers and skewed distributions)
for col in ZERO_AS_NAN_COLS:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"  Imputed '{col}' missing values with median = {median_val:.2f}")

assert df.isnull().sum().sum() == 0, "Dataset still contains NaN values after imputation!"
print("\nAll missing values resolved. Dataset shape:", df.shape)

# ---------------------------------------------------------------------------
# 4.  Feature engineering – keep minimal; the raw features are already clean
#     (Pregnancies and Age are integers by definition)
# ---------------------------------------------------------------------------
int_cols = ["Pregnancies", "Age", "Outcome"]
df[int_cols] = df[int_cols].astype(int)

# ---------------------------------------------------------------------------
# 5.  Train / test split
#     historical_data : 80 % – used for model training / EDA
#     current_data    : 20 % – used for drift detection later
#     Stratify by Outcome so both splits share the same class balance.
# ---------------------------------------------------------------------------
historical_df, current_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Outcome"],
)

# ---------------------------------------------------------------------------
# 6.  Introduce a subtle distribution shift in current_data to simulate
#     the kind of drift a production model would encounter over time.
#     • Glucose values shifted up by ~5 % (dietary changes in population)
#     • BMI shifted up by ~3 % (demographic trend)
#     These shifts are realistic and small enough not to be immediately
#     obvious but detectable by statistical drift tests.
# ---------------------------------------------------------------------------
DRIFT_ROW_FRACTION = 0.50    # fraction of current_data rows that receive the shift
GLUCOSE_DRIFT_FACTOR = 1.05  # +5 % – simulates dietary changes in the population
BMI_DRIFT_FACTOR = 1.03      # +3 % – simulates a demographic BMI trend
GLUCOSE_MAX = 199.0          # clinical upper bound for the dataset
BMI_MAX = 67.0               # clinical upper bound for the dataset

rng = np.random.default_rng(seed=0)

drift_mask = rng.random(len(current_df)) < DRIFT_ROW_FRACTION
current_df = current_df.copy()
current_df.loc[drift_mask, "Glucose"] = (
    current_df.loc[drift_mask, "Glucose"] * GLUCOSE_DRIFT_FACTOR
).clip(upper=GLUCOSE_MAX)
current_df.loc[drift_mask, "BMI"] = (
    current_df.loc[drift_mask, "BMI"] * BMI_DRIFT_FACTOR
).clip(upper=BMI_MAX)
# Round to same precision as source data
current_df["Glucose"] = current_df["Glucose"].round(1)
current_df["BMI"] = current_df["BMI"].round(1)

# ---------------------------------------------------------------------------
# 7.  Save the output files
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

historical_path = os.path.join(DATA_DIR, "historical_data.csv")
current_path = os.path.join(DATA_DIR, "current_data.csv")

historical_df.to_csv(historical_path, index=False)
current_df.to_csv(current_path, index=False)

print(f"\nOutput files written:")
print(f"  historical_data.csv : {len(historical_df)} rows → {historical_path}")
print(f"  current_data.csv    : {len(current_df)} rows → {current_path}")

# ---------------------------------------------------------------------------
# 8.  Quick sanity summary
# ---------------------------------------------------------------------------
print("\n--- historical_data.csv summary ---")
print(historical_df.describe().to_string())

print("\n--- Outcome distribution (historical) ---")
print(historical_df["Outcome"].value_counts().to_string())

print("\n--- Outcome distribution (current) ---")
print(current_df["Outcome"].value_counts().to_string())

print("\nData preparation complete ✓")
