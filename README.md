# Self-Healing ML Pipeline: Observability Phase 1

This repository contains the starting Phase 1 training pipeline for the Observability Platform. The script `train_pipeline.py` will autonomously download data from a Google Drive URL (or load a local CSV), process it through scikit-learn preprocessing steps, train an ensemble model (`RandomForestClassifier` or `XGBoost`), and save the inference-ready pipeline pipeline.

## Prerequisites
- Python 3.9+
- A Virtual Environment (Recommended)

## Setup Guidelines

**1. Create and Activate a Virtual Environment**
Open your terminal (PowerShell or Command Prompt) and run:
```bash
python -m venv venv
.\venv\Scripts\activate
```

**2. Install Dependencies**
Install the necessary ML packges included in `requirements.txt`:
```bash
pip install -r requirements.txt
```

**3. Run the Training Pipeline**
The script uses the `gdown` package under the hood, which correctly handles Google Drive URLs for large files (>200MB) without failing on the standard 'Google virus scan limit' warning.

Run the pipeline:
```bash
python train_pipeline.py --data-source "YOUR_GOOGLE_DRIVE_LINK_HERE" --target-column "TARGET" --model-type "xgb"
```

### CLI Arguments:
- `--data-source`: Provide a local `csv` filepath or a Google Drive link.
- `--target-column`: The name of the label column. For Home Credit Default Risk, it is typically `TARGET`.
- `--model-type`: Option to use `'rf'` (Random Forest) or `'xgb'` (XGBoost). Default is `'rf'`.
- `--model-path`: (Optional) The output filename to store the resulting model. Default is `pipeline_model.joblib`.

**4. Verification**
Once the script completes successfully, a `.joblib` file (by default `pipeline_model.joblib`) is saved in your directory. This file bundles your imputation methods, one-hot encoders, standardizations, and the decision model tree together into a neat sklearn pipeline object.
