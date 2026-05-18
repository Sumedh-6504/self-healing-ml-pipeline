import argparse
import logging
import re
import os

import gdown
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(data_source: str) -> pd.DataFrame:
    """
    Loads data from a local file or downwards from Google Drive.
    """
    input_path = data_source
    if data_source.startswith("http"):
        logger.info(f"Downloading dataset from URL: {data_source}")
        output_file = "downloaded_dataset.csv"
        
        # Extract ID if present in the URL
        match = re.search(r'/d/([a-zA-Z0-9_-]+)', data_source)
        if match:
            file_id = match.group(1)
            gdown.download(id=file_id, output=output_file, quiet=False)
        else:
            gdown.download(url=data_source, output=output_file, quiet=False)
            
        input_path = output_file
        
    logger.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Dataset loaded with shape: {df.shape}")
    return df


def build_pipeline(numeric_features, categorical_features, model_type='rf') -> Pipeline:
    """Builds the scikit-learn training pipeline."""
    logger.info(f"Building ML pipeline with {model_type}...")
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Choose model
    if model_type.lower() == 'xgb':
        classifier = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss')
    else:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline


def train_and_save_model(data_source: str, target_column: str, model_path: str, model_type: str):
    """Main training orchestrator."""
    # 1. Load Data
    df = load_data(data_source)
    
    # Ensure target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        
    # Drop rows where target is missing
    df = df.dropna(subset=[target_column])
    
    # 2. Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Using stratify ensures balanced train/test split for imbalanced datasets like Home Credit Risk
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # 3. Identify numeric and categorical columns
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    logger.info(f"Found {len(numeric_features)} numeric and {len(categorical_features)} categorical features.")

    # 4. Build Pipeline
    pipeline = build_pipeline(numeric_features, categorical_features, model_type)

    # 5. Train Model
    logger.info("Training model...")
    pipeline.fit(X_train, y_train)
    
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    logger.info(f"Model trained. Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")

    # 6. Save Model
    logger.info(f"Saving pipeline to {model_path}...")
    joblib.dump(pipeline, model_path)
    logger.info("Pipeline saved successfully. Observability Phase 1 Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: ML Model Training Pipeline")
    parser.add_argument("--data-source", type=str, required=True, help="Local path or Google Drive URL to the CSV dataset")
    parser.add_argument("--target-column", type=str, default="TARGET", help="Target column name (e.g., TARGET for Home Credit risk)")
    parser.add_argument("--model-path", type=str, default="pipeline_model.joblib", help="Path to save the joblib pipeline")
    parser.add_argument("--model-type", type=str, choices=['rf', 'xgb'], default='rf', help="Model type to train: 'rf' for Random Forest, 'xgb' for XGBoost")
    
    args = parser.parse_args()
    
    train_and_save_model(args.data_source, args.target_column, args.model_path, args.model_type)
