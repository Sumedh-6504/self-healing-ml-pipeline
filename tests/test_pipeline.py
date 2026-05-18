import pandas as pd
from unittest.mock import patch
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src.train_pipeline import build_pipeline, load_data, train_and_save_model


def test_build_pipeline_rf():
    """Test pipeline creation with Random Forest."""
    numeric_features = ["age", "income"]
    categorical_features = ["gender", "city"]

    pipeline = build_pipeline(numeric_features, categorical_features, model_type="rf")

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["classifier"], RandomForestClassifier)


def test_build_pipeline_xgb():
    """Test pipeline creation with XGBoost."""
    numeric_features = ["age", "income"]
    categorical_features = ["gender", "city"]

    pipeline = build_pipeline(numeric_features, categorical_features, model_type="xgb")

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["classifier"], xgb.XGBClassifier)


@patch("src.train_pipeline.pd.read_csv")
def test_load_data_local(mock_read_csv):
    """Test data loading from a local file."""
    mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    mock_read_csv.return_value = mock_df

    df = load_data("dummy_path.csv")

    mock_read_csv.assert_called_once_with("dummy_path.csv")
    pd.testing.assert_frame_equal(df, mock_df)


@patch("src.train_pipeline.gdown.download")
@patch("src.train_pipeline.pd.read_csv")
def test_load_data_gdrive(mock_read_csv, mock_download):
    """Test data loading from a Google Drive URL."""
    mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    mock_read_csv.return_value = mock_df

    # Fake Google Drive URL with ID
    url = "https://drive.google.com/file/d/1ABC123xyz/view"
    df = load_data(url)

    mock_download.assert_called_once_with(
        id="1ABC123xyz", output="downloaded_dataset.csv", quiet=False
    )
    mock_read_csv.assert_called_once_with("downloaded_dataset.csv")
    pd.testing.assert_frame_equal(df, mock_df)


@patch("src.train_pipeline.joblib.dump")
@patch("src.train_pipeline.load_data")
def test_train_and_save_model(mock_load_data, mock_joblib_dump):
    """Test the full training orchestrator."""
    # Create dummy data with enough rows for stratified split (test_size=0.2 means at least 10 rows for 2 classes)
    mock_df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "income": [
                50000,
                60000,
                70000,
                80000,
                90000,
                100000,
                110000,
                120000,
                130000,
                140000,
            ],
            "gender": ["M", "F", "F", "M", "M", "F", "M", "F", "M", "F"],
            "TARGET": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    mock_load_data.return_value = mock_df

    train_and_save_model("dummy_source", "TARGET", "dummy_model.joblib", "rf")

    # Verify load_data was called
    mock_load_data.assert_called_once_with("dummy_source")

    # Verify model saving was called
    mock_joblib_dump.assert_called_once()
    saved_pipeline = mock_joblib_dump.call_args[0][0]
    saved_path = mock_joblib_dump.call_args[0][1]

    assert isinstance(saved_pipeline, Pipeline)
    assert saved_path == "dummy_model.joblib"
