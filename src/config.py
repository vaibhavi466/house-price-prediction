from pathlib import Path

# Random seed
RANDOM_STATE = 42

# Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Dataset file name
RAW_DATA_FILE = RAW_DATA_DIR / "Bengaluru_House_Data.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_house_data.csv"

# Model file name
MODEL_PIPELINE_FILE = MODELS_DIR / "model_pipeline.joblib"
