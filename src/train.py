import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE
from src.pipeline import get_preprocessing_pipeline

def evaluate_predictions(y_true, y_pred):
    """
    Computes RMSE, MAE, and R^2 metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def train_and_evaluate(model, encoding_type, X_train, y_train, X_test, y_test, threshold=10):
    """
    Fits the preprocessing + estimator pipeline and returns prediction metrics on test set.
    """
    # Create the preprocessing pipeline
    preprocessor = get_preprocessing_pipeline(encoding_type=encoding_type, threshold=threshold)
    
    # Create the full pipeline
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # Fit the model
    full_pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = full_pipeline.predict(X_test)
    
    # Evaluate
    metrics = evaluate_predictions(y_test, y_pred)
    
    return full_pipeline, metrics
