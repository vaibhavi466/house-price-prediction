import joblib
import pandas as pd

from src.config import MODEL_PIPELINE_FILE

# Global model cache to avoid reloading model per request
_MODEL_PIPELINE = None


def get_model_pipeline():
    """
    Loads and returns the serialized scikit-learn pipeline, caching it for subsequent calls.
    """
    global _MODEL_PIPELINE
    if _MODEL_PIPELINE is None:
        if not MODEL_PIPELINE_FILE.exists():
            raise FileNotFoundError(
                f"Model pipeline file not found at {MODEL_PIPELINE_FILE}. Please train the model first."
            )
        _MODEL_PIPELINE = joblib.load(MODEL_PIPELINE_FILE)
    return _MODEL_PIPELINE


def predict_price(location: str, total_sqft: float, bath: int, bhk: int) -> float:
    """
    Predicts the house price in Lakhs INR based on inputs.
    """
    pipeline = get_model_pipeline()

    # Construct a DataFrame matching the model's feature structure
    input_data = pd.DataFrame(
        [
            {
                "location": location,
                "total_sqft": float(total_sqft),
                "bath": int(bath),
                "bhk": int(bhk),
            }
        ]
    )

    prediction = pipeline.predict(input_data)

    # Ensure prediction is a float and positive
    predicted_val = float(prediction[0])
    return max(0.0, predicted_val)
