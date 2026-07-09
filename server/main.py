import pandas as pd
import numpy as np
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from server.schemas import HousePricePredictionRequest, HousePricePredictionResponse, SHAPFeatureContribution
from src.config import MODEL_PIPELINE_FILE

# Global variables for loaded pipeline, explainer and location list
model_pipeline = None
shap_explainer = None
valid_locations = []
clean_feature_names = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager to load the model pipeline, initialize
    the SHAP explainer, and fetch the valid location list once at startup.
    """
    global model_pipeline, shap_explainer, valid_locations, clean_feature_names
    
    print("Loading serialized pipeline at startup...")
    if not MODEL_PIPELINE_FILE.exists():
        raise RuntimeError(f"Pipeline joblib not found at {MODEL_PIPELINE_FILE}. Train the model first.")
        
    model_pipeline = joblib.load(MODEL_PIPELINE_FILE)
    
    # Extract locations from LocationBucketTransformer
    try:
        preprocessor_pipeline = model_pipeline.named_steps["preprocessor"]
        col_transformer = preprocessor_pipeline.named_steps["preprocessor"]
        cat_transformer = col_transformer.named_transformers_["cat"]
        bucket_transformer = cat_transformer.named_steps["bucket"]
        
        # Sort and store the frequent locations list
        valid_locations = sorted(list(bucket_transformer.frequent_locations_))
    except Exception as e:
        print(f"Warning: Could not extract locations from bucket transformer: {e}")
        valid_locations = []
        
    # Get feature names out for SHAP contributions
    try:
        preprocessor = model_pipeline.named_steps["preprocessor"]
        raw_feature_names = preprocessor.get_feature_names_out()
        clean_feature_names = []
        for name in raw_feature_names:
            name = name.replace("cat__ohe__", "").replace("cat__bucket__", "").replace("num__", "")
            clean_feature_names.append(name)
    except Exception as e:
        print(f"Warning: Could not extract clean feature names: {e}")
        clean_feature_names = []

    # Initialize SHAP explainer
    try:
        regressor = model_pipeline.named_steps["model"]
        shap_explainer = shap.TreeExplainer(regressor)
    except Exception as e:
        print(f"Warning: Could not initialize SHAP explainer: {e}")
        shap_explainer = None
        
    yield
    print("Shutting down API server...")

app = FastAPI(
    title="Bangalore House Price Predictor API",
    description="FastAPI endpoint for house price predictions with model explainability.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Monitoring"])
def health_check():
    """
    Simple liveness check for the container/deployment.
    """
    return {"status": "healthy"}

@app.get("/locations", tags=["Metadata"])
def get_locations():
    """
    Returns the list of locations the model was trained on.
    """
    return {"locations": valid_locations}

@app.post("/predict", response_model=HousePricePredictionResponse, tags=["Inference"])
def predict(request: HousePricePredictionRequest):
    """
    Predicts house price and calculates the top 3 contributing factors via SHAP.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
        
    try:
        # 1. Create input DataFrame matching expected schema
        input_df = pd.DataFrame([{
            "location": request.location,
            "total_sqft": request.sqft,
            "bath": request.bath,
            "bhk": request.bhk
        }])
        
        # 2. Predict price
        pred_price = float(model_pipeline.predict(input_df)[0])
        
        # 3. Calculate SHAP contributions if explainer is available
        contributions = []
        if shap_explainer is not None and len(clean_feature_names) > 0:
            preprocessor = model_pipeline.named_steps["preprocessor"]
            input_trans = preprocessor.transform(input_df)
            
            # Convert to DataFrame
            input_trans_df = pd.DataFrame(input_trans, columns=clean_feature_names)
            
            # Compute SHAP values for this instance
            shaps = shap_explainer.shap_values(input_trans_df)
            
            # Handle list/numpy representations depending on shap version
            if isinstance(shaps, list):
                # For classification, but here we expect single target regression
                shap_contribs = shaps[0]
            else:
                shap_contribs = shaps[0]
                
            # Pair feature names with their shap values
            feature_contribs = []
            for name, val in zip(clean_feature_names, shap_contribs):
                # We skip features that have 0 contribution (e.g. unselected OHE locations)
                if abs(val) > 1e-5:
                    feature_contribs.append(
                        SHAPFeatureContribution(feature=name, contribution=float(val))
                    )
            
            # Sort by absolute contribution in descending order and keep top 3
            feature_contribs.sort(key=lambda x: abs(x.contribution), reverse=True)
            contributions = feature_contribs[:3]
            
        return HousePricePredictionResponse(
            predicted_price=max(0.0, pred_price),
            shap_contributions=contributions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
