# Architecture

This document describes the end-to-end architecture of the Bangalore House Price Predictor.

## System Diagram

```mermaid
flowchart TD
    A[Raw CSV\nBengaluru_House_Data.csv] --> B[Data Cleaning\nsrc/data_cleaning.py]
    B --> C[Feature Engineering\nsrc/feature_engineering.py]
    C --> D[Outlier Removal\nTraining split only]
    D --> E[Model Training + Comparison\nsrc/train.py + MLflow]
    E --> F[Serialized Pipeline\nmodels/model_pipeline.joblib]
    F --> G[FastAPI Server\nserver/main.py]
    G --> H[GET /locations]
    G --> I[POST /predict\n+ SHAP contributions]
    G --> J[GET /health]
    G --> K[Static Frontend\nclient/index.html]
    K --> L[Browser UI\nForm + Results + SHAP Drivers]
    L -->|fetch POST /predict| G
```

## Component Responsibilities

| Component | File(s) | Responsibility |
|:---|:---|:---|
| Data Cleaning | `src/data_cleaning.py` | Parse, clean, validate raw CSV. Drop bad rows. |
| Feature Engineering | `src/feature_engineering.py` | Location bucketing transformer (leakage-safe). |
| Outlier Removal | `src/data_cleaning.py` | 4-rule domain filter on training split only. |
| ML Pipeline | `src/pipeline.py` | ColumnTransformer: scale numerics, encode location. |
| Model Training | `src/train.py` | GridSearchCV × 6 models × 2 encodings. MLflow tracking. |
| Prediction | `src/predict.py` | Load serialized pipeline, run inference. |
| API Server | `server/main.py` | FastAPI: /health, /locations, /predict + SHAP. |
| Frontend | `client/` | Vanilla HTML/CSS/JS SPA, fetches /locations and /predict. |
| Containerization | `Dockerfile` | python:3.11-slim, non-root user, healthcheck. |
| CI | `.github/workflows/ci.yml` | Lint + test + Docker build + smoke test on every push. |
