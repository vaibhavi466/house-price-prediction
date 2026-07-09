# Implementation Plan: Bangalore House Price Predictor (End-to-End)

This plan outlines the design and implementation of a portfolio-grade, interview-defensible Bangalore House Price Predictor, built strictly following the guidelines in [AGENTS.md](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/AGENTS.MD).

The project will be built end-to-end, testing and verifying each phase before moving to the next. We will stop at the marked checkpoints (e.g., if raw data is missing, or at Phase 9 for deployment handoff).

---

## Proposed Phases & Architecture

### Phase 0: Project Setup
- **Objective**: Create folder structure, virtual environment, Makefile, and configure code quality tools (`black`, `flake8`, `isort`, `pytest`).
- **Files Touched/Created**:
  - [NEW] [Makefile](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/Makefile)
  - [NEW] [requirements.txt](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/requirements.txt)
  - [NEW] [.gitignore](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/.gitignore)
  - [NEW] [.flake8](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/.flake8)
  - [NEW] [pyproject.toml](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/pyproject.toml)
  - Placeholder files in `src/`, `server/`, `client/`, `tests/`, `models/`, `docs/`, and `PROGRESS.md`.

### Phase 1: Data Cleaning
- **Objective**: Load `Bengaluru_House_Data.csv`, parse BHK/Sqft, drop unneeded columns (`area_type`, `society`, `balcony`, `availability`), drop/impute nulls, and verify with tests. Run EDA in a notebook.
- **Files Touched/Created**:
  - [NEW] [src/data_cleaning.py](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/src/data_cleaning.py)
  - [NEW] [tests/test_data_cleaning.py](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/tests/test_data_cleaning.py)
  - [NEW] [notebooks/01_eda.ipynb](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/notebooks/01_eda.ipynb)
  - [NEW] [data/raw/SOURCE.md](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/data/raw/SOURCE.md)

### Phase 2: Feature Engineering
- **Objective**: Implement location bucketing (locations with <= 10 listings mapped to "other") strictly on the training split, set up one-hot and target encoding comparisons, and prepare geocoder stretch goals.
- **Files Touched/Created**:
  - [NEW] [src/feature_engineering.py](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/src/feature_engineering.py)
  - [NEW] [tests/test_feature_engineering.py](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/tests/test_feature_engineering.py)
  - [NEW] [docs/encoding_experiment.md](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/docs/encoding_experiment.md)

### Phase 3: Outlier Detection & Removal
- **Objective**: Implement 4 custom outlier removal filters (Sqft-per-BHK, Price-per-Sqft mean/std, BHK price-consistency, bathroom sanity) using training-split statistics only to prevent leakage.
- **Files Touched/Created**:
  - [NEW] [docs/outlier_removal_log.md](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/docs/outlier_removal_log.md)

### Phase 4: Model Building & Comparison
- **Objective**: Build an `sklearn.Pipeline` with `ColumnTransformer`. Grid search and cross-validate Linear Regression, Ridge, Lasso, Random Forest, XGBoost, and LightGBM under both one-hot and target encoding. Track runs in local MLflow. Explain final choice using SHAP.
- **Files Touched/Created**:
  - [NEW] [src/train.py](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/src/train.py)
  - [NEW] [models/comparison_table.md](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/models/comparison_table.md)
  - [NEW] [docs/mlflow_comparison.png](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/docs/mlflow_comparison.png)

### Phase 5: Testing
- **Objective**: Unit tests for core functions and API integration tests. Ensure code coverage is tracked.
- **Files Touched/Created**:
  - [NEW] [tests/test_predict.py](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/tests/test_predict.py)
  - [NEW] [tests/test_api.py](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/tests/test_api.py)

### Phase 6: Backend API (FastAPI)
- **Objective**: Create FastAPI server serving `/health`, `/locations`, and `/predict` (returning predicted price + top 3 SHAP features).
- **Files Touched/Created**:
  - [NEW] [server/main.py](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/server/main.py)
  - [NEW] [server/schemas.py](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/server/schemas.py)

### Phase 7: Frontend
- **Objective**: Write a sleek, vanilla HTML/CSS/JS frontend to interact with the FastAPI backend, demonstrating full-stack skill and rendering price + SHAP explanations.
- **Files Touched/Created**:
  - [NEW] [client/index.html](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/client/index.html)
  - [NEW] [client/style.css](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/client/style.css)
  - [NEW] [client/app.js](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/client/app.js)

### Phase 8: Containerization & CI
- **Objective**: Set up single-stage `python:3.11-slim` Dockerfile with health checks and a GitHub Actions workflow.
- **Files Touched/Created**:
  - [NEW] [Dockerfile](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/Dockerfile)
  - [NEW] [docker-compose.yml](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/docker-compose.yml)
  - [NEW] [.github/workflows/ci.yml](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/.github/workflows/ci.yml)

### Phase 9: Deployment Prep 🛑
- **Objective**: Create `render.yaml` and a precise deployment runbook in `docs/deployment.md`. Stop here for human handoff.
- **Files Touched/Created**:
  - [NEW] [render.yaml](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/render.yaml)
  - [NEW] [docs/deployment.md](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/docs/deployment.md)

### Phase 10: Documentation & Portfolio Polish
- **Objective**: Complete `README.md` containing mermaid diagrams, plots, comparison table, MLflow screenshot, and a real bias/limitations section based on residuals.
- **Files Touched/Created**:
  - [NEW] [README.md](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/README.md)
  - [NEW] [docs/architecture.md](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/docs/architecture.md)
  - [NEW] [docs/verbal_walkthrough.md](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/docs/verbal_walkthrough.md)
  - [NEW] [docs/resume_bullets.md](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/docs/resume_bullets.md)

### Phase 11: Final Verification & Reproducibility Audit
- **Objective**: Clean reconstruct the environment (`make clean && make setup && make all`), run tests, verify Docker image, check numbers across documents.

---

## User Review Required

> [!IMPORTANT]
> The project targets **Python 3.11** for the locked environment. Locally, the system Python version is **3.13.5**.
> I will configure the local virtual environment with the local Python version, but will use `python:3.11-slim` in the Dockerfile to guarantee compatibility with the locked spec. Let me know if you prefer a different setup.

> [!WARNING]
> We will need the raw dataset `Bengaluru_House_Data.csv` inside `data/raw/` to complete Phase 1. I will search for and attempt to fetch the original dataset first. If it cannot be sourced, I will halt and ask you to provide it.

---

## Verification Plan

### Automated Tests
- Run `make test` (triggers `pytest --cov=src --cov=server`) after each code phase.
- Run `make lint` (triggers `black --check .`, `flake8`, `isort --check .`) to verify style constraints.

### Manual Verification
- Swagger UI (`/docs`) verification of prediction responses.
- Browser test using subagent to click through the frontend and verify the SHAP explanations are rendered correctly.
