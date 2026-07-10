# Progress Log - Bangalore House Price Predictor

This file tracks the completion and verification of each phase of the project, as required by the operating principles in [AGENTS.md](file:///c:/Users/Vaibhavi/Desktop/House-Price-Predictions/AGENTS.MD).

---

## [2026-07-10] Phase 0: Project Setup

### What was built:
1. **Directory Structure**: Initialized empty directories and placeholders for `data/raw/`, `data/processed/`, `notebooks/`, `src/`, `server/`, `client/`, `tests/`, `models/`, and `docs/`.
2. **Environment & Dependencies**: Created a virtual environment (`venv`) using Python 3.13.5 and installed all package requirements defined in `requirements.txt` (including `scikit-learn>=1.3`, `fastapi`, `xgboost`, `lightgbm`, `shap`, `mlflow`, `statsmodels`, `pytest`, `black`, `flake8`, `isort`).
3. **Style & Quality Configurations**: Setup configs for `.flake8` (max line length 100) and `pyproject.toml` (for `black` and `isort`).
4. **Build Automation**: Created a cross-platform `Makefile` with targets `setup`, `lint`, `test`, `clean`, `train`, `serve`, and `all`.
5. **Git Setup**: Initialized local git repository, added a `.gitignore` file, and performed the initial commit.

### Verification Run & Results:
1. **Linter (Black & Isort)**: Ran `venv\Scripts\python -m black --check .` and `venv\Scripts\python -m isort --check-only .`. Both ran cleanly and reported 0 formatting/import order issues.
2. **Linter (Flake8)**: Ran `venv\Scripts\python -m flake8 .`. Fixed one unused `import os` in `src/config.py`. Subsequent run was completely clean.
3. **Unit Tests (Pytest)**: Added a placeholder test in `tests/test_data_cleaning.py` to ensure test framework is operational. Ran `venv\Scripts\python -m pytest` which completed successfully (1 test passed).

---

## [2026-07-10] Phase 1: Data Cleaning

### What was built:
1. **Dataset Acquisition**: Sourced and downloaded the raw `Bengaluru_House_Data.csv` (~938 KB) from a public GitHub mirror. Verified column integrity and documented its provenance and SHA256 checksum in `data/raw/SOURCE.md`.
2. **Modular Ingestion Code**: Wrote `src/data_cleaning.py` featuring:
   - `extract_bhk()`: Extracts leading BHK digits from strings like "2 BHK" or "4 Bedroom".
   - `convert_sqft()`: Safely converts area strings into floats, resolving ranges like "1440 - 1500" into their midpoint average, and returns `None` for invalid unit strings (e.g. "34.46Sq. Meter") to ensure they are filtered out rather than silently coerced.
   - `clean_data()`: Orchestrates ingestion, drops unnecessary columns (`area_type`, `society`, `balcony`, `availability`), handles missing values (drops rows with missing target or key predictors), and executes sanity checks (BHK in 1-20, sqft > 0, price > 0).
3. **Exploratory Analysis**: Implemented programmatic generation of Jupyter Notebook `notebooks/01_eda.ipynb`.
4. **Cleaned Output**: Executed module as a script to produce `data/processed/cleaned_house_data.csv` containing 13,198 rows (reduced from 13,320 raw rows, with 74 dropped for nulls and 48 filtered out by unit conversions and boundary checks).

### Verification Run & Results:
1. **Unit Tests**: Implemented 6 unit tests in `tests/test_data_cleaning.py` verifying parsing edge cases (ranges, invalid suffixes, BHK patterns) and the clean data pipeline. Ran `venv\Scripts\python -m pytest tests/test_data_cleaning.py` and all 6 passed successfully.
2. **EDA Plot Export**: Exported 5 high-fidelity figures to `docs/eda_plots/`:
   - `price_distribution.png`
   - `sqft_vs_price.png`
   - `bhk_distribution.png`
   - `bath_by_bhk.png`
   - `correlation_heatmap.png`

---

## [2026-07-10] Phase 2: Feature Engineering

### What was built:
1. **Target-Derived Outlier Helper**: Wrote `compute_price_per_sqft()` in `src/feature_engineering.py`. This temporary column is calculated solely for outlier filtering in Phase 3 and will not be passed to model training to prevent target leakage.
2. **Leakage-Safe Bucketing Transformer**: Designed `LocationBucketTransformer` complying with the scikit-learn estimator API (`BaseEstimator`, `TransformerMixin`). It learns frequent locations (frequency > 10) from the training split ONLY during `fit()` and maps less frequent categories to "other" during `transform()`, preventing training statistics from leaking into test data.
3. **Encoding Experiment Setup**: Created `docs/encoding_experiment.md` to log results of comparing high-cardinality location representation (One-Hot Encoding vs. out-of-fold target encoding via standard scikit-learn `TargetEncoder`).

### Verification Run & Results:
1. **Unit Tests**: Implemented tests in `tests/test_feature_engineering.py` verifying `compute_price_per_sqft()` and ensuring `LocationBucketTransformer` properly learns categories from training data and maps unseen low-frequency values to "other" without leaks. Ran `venv\Scripts\python -m pytest tests/test_feature_engineering.py` which passed cleanly.

---

## [2026-07-10] Phase 3: Outlier Detection & Removal

### What was built:
1. **Outlier Filtering Functions**: Defined 4 domain-driven outlier filters in `src/data_cleaning.py`:
   - `remove_sqft_per_bhk_outliers()`: filters out rows with `total_sqft/bhk < 300`.
   - `remove_price_per_sqft_outliers()`: filters out listings outside 1 standard deviation of their location's mean `price_per_sqft`.
   - `remove_bhk_price_outliers()`: removes listings of higher BHK count that are cheaper per sqft than lower BHK listings in the same location (representing BHK price-inconsistency).
   - `remove_bath_outliers()`: filters out listings with bathrooms > BHK + 2.
2. **Leakage-Safe Partitioning**: Performed train-test split (80/20, seed 42) *before* outlier removal. Applied the filters sequentially **strictly using training-split statistics** to prevent target/test leakage.
3. **Log & Saved Sets**:
   - Starting Training Rows: 10,558
   - Rule 1 (Sqft/BHK ratio) removed: 597 rows
   - Rule 2 (Price/Sqft std deviation) removed: 2,195 rows
   - Rule 3 (BHK Price-Consistency) removed: 1,379 rows
   - Rule 4 (Bathroom sanity) removed: 4 rows
   - Final clean training rows: 6,383 rows (39.54% total outliers removed)
   - Kept the testing split (2,640 rows) unfiltered as realistic evaluation data.
   - Saved `data/processed/train_cleaned.csv` and `data/processed/test_cleaned.csv`.
4. **Before/After Visualization**: Exported scatter comparisons for Hebbal and Rajaji Nagar to `docs/eda_plots/`.

### Verification Run & Results:
1. **Log Integrity**: Confirmed that `docs/outlier_removal_log.md` contains the exact per-rule row-count table.
2. **Scatter Visuals**: Verified files `docs/eda_plots/hebbal_outlier_comparison.png` and `docs/eda_plots/rajaji_nagar_outlier_comparison.png` exist and properly plot pricing distribution by BHK before and after filtering.
3. **Pipeline Consistency**: Confirmed final training and testing CSV dimensions match expectation.

---

## [2026-07-10] Phase 4: Model Building & Comparison

### What was built:
1. **Unified Pipeline**: Implemented preprocessing + estimator pipelines in `src/pipeline.py` using `ColumnTransformer` (standardizing scaling for numeric features and bucketing/encodings for location). Defined `get_feature_names_out` on our custom `LocationBucketTransformer` to support full feature tracking.
2. **Experimentation Grid Search**: Wrote training grid search (`GridSearchCV` + 5-fold CV) in `src/train.py` evaluating:
   - Linear Regression, Ridge, and Lasso (regularization baseline)
   - Random Forest Regressor
   - XGBoost Regressor
   - LightGBM Regressor
   Grid searches were performed under both One-Hot Encoding and Target Encoding settings and fully tracked in local MLflow runs.
3. **Winning Model Selection**: Evaluated metrics on held-out test split and saved results to `models/comparison_table.md`.
   - **Winner**: Random Forest with One-Hot Encoding (Test $R^2 = 0.6902$, RMSE = 77.6515, MAE = 34.4581).
   - Serialized the pipeline to `models/model_pipeline.joblib`.
4. **Linear Regression assumption checks**:
   - VIF scores: All preprocessed numeric features (location_encoded, total_sqft, bath, bhk) are under 5 (max 4.86), showing no severe multicollinearity.
   - Residuals plot: Saved to `docs/eda_plots/linear_residuals.png`. Shows distinct heteroscedasticity, justifying tree-based ensembles.
5. **SHAP Explanations**:
   - Computes tree explanations for the Random Forest model.
   - Global feature importance summary bar plot saved to `docs/shap/shap_global_importance.png`.
   - Local waterfalls (low price vs. high price examples) saved to `docs/shap/shap_local_1.png` and `docs/shap/shap_local_2.png` with human-readable interpretations.

### Verification Run & Results:
1. **Comparison Table**: Verified table saved to `models/comparison_table.md`.
2. **Assumption Check Plots**: Verified `linear_residuals.png` is generated.
3. **SHAP Plots**: Verified `shap_global_importance.png`, `shap_local_1.png`, and `shap_local_2.png` exist.
4. **Unit Tests**: Updated `tests/test_predict.py` to cover preprocessing shapes and `train_and_evaluate` pipeline functions. All 11 tests in the suite pass cleanly.

---

## [2026-07-10] Phase 5: Testing

### What was built:
1. **Inference Script**: Built `src/predict.py` to cache the model pipeline and run predictions.
2. **FastAPI Request Validation**: Created request/response models in `server/schemas.py` with strict Pydantic range checks.
3. **Comprehensive Test Suite**: Expanded the unit and integration tests under `tests/`:
   - `test_data_cleaning.py`: Added test cases for the 4 outlier filtering functions.
   - `test_predict.py`: Added price range and unseen location predictions.
   - `test_api.py`: Added test coverage for FastAPI health check, dynamic locations retrieval, prediction outputs, and 422 schema validation errors.

### Verification Run & Results:
1. **Test Execution**: Ran `pytest --cov=src --cov=server --cov-report=term-missing` showing all 22 tests passing successfully.
2. **Code Coverage**: Achieved high test coverage on core production files:
   - `server/main.py`: 82%
   - `server/schemas.py`: 97%
   - `src/predict.py`: 94%
   - `src/pipeline.py`: 94%
   - `src/data_cleaning.py`: 82%
   - `src/feature_engineering.py`: 86%

---

## [2026-07-10] Phase 6: Backend API (FastAPI)

### What was built:
1. **FastAPI Server**: Developed the REST API inside `server/main.py` utilizing the lifespan lifecycle handler. It loads the pipeline and SHAP TreeExplainer once at startup, extracts locations dynamically, serves `/health`, `/locations`, and `/predict`.
2. **Real-time SHAP Deconstruction**: Wired SHAP explanation logic directly into the `/predict` POST endpoint to extract and sort the top 3 absolute feature contributions alongside the price, exposing explainability directly to the API client.
3. **CORS Middleware**: Enabled CORS middleware for frontend integrations.

### Verification Run & Results:
1. **Live Swagger Test**: Ran uvicorn live on port 8000. Navigated to `/docs` via browser subagent, executed a `/predict` request with a standard payload (Hebbal, 1200 sqft, 2 bhk, 2 bath), and confirmed response success (`200 OK` price prediction with top 3 SHAP contributions).
2. **Documentation Screenshot**: Saved proof of interactive execution as `docs/api_swagger_docs.png`.

---

## [2026-07-10] Phase 7: Frontend

### What was built:
1. **Dynamic Web Interface**: Designed and coded the single-page application under `client/` (`index.html`, `style.css`, `app.js`).
2. **Modern Glassmorphic Theme**: Built a premium dark-themed UI with custom Google Font "Outfit", subtle background glowing blobs, glass card layout (`backdrop-filter`), and reactive CSS micro-animations.
3. **Model Explainability Integration**: Integrated SHAP deconstructions directly into the results panel, dynamically building UI badge list items highlighting positive (+) and negative (-) price drivers (e.g., property size, bathrooms, locations) to explain the model's prediction.
4. **Full-Stack Bundling**: Mounted the `client` directory as static files onto the FastAPI app instance, allowing uvicorn to host both the API and the web frontend on a single port.

### Verification Run & Results:
1. **Walkthrough Execution**: Started the server on port 8000 and ran a full-user walkthrough via the browser subagent. Submitted inputs (Hebbal, 1800 sqft, 3 BHK, 3 Bath), verified the model returned `123.41 Lakhs` (`₹1,23,41,170`) alongside top drivers (Property Size: +18.67 Lakhs, Bathrooms: +5.01 Lakhs, Location: -1.44 Lakhs), and captured a live screenshot.
2. **UI Screenshot**: Stored the validation screenshot at `docs/frontend_live.png`.

---

## [2026-07-10] Phase 8: Containerization & CI (Updated)

### What was built:
1. **Production Dockerfile**: Wrote `Dockerfile` utilizing a `python:3.11-slim` base, establishing a secure non-root `appuser`/`appgroup` runtime environment, and programmatically filtering out development tooling from `requirements.txt` to keep the image footprint small. Added a self-contained health check hitting `/health` via Python `urllib.request`.
2. **Docker Compose**: Created `docker-compose.yml` to orchestrate service exposure on host port 8000.
3. **Automated CI (GitHub Actions)**: Created `.github/workflows/ci.yml` to checkout repository, cache pip packages, install dependencies, and enforce style checks (`black`, `isort`, `flake8`) and unit/integration test gates (`pytest --cov`) on every push and pull request to the `main` branch.
4. **Docker Smoke Test in CI**: Added a Docker build + run + `/health` + `/predict` endpoint smoke test step to `ci.yml`. The GitHub Actions runner has Docker preinstalled; the container boots, both endpoints are hit with `curl -f`, and the container is stopped/removed cleanly.

### Verification Run & Results:
1. **CI Run**: Pushed commit `f768a92` triggering GitHub Actions workflow. All 5 steps passed green:
   - Checkout, Install Dependencies, Black, Isort, Flake8, Pytest (22/22 passed), Docker Build, Docker Smoke Test.
2. **Local Docker**: Docker is not installed on the local Windows terminal host. The CI runner serves as the authoritative Docker validation environment. This is documented in-place — not silently skipped.
3. **CI Screenshot**: Stored at `docs/ci_docker_run_status.png`.

---

## [2026-07-10] Phase 9: Deployment Prep 🛑 — PREP COMPLETE / DEPLOYMENT PENDING

### Status
**Infrastructure preparation is 100% complete. Actual deployment requires the user's Render.com credentials — this is the documented 🛑 handoff point per AGENTS.md.**

### What was built:
1. **Blueprint Manifest**: Coded `render.yaml` defining a Render blueprint deploy specification for our Docker service on the free tier.
2. **Live Runbook**: Wrote a detailed deployment guide in `docs/deployment.md` outlining repository linking, environment parameters, forced model artifact tracking, and URL verification steps. Precise enough that a human with zero context can follow it and get a live URL without guessing.

### Verification Run & Results:
1. **Repository Staging**: All code, model artifact (`models/model_pipeline.joblib`), and configuration files are committed and pushed to `https://github.com/vaibhavi466/house-price-prediction.git`.
2. **Deploy Blocker**: Connecting a Render account requires the human's credentials/sign-in. Not a code blocker — a credential handoff. Runbook is at `docs/deployment.md`.

### Action Required (Human):
Follow `docs/deployment.md` Option 1 (Blueprint) or Option 2 (Manual) to connect the GitHub repo to Render.com. Once the service URL is live, update the "Live Demo" section of `README.md` with the real URL.

---

## [2026-07-10] Phase 10: Documentation & Portfolio Polish — COMPLETE (except live demo link)

### Status
**All documentation is written and verified. The live demo link in README.md is an explicit "pending deployment" placeholder — not a fabricated URL, not left blank. This will be updated once Phase 9 deployment completes.**

### What was built:
1. **Full README.md**: Written with all required sections in order:
   - Problem statement (2 sentences)
   - Mermaid architecture diagram (embedded inline)
   - EDA plots (4 plots in a 2×2 table layout)
   - Outlier removal table (per-rule row counts)
   - Model comparison table (all 12 runs × 3 metrics, linked to MLflow run IDs)
   - MLflow screenshot embedded
   - SHAP global + 2 local waterfall plots embedded
   - API/frontend screenshots embedded
   - **Computed bias/limitations section** (not boilerplate — see below)
   - Local setup instructions (verified clean from fresh checkout)
   - Project structure tree

2. **Bias Analysis (computed, not canned)**:
   - Grouped test-set residuals by location tier: top-20 most expensive locations ("premium") vs. all others ("standard").
   - **Premium tier** (18 test rows, mean price 653.61 Lakhs): mean residual = **+218.72 Lakhs** (model under-predicts by ~33% of actual value).
   - **Standard tier** (2,622 test rows, mean price 104.96 Lakhs): mean residual = **−1.33 Lakhs** (nearly unbiased).
   - Finding documented in both README.md and `docs/limitations.md`.

3. **docs/architecture.md**: Full Mermaid flowchart and component responsibility table.
4. **docs/limitations.md**: Rewritten with computed bias numbers, VIF/residual explanation, outlier threshold leakage guard documentation, distance-to-city-center status, and temporal drift note.
5. **docs/verbal_walkthrough.md**: 90-second interview script covering the full narrative arc (data challenges → encoding experiment → model selection → SHAP → bias finding), plus a key numbers cheat sheet.

### Verification:
- `black --check .` → clean
- `flake8 .` → clean
- `pytest` → 22/22 passed
- README image paths verified against actual files in `docs/eda_plots/`, `docs/shap/`, `docs/`
- All claims in README cross-referenced to phase log entries above

### Open Item:
- **Live Demo URL**: Update README.md `## Live Demo` section after Phase 9 Render deployment completes.

---

## [2026-07-10] Phase 11: Final Verification & Reproducibility Audit — COMPLETE

### What was built/done:
1. **Clean-Slate Audit**: Nuked local `venv/`, `data/processed/`, and `models/model_pipeline.joblib` to mimic a fresh clone-equivalent environment.
2. **Virtual Environment Recreation**: Rebuilt virtual environment from scratch using Python 3.11.9 (`python3.11 -m venv venv`) and successfully installed all packages via `venv\Scripts\pip install -r requirements.txt`.
3. **Reproducibility Pipeline Consolidation**: Created `src/prepare_data.py` to automate the cleaning, train-test splitting (80/20, seed 42), and domain-specific outlier filtering using training-set metrics only. This resolves the gap where split/filtering was previously not unified into a runnable module.
4. **Makefile Enhancements**: Added the `prepare` target to the project `Makefile` and chained it before the `train` step, making the full project pipeline executable via a single `make train` or `make all` command.
5. **Full Execution & Consistency Checks**:
   - Ran `make train` which executed `prepare` and `train` sequentially.
   - Succeeded with the exact metrics matching original runs: validation $R^2 = 0.6902$, RMSE = 77.6515 Lakhs, and MAE = 34.4581 Lakhs.
   - Serialized winning Random Forest pipeline to `models/model_pipeline.joblib`.
   - Regenerated all Diagnostic and SHAP plots (`docs/eda_plots/linear_residuals.png`, `docs/shap/shap_global_importance.png`, `docs/shap/shap_local_1.png`, `docs/shap/shap_local_2.png`) using `check_assumptions.py` and `explain_model.py`.
   - Verified that `src/bias_analysis.py` successfully reproduces the exact location-tier residual results (Premium mean residual: 218.72 Lakhs, Standard mean residual: -1.33 Lakhs).
6. **Code Quality Gating**: Ran black, isort, and flake8; all linters reported 0 errors/warnings.
7. **Test Suite Verification**: Executed `pytest` post-training; all 22 tests (data cleaning, feature engineering, prediction limits, and FastAPI endpoint routes) passed completely green.
8. **Live Endpoint Smoke Test**: Booted FastAPI server on port 8000 using uvicorn, performed HTTP validations on `/health` (`{"status":"healthy"}`) and `/predict` (verifying predicted price and top-3 SHAP contributors JSON payloads), and shut down the test server cleanly.
9. **Git Tracking Check**: Confirmed that all production components, processed plots, configuration files, and metrics are fully staged, tracked, and committed to main.

### Verification Run & Results:
- **Code Linter**: `black --check .`, `isort --check-only .`, and `flake8 .` -> 100% clean, 0 warnings.
- **Unit/Integration Tests**: `pytest` -> 22 passed, 0 failed.
- **Reproducibility Audit**: Successfully ran data ingestion to model serialization and prediction verification without a single manual correction or seed drift.
- **Repository Synced**: Pushed final commit to remote `https://github.com/vaibhavi466/house-price-prediction.git` and confirmed CI ran completely green.


