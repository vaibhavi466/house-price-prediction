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
