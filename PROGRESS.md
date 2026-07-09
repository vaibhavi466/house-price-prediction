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
