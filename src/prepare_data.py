"""
src/prepare_data.py — Full data preparation pipeline.

Orchestrates the end-to-end flow from raw CSV to cleaned, split, and
outlier-filtered train/test CSVs that src/train.py expects.

Steps:
  1. Load raw CSV
  2. Clean (parse BHK, convert sqft, drop nulls, sanity checks)
  3. Train/test split (80/20, RANDOM_STATE=42)
  4. Compute price_per_sqft on training split
  5. Apply four outlier filters using training-split statistics only
  6. Save train_cleaned.csv and test_cleaned.csv to data/processed/

The test split is intentionally left unfiltered so evaluation metrics
reflect real-world uncleaned inputs (documented design decision).

Usage:
    python -m src.prepare_data
    # or from repo root:
    python src/prepare_data.py
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR, RANDOM_STATE, RAW_DATA_FILE
from src.data_cleaning import (
    clean_data,
    remove_bath_outliers,
    remove_bhk_price_outliers,
    remove_price_per_sqft_outliers,
    remove_sqft_per_bhk_outliers,
)
from src.feature_engineering import compute_price_per_sqft


def prepare_data() -> None:
    """
    Full data preparation pipeline: clean → split → outlier removal → save.
    Logs row counts at every step to match docs/outlier_removal_log.md.
    """
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {RAW_DATA_FILE}. "
            "Place Bengaluru_House_Data.csv in data/raw/ before running."
        )

    # ── Step 1: Load & Clean ──────────────────────────────────────────────────
    print(f"Loading raw data from {RAW_DATA_FILE}...")
    df_raw = pd.read_csv(RAW_DATA_FILE)
    print(f"  Raw shape: {df_raw.shape}")

    df_clean = clean_data(df_raw)
    print(f"  Cleaned shape: {df_clean.shape}")

    # ── Step 2: Train/Test Split (80/20, seeded) ──────────────────────────────
    train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=RANDOM_STATE)
    print(f"\nTrain rows before outlier removal: {len(train_df)}")
    print(f"Test rows (kept unfiltered):       {len(test_df)}")

    # ── Step 3: price_per_sqft on training split (for outlier filters) ────────
    # Computed on training data only — not fed to model (would leak target).
    train_df = compute_price_per_sqft(train_df.copy())

    # ── Step 4: Apply Outlier Filters (training split only) ───────────────────
    n0 = len(train_df)

    train_df = remove_sqft_per_bhk_outliers(train_df)
    n1 = len(train_df)
    print(f"\nAfter sqft/BHK filter:      {n1} rows (removed {n0 - n1})")

    train_df = remove_price_per_sqft_outliers(train_df)
    n2 = len(train_df)
    print(f"After price/sqft filter:    {n2} rows (removed {n1 - n2})")

    train_df = remove_bhk_price_outliers(train_df)
    n3 = len(train_df)
    print(f"After BHK consistency:      {n3} rows (removed {n2 - n3})")

    train_df = remove_bath_outliers(train_df)
    n4 = len(train_df)
    print(f"After bath sanity filter:   {n4} rows (removed {n3 - n4})")

    total_removed = n0 - n4
    pct = total_removed / n0 * 100
    print(f"\nTotal removed from training: {total_removed} ({pct:.2f}%)")
    print(f"Final training rows:          {n4}")

    # Drop price_per_sqft helper column before saving (it's a derived feature
    # computed from the target and must not be passed to the model).
    train_df = train_df.drop(columns=["price_per_sqft"], errors="ignore")

    # ── Step 5: Save Outputs ──────────────────────────────────────────────────
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_out = PROCESSED_DATA_DIR / "train_cleaned.csv"
    test_out = PROCESSED_DATA_DIR / "test_cleaned.csv"
    cleaned_out = PROCESSED_DATA_DIR / "cleaned_house_data.csv"

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    df_clean.to_csv(cleaned_out, index=False)

    print(f"\nSaved: {train_out}")
    print(f"Saved: {test_out}")
    print(f"Saved: {cleaned_out}")


if __name__ == "__main__":
    # Allow running as `python src/prepare_data.py` from repo root
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prepare_data()
