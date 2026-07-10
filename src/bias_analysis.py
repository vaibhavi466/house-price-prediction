"""
src/bias_analysis.py — Phase 10 bias analysis.

Groups test-set residuals (actual − predicted) by location tier to detect
systematic over/under-prediction. Prints a per-tier summary and the top-20
premium locations used to define the tier boundary.

Usage:
    python -m src.bias_analysis
    # or from repo root:
    python src/bias_analysis.py
"""

import os
import sys
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).parent.parent / "models"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Number of top-priced locations that define the "premium" tier
PREMIUM_N = 20


def compute_bias_by_tier(
    test_csv: Path = PROCESSED_DATA_DIR / "test_cleaned.csv",
    train_csv: Path = PROCESSED_DATA_DIR / "train_cleaned.csv",
    model_path: Path = MODELS_DIR / "model_pipeline.joblib",
    premium_n: int = PREMIUM_N,
) -> pd.DataFrame:
    """
    Load the test split and the serialized pipeline, compute residuals, and
    return a DataFrame summarising mean/median residual by location tier.

    Parameters
    ----------
    test_csv   : Path to the cleaned test split.
    train_csv  : Path to the cleaned training split (used to define tier
                 boundaries from training-only statistics — no leakage).
    model_path : Path to the serialized sklearn Pipeline (.joblib).
    premium_n  : Number of top-priced locations (by training mean price)
                 that constitute the 'premium' tier.

    Returns
    -------
    summary_df : DataFrame with columns tier, count, mean_actual_price,
                 mean_residual, median_residual.
    """
    test_df = pd.read_csv(test_csv)
    train_df = pd.read_csv(train_csv)
    pipeline = joblib.load(model_path)

    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"].values

    y_pred = pipeline.predict(X_test)
    residuals = y_test - y_pred  # actual − predicted

    # Define tier using training-split statistics only
    loc_mean_price = train_df.groupby("location")["price"].mean()
    top_locations = set(loc_mean_price.nlargest(premium_n).index.tolist())

    test_df = test_df.copy()
    test_df["residual"] = residuals
    test_df["tier"] = test_df["location"].apply(
        lambda loc: "premium" if loc in top_locations else "standard"
    )

    summary = (
        test_df.groupby("tier")
        .agg(
            count=("residual", "count"),
            mean_actual_price=("price", "mean"),
            mean_residual=("residual", "mean"),
            median_residual=("residual", "median"),
        )
        .reset_index()
    )
    return summary, loc_mean_price, top_locations


def main() -> None:
    summary, loc_mean_price, top_locations = compute_bias_by_tier()

    print("=== Bias Analysis: Residuals by Location Tier ===\n")
    for _, row in summary.iterrows():
        print(f"Tier: {row['tier'].upper()}")
        print(f"  Count:                   {int(row['count'])}")
        print(f"  Mean actual price:       {row['mean_actual_price']:.2f} Lakhs")
        print(f"  Mean residual:           {row['mean_residual']:.2f} Lakhs")
        print(f"  Median residual:         {row['median_residual']:.2f} Lakhs")
        print()

    print(f"Top {len(top_locations)} premium locations (by training mean price):")
    for loc in loc_mean_price.nlargest(len(top_locations)).index:
        print(f"  {loc}: {loc_mean_price[loc]:.1f} Lakhs avg")


if __name__ == "__main__":
    # When run as `python src/bias_analysis.py` the repo root is not on sys.path,
    # but `src` imports (and the joblib-pickled pipeline) need it.  Add it here.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
