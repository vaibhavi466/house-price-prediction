import re

import numpy as np
import pandas as pd


def extract_bhk(size_str: str) -> int | None:
    """
    Parses BHK/Bedroom size string and returns the number of bedrooms as an integer.
    E.g., "2 BHK" -> 2, "4 Bedroom" -> 4. Returns None if malformed or null.
    """
    if pd.isna(size_str) or not isinstance(size_str, str):
        return None

    # Extract leading integer digit(s)
    match = re.match(r"^(\d+)\s*(?:bhk|bedroom|rk|bed|room)?", size_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def convert_sqft(sqft_str: str) -> float | None:
    """
    Converts total_sqft string to float. If it is a range (e.g., '2100 - 2850'),
    returns the average of the range. If float, returns it. If invalid unit, returns None.
    """
    if pd.isna(sqft_str) or not isinstance(sqft_str, str):
        try:
            val = float(sqft_str)
            return val if val > 0 else None
        except (ValueError, TypeError):
            return None

    sqft_str = sqft_str.strip()

    # Case 1: Simple numeric string
    try:
        val = float(sqft_str)
        return val if val > 0 else None
    except ValueError:
        pass

    # Case 2: Range like '2100 - 2850'
    range_match = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$", sqft_str)
    if range_match:
        try:
            start = float(range_match.group(1))
            end = float(range_match.group(2))
            val = (start + end) / 2.0
            return val if val > 0 else None
        except ValueError:
            return None

    # Case 3: Numeric followed by non-numeric suffix (e.g., "34.46Sq. Meter" or "1000Sq. Yards")
    # We do NOT attempt unit conversion to avoid fabrication of features. We return None so the row gets dropped.
    return None


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw Bangalore housing dataset:
    - Drops unneeded columns (area_type, society, balcony, availability)
    - Resolves nulls: drops rows with nulls in key fields (location, size, bath, total_sqft, price)
    - Extracts bhk and converts total_sqft
    - Asserts data bounds (sqft > 0, bhk between 1 and 20, price > 0)
    """
    # 1. Drop unneeded columns
    cols_to_drop = ["area_type", "society", "balcony", "availability"]
    # Filter to columns that actually exist in the dataframe to avoid errors
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 2. Drop rows where critical columns are null
    # Rationale:
    # - location: only 1 null row; drop.
    # - size: 16 null rows; we cannot reliably impute bedroom counts without context; drop.
    # - bath: 73 null rows; small percentage (~0.5%), dropping is safer than fabricating; drop.
    # - price: target variable; drop.
    # - total_sqft: core feature; drop.
    initial_len = len(df)
    df = df.dropna(subset=["location", "size", "bath", "total_sqft", "price"])
    print(
        f"Dropped {initial_len - len(df)} rows due to missing values in location, size, bath, total_sqft, or price."
    )

    # Copy to avoid setting-with-copy warning
    df = df.copy()

    # 3. Parse BHK
    df["bhk"] = df["size"].apply(extract_bhk)
    df = df.dropna(subset=["bhk"])
    df["bhk"] = df["bhk"].astype(int)

    # Drop the original 'size' column as it's replaced by 'bhk'
    df = df.drop(columns=["size"])

    # 4. Convert total_sqft
    df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
    df = df.dropna(subset=["total_sqft"])
    df["total_sqft"] = df["total_sqft"].astype(float)

    # 5. Type conversions
    df["bath"] = df["bath"].astype(int)
    df["price"] = df["price"].astype(float)

    # 6. Sanity assertions (data boundary checks)
    # - BHK should be within a reasonable range (1 to 20 BHK)
    # - total_sqft should be positive and greater than 0
    # - price should be positive and greater than 0
    # Filter rows that violate these checks to make the pipeline robust
    df = df[(df["bhk"] >= 1) & (df["bhk"] <= 20) & (df["total_sqft"] > 0) & (df["price"] > 0)]

    # Assert sanity check limits on final clean DataFrame
    assert (df["bhk"] >= 1).all() and (df["bhk"] <= 20).all(), "BHK values out of bounds (1-20)"
    assert (df["total_sqft"] > 0).all(), "total_sqft must be positive"
    assert (df["price"] > 0).all(), "price must be positive"

    return df


def remove_sqft_per_bhk_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out listings where total_sqft / bhk < 300.
    """
    return df[df["total_sqft"] / df["bhk"] >= 300]


def remove_price_per_sqft_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups by location and filters out listings whose price_per_sqft is outside
    1 standard deviation of that location's mean price_per_sqft.
    """
    df_out = pd.DataFrame()
    for _, subdf in df.groupby("location"):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        if pd.isna(st) or st == 0:
            df_out = pd.concat([df_out, subdf], ignore_index=True)
        else:
            reduced_df = subdf[
                (subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))
            ]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


def remove_bhk_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes higher-BHK listings in a location that are cheaper per sqft than the
    mean price_per_sqft of the next lower BHK tier in that same location.
    """
    exclude_indices = np.array([])
    for location, location_df in df.groupby("location"):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby("bhk"):
            bhk_stats[bhk] = {
                "mean": np.mean(bhk_df.price_per_sqft),
                "std": np.std(bhk_df.price_per_sqft),
                "count": len(bhk_df),
            }
        for bhk, bhk_df in location_df.groupby("bhk"):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats["count"] > 5:
                exclude_indices = np.append(
                    exclude_indices, bhk_df[bhk_df.price_per_sqft < stats["mean"]].index
                )
    return df.drop(exclude_indices, axis="index")


def remove_bath_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out listings where bathrooms > bhk + 2.
    """
    return df[df["bath"] <= df["bhk"] + 2]


if __name__ == "__main__":
    from src.config import CLEANED_DATA_FILE, RAW_DATA_FILE

    print(f"Loading raw data from {RAW_DATA_FILE}...")
    if not RAW_DATA_FILE.exists():
        print(f"ERROR: Raw data file not found at {RAW_DATA_FILE}")
        exit(1)
    df = pd.read_csv(RAW_DATA_FILE)
    print(f"Original shape: {df.shape}")
    cleaned_df = clean_data(df)
    print(f"Cleaned shape: {cleaned_df.shape}")

    # Ensure processed directory exists
    CLEANED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(CLEANED_DATA_FILE, index=False)
    print(f"Saved cleaned data to {CLEANED_DATA_FILE}")
