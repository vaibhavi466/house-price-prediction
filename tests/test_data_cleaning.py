import numpy as np
import pandas as pd

from src.data_cleaning import clean_data, convert_sqft, extract_bhk


def test_extract_bhk():
    assert extract_bhk("2 BHK") == 2
    assert extract_bhk("4 Bedroom") == 4
    assert extract_bhk("1 BHK") == 1
    assert extract_bhk("10 BHK") == 10
    assert extract_bhk("1 RK") == 1
    assert extract_bhk(None) is None
    assert extract_bhk(np.nan) is None
    assert extract_bhk("invalid") is None


def test_convert_sqft_numeric():
    assert convert_sqft("1200") == 1200.0
    assert convert_sqft("1200.5") == 1200.5
    assert convert_sqft(1500) == 1500.0
    assert convert_sqft(-100) is None


def test_convert_sqft_range():
    assert convert_sqft("2100 - 2850") == 2475.0
    assert convert_sqft("1000 - 1200") == 1100.0
    assert convert_sqft("1000-1200") == 1100.0


def test_convert_sqft_invalid_units():
    # Suffixes should return None so that row gets dropped
    assert convert_sqft("34.46Sq. Meter") is None
    assert convert_sqft("1000Sq. Yards") is None
    assert convert_sqft("Ground") is None
    assert convert_sqft("Perch") is None


def test_clean_data_pipeline():
    raw_data = {
        "area_type": [
            "Super built-up  Area",
            "Plot  Area",
            "Built-up  Area",
            "Super built-up  Area",
        ],
        "availability": ["Ready To Move", "Ready To Move", "Ready To Move", "18-Dec"],
        "location": ["Electronic City Phase II", "Chikka Tirupathi", "Uttarahalli", "Kothanur"],
        "size": ["2 BHK", "4 Bedroom", "3 BHK", "2 BHK"],
        "society": ["Coomee ", "Theanmp", np.nan, np.nan],
        "total_sqft": [
            "1056",
            "2600",
            "1440 - 1500",
            "34.46Sq. Meter",
        ],  # 4th has invalid sqft unit, should be dropped
        "bath": [2.0, 5.0, 2.0, 2.0],
        "balcony": [1.0, 3.0, 1.0, 1.0],
        "price": [39.07, 120.0, 62.0, 50.0],
    }
    df = pd.DataFrame(raw_data)
    cleaned_df = clean_data(df)

    # 4th row should have been dropped due to invalid sqft unit
    assert len(cleaned_df) == 3
    assert "area_type" not in cleaned_df.columns
    assert "society" not in cleaned_df.columns
    assert "balcony" not in cleaned_df.columns
    assert "availability" not in cleaned_df.columns

    assert list(cleaned_df["bhk"]) == [2, 4, 3]
    assert list(cleaned_df["total_sqft"]) == [1056.0, 2600.0, 1470.0]
    assert list(cleaned_df["bath"]) == [2, 5, 2]
    assert list(cleaned_df["price"]) == [39.07, 120.0, 62.0]


def test_clean_data_out_of_bounds():
    raw_data = {
        "location": ["Electronic City Phase II"],
        "size": ["22 BHK"],  # Out of bounds (1-20)
        "total_sqft": ["1056"],
        "bath": [2.0],
        "price": [39.07],
    }
    df = pd.DataFrame(raw_data)
    cleaned_df = clean_data(df)
    assert len(cleaned_df) == 0  # Should be filtered out due to BHK > 20


def test_remove_sqft_per_bhk_outliers():
    from src.data_cleaning import remove_sqft_per_bhk_outliers

    df = pd.DataFrame({"total_sqft": [600.0, 500.0], "bhk": [2, 1]})
    # index 0: 600/2 = 300 (keep), index 1: 500/1 = 500 (keep)
    res = remove_sqft_per_bhk_outliers(df)
    assert len(res) == 2

    df_outlier = pd.DataFrame({"total_sqft": [500.0, 600.0], "bhk": [2, 1]})
    # index 0: 500/2 = 250 (drop), index 1: 600/1 = 600 (keep)
    res_outlier = remove_sqft_per_bhk_outliers(df_outlier)
    assert len(res_outlier) == 1
    assert res_outlier.iloc[0]["total_sqft"] == 600.0


def test_remove_price_per_sqft_outliers():
    from src.data_cleaning import remove_price_per_sqft_outliers

    # Hebbal has 3 listings: one is a huge outlier
    df = pd.DataFrame(
        {"location": ["Hebbal", "Hebbal", "Hebbal"], "price_per_sqft": [5000.0, 5200.0, 15000.0]}
    )
    res = remove_price_per_sqft_outliers(df)
    # Mean is 8400, std is ~4669. Band is [3730, 13069].
    # 5000 and 5200 are in. 15000 is out.
    assert len(res) == 2
    assert 15000.0 not in res["price_per_sqft"].values


def test_remove_bhk_price_outliers():
    from src.data_cleaning import remove_bhk_price_outliers

    # Hebbal has 2 BHK and 3 BHK listings.
    # The 2 BHK has 6 listings (mean price_per_sqft = 5000)
    # The 3 BHK has 1 listing (price_per_sqft = 4000) -> cheaper per sqft than lower tier mean!
    df = pd.DataFrame(
        {
            "location": ["Hebbal"] * 7,
            "bhk": [2, 2, 2, 2, 2, 2, 3],
            "price_per_sqft": [5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 4000.0],
        },
        index=list(range(7)),
    )

    res = remove_bhk_price_outliers(df)
    # The 3 BHK listing (index 6) should be removed
    assert len(res) == 6
    assert 3 not in res["bhk"].values


def test_remove_bath_outliers():
    from src.data_cleaning import remove_bath_outliers

    df = pd.DataFrame({"bhk": [2, 2], "bath": [4, 5]})
    # index 0: bath=4 (<= 2+2=4 -> keep), index 1: bath=5 (> 2+2=4 -> drop)
    res = remove_bath_outliers(df)
    assert len(res) == 1
    assert res.iloc[0]["bath"] == 4
