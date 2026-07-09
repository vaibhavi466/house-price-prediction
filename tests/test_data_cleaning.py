import pandas as pd
import numpy as np
import pytest
from src.data_cleaning import extract_bhk, convert_sqft, clean_data

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
        "area_type": ["Super built-up  Area", "Plot  Area", "Built-up  Area", "Super built-up  Area"],
        "availability": ["Ready To Move", "Ready To Move", "Ready To Move", "18-Dec"],
        "location": ["Electronic City Phase II", "Chikka Tirupathi", "Uttarahalli", "Kothanur"],
        "size": ["2 BHK", "4 Bedroom", "3 BHK", "2 BHK"],
        "society": ["Coomee ", "Theanmp", np.nan, np.nan],
        "total_sqft": ["1056", "2600", "1440 - 1500", "34.46Sq. Meter"], # 4th has invalid sqft unit, should be dropped
        "bath": [2.0, 5.0, 2.0, 2.0],
        "balcony": [1.0, 3.0, 1.0, 1.0],
        "price": [39.07, 120.0, 62.0, 50.0]
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
        "size": ["22 BHK"], # Out of bounds (1-20)
        "total_sqft": ["1056"],
        "bath": [2.0],
        "price": [39.07]
    }
    df = pd.DataFrame(raw_data)
    cleaned_df = clean_data(df)
    assert len(cleaned_df) == 0  # Should be filtered out due to BHK > 20
