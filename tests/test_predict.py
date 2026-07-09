import pandas as pd
import numpy as np
import pytest
from src.pipeline import get_preprocessing_pipeline

def test_preprocessing_pipeline_onehot():
    # Setup dummy data
    X_train = pd.DataFrame({
        "location": ["Hebbal", "Hebbal", "Rajaji Nagar", "Rajaji Nagar", "Whitefield", "Whitefield", "Whitefield", "Whitefield", "Whitefield", "Whitefield", "Whitefield"],
        "total_sqft": [1000.0, 1200.0, 1500.0, 1800.0, 2000.0, 2200.0, 2100.0, 2500.0, 2400.0, 2300.0, 2200.0],
        "bath": [2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4],
        "bhk": [2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4]
    })
    y_train = pd.Series([30, 35, 60, 65, 80, 85, 82, 90, 88, 86, 84])
    
    # We set threshold to 2 for testing
    pipeline = get_preprocessing_pipeline(encoding_type="onehot", threshold=2)
    pipeline.fit(X_train, y_train)
    
    X_trans = pipeline.transform(X_train)
    
    # Check shape:
    # Frequent locations (>2): 'Whitefield' (7 listings). Others ('Hebbal', 'Rajaji Nagar') are mapped to 'other'.
    # OneHot Encoder categories will be ['Whitefield', 'other'], so 2 columns.
    # Numerical features: ['total_sqft', 'bath', 'bhk'], so 3 columns.
    # Total features: 5.
    assert X_trans.shape[1] == 5

def test_preprocessing_pipeline_target():
    X_train = pd.DataFrame({
        "location": ["Hebbal", "Hebbal", "Rajaji Nagar", "Rajaji Nagar", "Whitefield", "Whitefield", "Whitefield", "Whitefield", "Whitefield", "Whitefield", "Whitefield"],
        "total_sqft": [1000.0, 1200.0, 1500.0, 1800.0, 2000.0, 2200.0, 2100.0, 2500.0, 2400.0, 2300.0, 2200.0],
        "bath": [2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4],
        "bhk": [2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4]
    })
    y_train = pd.Series([30, 35, 60, 65, 80, 85, 82, 90, 88, 86, 84])
    
    pipeline = get_preprocessing_pipeline(encoding_type="target", threshold=2)
    pipeline.fit(X_train, y_train)
    
    X_trans = pipeline.transform(X_train)
    
    # Check shape:
    # Target encoding replaces 'location' with 1 numeric column.
    # Numerical features: ['total_sqft', 'bath', 'bhk'] (3 columns).
    # Total features: 4.
    assert X_trans.shape[1] == 4
