import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def compute_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes price_per_sqft column.
    WARNING: This column is derived from the target 'price' and must ONLY be used
    for outlier detection in Phase 3. It must NEVER be fed to the machine learning model.
    """
    df = df.copy()
    # price is in Lakhs, so we multiply by 100,000 to get absolute price in Rupees
    df["price_per_sqft"] = (df["price"] * 100000) / df["total_sqft"]
    return df

class LocationBucketTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for bucketing high-cardinality location values.
    Locations with frequency <= threshold in the training data are mapped to 'other'.
    Fitted ONLY on the training split to prevent test-set leakage.
    """
    def __init__(self, threshold: int = 10):
        self.threshold = threshold
        self.frequent_locations_ = set()

    def fit(self, X, y=None):
        # Handle cases where X is a DataFrame, Series, or NumPy array
        if isinstance(X, pd.DataFrame):
            # Expecting a single column DataFrame or Series
            col = X.columns[0]
            series = X[col]
        elif isinstance(X, pd.Series):
            series = X
        else:
            series = pd.Series(X.ravel())

        counts = series.value_counts()
        self.frequent_locations_ = set(counts[counts > self.threshold].index)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            col = X.columns[0]
            res = X[col].apply(lambda loc: loc if loc in self.frequent_locations_ else "other")
            return pd.DataFrame(res, columns=[col])
        elif isinstance(X, pd.Series):
            return X.apply(lambda loc: loc if loc in self.frequent_locations_ else "other")
        else:
            flat = X.ravel()
            res = np.array([loc if loc in self.frequent_locations_ else "other" for loc in flat])
            return res.reshape(X.shape)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(["location"], dtype=object)
        return np.asarray(input_features, dtype=object)

