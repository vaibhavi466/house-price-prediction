from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src.feature_engineering import LocationBucketTransformer


def get_preprocessing_pipeline(encoding_type: str = "onehot", threshold: int = 10) -> Pipeline:
    """
    Creates and returns a scikit-learn Pipeline representing the preprocessing steps.
    Supports either 'onehot' or 'target' encoding for the location feature.
    """
    if encoding_type not in ["onehot", "target"]:
        raise ValueError("encoding_type must be either 'onehot' or 'target'")

    # 1. Define categorical preprocessor branch
    if encoding_type == "onehot":
        cat_transformer = Pipeline(
            steps=[
                ("bucket", LocationBucketTransformer(threshold=threshold)),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    else:
        # target encoder in scikit-learn is cross-fitted internally to prevent target leakage.
        # We specify target_type="continuous" to handle regression targets correctly,
        # and pass a KFold generator to avoid deprecation warnings.
        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        cat_transformer = Pipeline(
            steps=[
                ("bucket", LocationBucketTransformer(threshold=threshold)),
                ("target_enc", TargetEncoder(target_type="continuous", cv=cv_splitter)),
            ]
        )

    # 2. Define numerical preprocessor branch
    num_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # 3. Combine both branches into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, ["location"]),
            ("num", num_transformer, ["total_sqft", "bath", "bhk"]),
        ],
        remainder="drop",
    )

    # Return as a Pipeline for modularity
    return Pipeline(steps=[("preprocessor", preprocessor)])
