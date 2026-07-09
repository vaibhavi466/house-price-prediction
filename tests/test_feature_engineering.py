import pandas as pd

from src.feature_engineering import LocationBucketTransformer, compute_price_per_sqft


def test_compute_price_per_sqft():
    df = pd.DataFrame({"price": [10.0, 50.0], "total_sqft": [1000.0, 2000.0]})
    result_df = compute_price_per_sqft(df)

    # price is in Lakhs, so price_per_sqft = (price * 100,000) / total_sqft
    # For index 0: (10.0 * 100000) / 1000.0 = 1000.0
    # For index 1: (50.0 * 100000) / 2000.0 = 2500.0
    assert "price_per_sqft" in result_df.columns
    assert result_df["price_per_sqft"].iloc[0] == 1000.0
    assert result_df["price_per_sqft"].iloc[1] == 2500.0


def test_location_bucket_transformer():
    # Set threshold to 2 for easier testing
    transformer = LocationBucketTransformer(threshold=2)

    train_locations = pd.Series(
        [
            "A",
            "A",
            "A",  # 3 times (> 2) -> keep
            "B",
            "B",  # 2 times (<= 2) -> other
            "C",  # 1 time (<= 2) -> other
        ]
    )

    transformer.fit(train_locations)

    # Only "A" has count > 2
    assert transformer.frequent_locations_ == {"A"}

    # Test transform on training data
    transformed_train = transformer.transform(train_locations)
    assert list(transformed_train) == ["A", "A", "A", "other", "other", "other"]

    # Test transform on unseen test data
    test_locations = pd.Series(["A", "B", "C", "D"])
    transformed_test = transformer.transform(test_locations)
    assert list(transformed_test) == ["A", "other", "other", "other"]
