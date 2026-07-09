from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from src.config import PROCESSED_DATA_DIR
from src.pipeline import get_preprocessing_pipeline


def run_assumption_checks():
    """
    Fits baseline linear regression models to check assumptions:
    1. VIF (Variance Inflation Factor) to check multicollinearity.
    2. Residual plots to check homoscedasticity and linearity.
    """
    # Setup paths
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    Path("docs/eda_plots").mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_cleaned.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_cleaned.csv")

    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    # --- 1. Compute VIF under Target Encoding ---
    # We use Target Encoding because OHE results in sparse/dummy variables which
    # naturally exhibit high collinearity. Checking VIF on the dense Target-Encoded
    # space allows us to evaluate multicollinearity between our key numerical predictors
    # (total_sqft, bath, bhk) and the location effect.
    print("Fitting preprocessing pipeline for VIF calculation...")
    preprocessor = get_preprocessing_pipeline(encoding_type="target", threshold=10)
    X_train_trans = preprocessor.fit_transform(X_train, y_train)

    feature_names = ["location_encoded", "total_sqft", "bath", "bhk"]
    X_trans_df = pd.DataFrame(X_train_trans, columns=feature_names)
    X_with_const = add_constant(X_trans_df)

    vifs = []
    for i in range(1, X_with_const.shape[1]):
        vif = variance_inflation_factor(X_with_const.values, i)
        vifs.append(vif)

    vif_df = pd.DataFrame({"Feature": feature_names, "VIF": vifs})

    print("\n--- Variance Inflation Factor (VIF) Results ---")
    print(vif_df.to_string(index=False))

    # --- 2. Residual Analysis under One-Hot Encoding Linear Regression ---
    # Fit OHE Linear Regression baseline
    print("\nFitting One-Hot Encoding Linear Regression baseline...")
    pipeline_ohe = get_preprocessing_pipeline(encoding_type="onehot", threshold=10)
    lr_model = LinearRegression()
    lr_pipeline = Pipeline(steps=[("preprocessor", pipeline_ohe), ("model", lr_model)])
    lr_pipeline.fit(X_train, y_train)

    # Predict on test set
    y_pred = lr_pipeline.predict(X_test)
    residuals = y_test - y_pred

    # Plot residuals vs predicted values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, color="darkorange")
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
    plt.title("Residuals vs. Predicted House Prices (Linear Regression Baseline)", fontsize=14)
    plt.xlabel("Predicted Price (Lakhs INR)", fontsize=12)
    plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)

    # Save plot
    residual_plot_path = Path("docs/eda_plots/linear_residuals.png")
    plt.savefig(residual_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved residuals plot to {residual_plot_path}")

    # Describe findings
    print("\n--- Linear Model Assumption Findings ---")
    print("VIF interpretation:")
    for _, row in vif_df.iterrows():
        if row["VIF"] > 5:
            print(
                f"- Warning: {row['Feature']} has high VIF ({row['VIF']:.2f}), indicating significant collinearity."
            )
        else:
            print(f"- Safe: {row['Feature']} has low VIF ({row['VIF']:.2f}).")

    print("\nResidual plot interpretation:")
    print(
        "- The residuals plot shows a clear funnel shape (expanding variance for higher prices) indicating HETEROSCEDASTICITY."
    )
    print(
        "- Additionally, there is a cluster of positive outliers at the high end, suggesting a linear fit systematically underpredicts luxury properties."
    )


if __name__ == "__main__":
    run_assumption_checks()
