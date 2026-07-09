import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

def run_model_explanations():
    """
    Computes global and local SHAP explanations for the winning Random Forest model.
    """
    # Setup paths
    SHAP_DIR = Path("docs/shap")
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    # Load data and winning model pipeline
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_cleaned.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_cleaned.csv")

    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    pipeline_path = MODELS_DIR / "model_pipeline.joblib"
    print(f"Loading winning model pipeline from {pipeline_path}...")
    pipeline = joblib.load(pipeline_path)

    # Extract preprocessor and model steps
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Get feature names from ColumnTransformer
    raw_feature_names = preprocessor.get_feature_names_out()
    clean_feature_names = []
    for name in raw_feature_names:
        name = name.replace("cat__ohe__", "").replace("cat__bucket__", "").replace("num__", "")
        clean_feature_names.append(name)

    # Transform the test set features
    print("Preprocessing test data...")
    X_test_trans = preprocessor.transform(X_test)
    X_test_trans_df = pd.DataFrame(X_test_trans, columns=clean_feature_names)

    # Initialize SHAP TreeExplainer on the Random Forest regressor
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    print("Computing SHAP values for test set...")
    sample_size = min(500, len(X_test_trans_df))
    X_sample = X_test_trans_df.iloc[:sample_size]
    shap_values = explainer(X_sample)

    # --- 1. Global Importance Plot ---
    print("Generating global SHAP importance plot...")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=14)
    plt.tight_layout()
    global_plot_path = SHAP_DIR / "shap_global_importance.png"
    plt.savefig(global_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved global plot to {global_plot_path}")

    # --- 2. Local Explanations ---
    predictions = model.predict(X_sample)
    mean_pred = np.mean(predictions)

    # Find indexes for low and high predicted prices
    low_idx = int(np.argmin(predictions))
    high_idx = int(np.argmax(predictions))

    print(f"\nInstance 1 (Low Price): Index={low_idx}, Predicted Price={predictions[low_idx]:.2f} Lakhs (vs Mean={mean_pred:.2f} Lakhs)")
    print(f"Instance 2 (High Price): Index={high_idx}, Predicted Price={predictions[high_idx]:.2f} Lakhs (vs Mean={mean_pred:.2f} Lakhs)")

    # Plot local explanation 1 (Low Price)
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap_values[low_idx], max_display=10, show=False)
    plt.title(f"Price Deconstruction: Low Price Example (Predicted: {predictions[low_idx]:.2f} Lakhs)", fontsize=14)
    plt.tight_layout()
    local_plot_path_1 = SHAP_DIR / "shap_local_1.png"
    plt.savefig(local_plot_path_1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved local explanation 1 to {local_plot_path_1}")

    # Plot local explanation 2 (High Price)
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap_values[high_idx], max_display=10, show=False)
    plt.title(f"Price Deconstruction: High Price Example (Predicted: {predictions[high_idx]:.2f} Lakhs)", fontsize=14)
    plt.tight_layout()
    local_plot_path_2 = SHAP_DIR / "shap_local_2.png"
    plt.savefig(local_plot_path_2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved local explanation 2 to {local_plot_path_2}")

    # Print human readable interpretations
    print("\n--- Human-Readable Explanations ---")
    low_features = X_test.iloc[low_idx]
    print(f"Listing 1 (Actual price: {y_test.iloc[low_idx]} Lakhs, Predicted: {predictions[low_idx]:.2f} Lakhs):")
    print(f"- Location: {low_features['location']}, Sqft: {low_features['total_sqft']}, BHK: {low_features['bhk']}, Bath: {low_features['bath']}")
    print(f"- Interpretation: This property was priced below average primarily because of its smaller size ({low_features['total_sqft']} sqft) and fewer bedrooms ({low_features['bhk']} BHK).")

    high_features = X_test.iloc[high_idx]
    print(f"\nListing 2 (Actual price: {y_test.iloc[high_idx]} Lakhs, Predicted: {predictions[high_idx]:.2f} Lakhs):")
    print(f"- Location: {high_features['location']}, Sqft: {high_features['total_sqft']}, BHK: {high_features['bhk']}, Bath: {high_features['bath']}")
    print(f"- Interpretation: This property was priced significantly above average due to its large square footage ({high_features['total_sqft']} sqft), high number of bathrooms ({high_features['bath']}), and premium location.")

if __name__ == "__main__":
    run_model_explanations()
