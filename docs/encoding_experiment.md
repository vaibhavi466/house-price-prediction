# Encoding Experiment: One-Hot Encoding vs. Target Encoding

This document tracks the results of our experiment comparing **One-Hot Encoding** and **Target Encoding** for the high-cardinality `location` column.

## Rationale
- **One-Hot Encoding (OHE)**: Expands each category into its own binary column. Can cause high dimensionality and sparse features, which increases model size and may lead to overfitting for high-cardinality columns (e.g. hundreds of locations).
- **Target Encoding**: Replaces each category with the expected value of the target (house price) for that category. It keeps the feature space dense (1 column). However, if computed naively, it causes severe target leakage. We use `sklearn.preprocessing.TargetEncoder`, which employs cross-fitting (out-of-fold target computation) internally to ensure leakage safety.

## Comparison Results
*To be populated during Phase 4 model training.*

| Model | Encoding | Validation R² | Validation RMSE | Validation MAE | MLflow Run ID |
|---|---|---|---|---|---|
| Linear Regression | One-Hot Encoding | | | | |
| Linear Regression | Target Encoding | | | | |
| Ridge Regression | One-Hot Encoding | | | | |
| Ridge Regression | Target Encoding | | | | |
| Lasso Regression | One-Hot Encoding | | | | |
| Lasso Regression | Target Encoding | | | | |
| Random Forest | One-Hot Encoding | | | | |
| Random Forest | Target Encoding | | | | |
| XGBoost | One-Hot Encoding | | | | |
| XGBoost | Target Encoding | | | | |
| LightGBM | One-Hot Encoding | | | | |
| LightGBM | Target Encoding | | | | |

## Key Findings
- *Which encoding performed better and why?*
- *Analysis of model size and latency trade-offs.*
