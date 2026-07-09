# Encoding Experiment: One-Hot Encoding vs. Target Encoding

This document tracks the results of our experiment comparing **One-Hot Encoding** and **Target Encoding** for the high-cardinality `location` column.

## Rationale
- **One-Hot Encoding (OHE)**: Expands each category into its own binary column. Can cause high dimensionality and sparse features, which increases model size and may lead to overfitting for high-cardinality columns (e.g. hundreds of locations).
- **Target Encoding**: Replaces each category with the expected value of the target (house price) for that category. It keeps the feature space dense (1 column). However, if computed naively, it causes severe target leakage. We use `sklearn.preprocessing.TargetEncoder`, which employs cross-fitting (out-of-fold target computation) internally to ensure leakage safety.

## Comparison Results
The models were trained on the cleaned training split (6,383 rows) and evaluated on the unfiltered test split (2,640 rows).

| Model | Encoding | Validation R² | Validation RMSE | Validation MAE | MLflow Run ID |
|---|---|---|---|---|---|
| Random Forest | One-Hot Encoding | 0.6902 | 77.6515 | 34.4581 | c138f0e4a7d947ef87f2673dfcef992a |
| Random Forest | Target Encoding | 0.6563 | 81.7818 | 33.5472 | d7470578ed464f3fbeab47dc794f5058 |
| LightGBM | Target Encoding | 0.6363 | 84.1328 | 34.0972 | 9803bf6dfc5c4eb5b4747c35e341dd8d |
| LightGBM | One-Hot Encoding | 0.6339 | 84.4030 | 34.9474 | 155f4f2e144f424cb62cf97fd2816519 |
| XGBoost | One-Hot Encoding | 0.6088 | 87.2489 | 34.7349 | 2d0369688df044bea67f5cc36a3f1a5f |
| XGBoost | Target Encoding | 0.5950 | 88.7725 | 34.3629 | 2401b069af2549209b7db5d954d1d893 |
| Ridge | One-Hot Encoding | 0.5913 | 89.1847 | 38.7736 | aeb9ac360b77437287b3625927062b6b |
| Linear Regression | One-Hot Encoding | 0.5904 | 89.2806 | 39.1957 | 0c98902a77214929bd4097fa2caed3fc |
| Lasso | Target Encoding | 0.5807 | 90.3343 | 40.4152 | 7cd7631d1a51460cbb9c4d66c568c96c |
| Ridge | Target Encoding | 0.5798 | 90.4306 | 40.7526 | 47e5c19ddfe04d2ea668a8e8bce720bd |
| Linear Regression | Target Encoding | 0.5797 | 90.4372 | 40.7841 | 9bccdd60d87b4c6e97b5c5399b6c499d |
| Lasso | One-Hot Encoding | 0.5744 | 91.0078 | 40.1882 | 06c5568e216941c4ac848b51b58678c1 |

## Key Findings
- **Performance Winner**: Random Forest with One-Hot Encoding yielded the highest $R^2$ score of **0.6902** and lowest RMSE of **77.6515**.
- **Target Encoding Trade-off**: Target encoding performed very competitively, especially for tree-based ensembles (LightGBM actually saw slightly *better* R² under target encoding: 0.6363 vs 0.6339). For Random Forest, target encoding yielded a slightly better MAE (33.5472) compared to OHE (34.4581), showing that target encoding does not suffer from high-dimensional sparsity issues and can make more robust average predictions.
- **Linear Models & OHE**: Linear models (Linear Regression, Ridge) performed noticeably better under One-Hot Encoding compared to Target Encoding because OHE allows the linear model to learn discrete, independent price offsets for each individual location, whereas target encoding forces a single linear relationship between the mean location target and the price, which limits model flexibility.
