# Encoding Experiment: One-Hot Encoding vs. Target Encoding

This document tracks the results of our experiment comparing **One-Hot Encoding** and **Target Encoding** for the high-cardinality `location` column.

## Rationale
- **One-Hot Encoding (OHE)**: Expands each category into its own binary column. Can cause high dimensionality and sparse features, which increases model size and may lead to overfitting for high-cardinality columns (e.g. hundreds of locations).
- **Target Encoding**: Replaces each category with the expected value of the target (house price) for that category. It keeps the feature space dense (1 column). However, if computed naively, it causes severe target leakage. We use `sklearn.preprocessing.TargetEncoder`, which employs cross-fitting (out-of-fold target computation) internally to ensure leakage safety.

## Comparison Results
The models were trained on the cleaned training split (6,383 rows) and evaluated on the unfiltered test split (2,640 rows).

| Model | Encoding | Validation R² | Validation RMSE | Validation MAE | MLflow Run ID |
|---|---|---|---|---|---|
| Random Forest | One-Hot Encoding | 0.6902 | 77.6515 | 34.4581 | 73978a0e7c354b8fae30233d0f23e352 |
| Random Forest | Target Encoding | 0.6563 | 81.7818 | 33.5472 | 02bf26553f834c158ce1d676a4f9905e |
| LightGBM | Target Encoding | 0.6363 | 84.1328 | 34.0972 | ffe0081b130c494baa56a2631afc1ab3 |
| LightGBM | One-Hot Encoding | 0.6339 | 84.4030 | 34.9474 | 653a86f75e204704b20ea197dc961810 |
| XGBoost | One-Hot Encoding | 0.6088 | 87.2489 | 34.7349 | 48bedb9235dc4d68872867f46799486b |
| XGBoost | Target Encoding | 0.5950 | 88.7725 | 34.3629 | 3e13bd82fec34b41bceda05f608cea7a |
| Ridge | One-Hot Encoding | 0.5913 | 89.1847 | 38.7736 | 5448156ec247486abe3798e4e688a074 |
| Linear Regression | One-Hot Encoding | 0.5904 | 89.2806 | 39.1957 | bbaade32eaea4d6f8c7e1c89579989c0 |
| Lasso | Target Encoding | 0.5807 | 90.3343 | 40.4152 | 5ed7ef07d06b4f90a22c8c0321136730 |
| Ridge | Target Encoding | 0.5798 | 90.4306 | 40.7526 | fd3c6c05c9804c48a614300ef3e55606 |
| Linear Regression | Target Encoding | 0.5797 | 90.4372 | 40.7841 | bebf700f70fb4737b61a137cdfc8e3fe |
| Lasso | One-Hot Encoding | 0.5744 | 91.0078 | 40.1882 | 290283f8debe41cea7d65810d66460de |

## Key Findings
- **Performance Winner**: Random Forest with One-Hot Encoding yielded the highest $R^2$ score of **0.6902** and lowest RMSE of **77.6515**.
- **Target Encoding Trade-off**: Target encoding performed very competitively, especially for tree-based ensembles (LightGBM actually saw slightly *better* R² under target encoding: 0.6363 vs 0.6339). For Random Forest, target encoding yielded a slightly better MAE (33.5472) compared to OHE (34.4581), showing that target encoding does not suffer from high-dimensional sparsity issues and can make more robust average predictions.
- **Linear Models & OHE**: Linear models (Linear Regression, Ridge) performed noticeably better under One-Hot Encoding compared to Target Encoding because OHE allows the linear model to learn discrete, independent price offsets for each individual location, whereas target encoding forces a single linear relationship between the mean location target and the price, which limits model flexibility.
