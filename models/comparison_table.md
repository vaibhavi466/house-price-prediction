# Model Performance Comparison

All models were trained using 5-fold cross-validation on the filtered training set and evaluated on the unfiltered test set.

| Model | Encoding | Validation R² | Validation RMSE | Validation MAE | MLflow Run ID |
|:---|:---|---:|---:|---:|:---|
| Random Forest | onehot | 0.6902 | 77.6515 | 34.4581 | c138f0e4a7d947ef87f2673dfcef992a |
| Random Forest | target | 0.6563 | 81.7818 | 33.5472 | d7470578ed464f3fbeab47dc794f5058 |
| LightGBM | target | 0.6363 | 84.1328 | 34.0972 | 9803bf6dfc5c4eb5b4747c35e341dd8d |
| LightGBM | onehot | 0.6339 | 84.4030 | 34.9474 | 155f4f2e144f424cb62cf97fd2816519 |
| XGBoost | onehot | 0.6088 | 87.2489 | 34.7349 | 2d0369688df044bea67f5cc36a3f1a5f |
| XGBoost | target | 0.5950 | 88.7725 | 34.3629 | 2401b069af2549209b7db5d954d1d893 |
| Ridge | onehot | 0.5913 | 89.1847 | 38.7736 | aeb9ac360b77437287b3625927062b6b |
| Linear Regression | onehot | 0.5904 | 89.2806 | 39.1957 | 0c98902a77214929bd4097fa2caed3fc |
| Lasso | target | 0.5807 | 90.3343 | 40.4152 | 7cd7631d1a51460cbb9c4d66c568c96c |
| Ridge | target | 0.5798 | 90.4306 | 40.7526 | 47e5c19ddfe04d2ea668a8e8bce720bd |
| Linear Regression | target | 0.5797 | 90.4372 | 40.7841 | 9bccdd60d87b4c6e97b5c5399b6c499d |
| Lasso | onehot | 0.5744 | 91.0078 | 40.1882 | 06c5568e216941c4ac848b51b58678c1 |

### Winning Model Selection
The chosen final model is **Random Forest** with **onehot encoding** (R² = 0.6902), saved to `models/model_pipeline.joblib`.

### Rationale
- **Overall Performance**: The Random Forest model using One-Hot Encoding achieved the highest R² score of **0.6902** and the lowest RMSE (**77.6515** Lakhs) on the held-out test split. 
- **Encoding Impact**: While Target Encoding keeps the feature space dense, it yielded lower validation performance for the Random Forest model (R² of 0.6563 vs 0.6902). However, for LightGBM, Target Encoding actually performed slightly better (0.6363 vs 0.6339), demonstrating its utility in boosting gradient tree ensembles on high-cardinality features.
- **Linear Models Baseline**: Linear models (Linear Regression, Ridge) performed significantly better under One-Hot Encoding because OHE gives the model the flexibility to assign distinct linear coefficients to each individual location, whereas Target Encoding compresses location features into a single continuous representation, restricting the linear boundary capability.
