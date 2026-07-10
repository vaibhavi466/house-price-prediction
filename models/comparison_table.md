# Model Performance Comparison

All models were trained using 5-fold cross-validation on the filtered training set and evaluated on the unfiltered test set.

| Model             | Encoding   |   Validation R² |   Validation RMSE |   Validation MAE | MLflow Run ID                    |
|:------------------|:-----------|----------------:|------------------:|-----------------:|:---------------------------------|
| Random Forest     | onehot     |        0.690151 |           77.6515 |          34.4581 | 73978a0e7c354b8fae30233d0f23e352 |
| Random Forest     | target     |        0.656313 |           81.7818 |          33.5472 | 02bf26553f834c158ce1d676a4f9905e |
| LightGBM          | target     |        0.636268 |           84.1328 |          34.0972 | ffe0081b130c494baa56a2631afc1ab3 |
| LightGBM          | onehot     |        0.633928 |           84.403  |          34.9474 | 653a86f75e204704b20ea197dc961810 |
| XGBoost           | onehot     |        0.608826 |           87.2489 |          34.7349 | 48bedb9235dc4d68872867f46799486b |
| XGBoost           | target     |        0.595044 |           88.7725 |          34.3629 | 3e13bd82fec34b41bceda05f608cea7a |
| Ridge             | onehot     |        0.591276 |           89.1847 |          38.7736 | 5448156ec247486abe3798e4e688a074 |
| Linear Regression | onehot     |        0.590396 |           89.2806 |          39.1957 | bbaade32eaea4d6f8c7e1c89579989c0 |
| Lasso             | target     |        0.58067  |           90.3343 |          40.4152 | 5ed7ef07d06b4f90a22c8c0321136730 |
| Ridge             | target     |        0.579776 |           90.4306 |          40.7526 | fd3c6c05c9804c48a614300ef3e55606 |
| Linear Regression | target     |        0.579714 |           90.4372 |          40.7841 | bebf700f70fb4737b61a137cdfc8e3fe |
| Lasso             | onehot     |        0.574395 |           91.0078 |          40.1882 | 290283f8debe41cea7d65810d66460de |

### Winning Model Selection
The chosen final model is **Random Forest** with **onehot encoding** (R² = 0.6902), saved to `models/model_pipeline.joblib`.
