# Limitations & Modeling Assumptions

This document discusses model limitations, bias, and methodological choices made during model evaluation.

## VIF and Residual Plot Feature Choices
During Phase 4 baseline validation, we evaluated assumptions on two different representations of our Linear Regression model:
- **Variance Inflation Factor (VIF) on Target-Encoded Features**: One-Hot Encoding (OHE) expands categories into hundreds of separate binary columns. Because these columns are sparse, binary, and mutually exclusive, they inherently exhibit high multicollinearity. Running a VIF check on OHE features yields infinite or artificially inflated VIF scores that do not reflect true predictor relationships. To perform a meaningful VIF analysis on our core numeric predictors (`total_sqft`, `bath`, `bhk`), we calculated VIF on a dense **Target-Encoded** representation of the features. This allowed us to assess whether collinearity exists between the location pricing effect and property size or bedroom/bathroom counts.
- **Residual Analysis on One-Hot Encoded Baseline**: The residual scatter plot (`docs/eda_plots/linear_residuals.png`) was plotted using predictions from the **One-Hot Encoded** Linear Regression baseline model. Since OHE was our primary baseline (yielding a slightly higher $R^2$ of 0.5904 compared to 0.5797 for Target Encoding), checking its residuals was critical to verify if our best-performing linear baseline satisfied the assumptions of homoscedasticity and linearity.

## Model Limitations
1. **Heteroscedasticity**: The residual plot shows a clear funnel shape, indicating that prediction errors grow larger for higher-priced properties. Standard linear baselines are therefore less reliable in the premium/luxury segment.
2. **High-Cardinality Locations**: Although the location bucketing transformer handles low-frequency listings by mapping them to "other", properties in extremely rare locations will rely entirely on the average global fallback price, leading to lower local prediction accuracy.
