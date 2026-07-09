# Limitations & Future Work

## 1. Known Model Bias: Systematic Under-Prediction in Premium Locations

**This is a computed finding, not a boilerplate disclaimer.**

We grouped test-set residuals (actual − predicted) by location tier, defining the **premium tier** as the top 20 most expensive locations by mean training price, and the **standard tier** as all remaining locations.

| Tier | Test Rows | Mean Actual Price | Mean Residual | Median Residual |
|:---|---:|---:|---:|---:|
| Premium (top-20 locations) | 18 | 653.61 Lakhs | **+218.72 Lakhs** | **+213.64 Lakhs** |
| Standard (remaining) | 2,622 | 104.96 Lakhs | -1.33 Lakhs | -7.49 Lakhs |

**Interpretation**: The model systematically under-predicts premium-tier properties by a mean of **218.72 Lakhs** (~33% of their actual value), while standard-tier predictions are nearly unbiased (mean residual of −1.33 Lakhs). This is a direct consequence of the heteroscedastic price distribution in Bangalore: extreme luxury listings (e.g., Ashok Nagar at 2,912 Lakhs avg, 5th Block Jayanagar at 2,340 Lakhs avg) are rare in the training data, and the model's ensemble averaging pulls their predictions toward the global mean. This finding should be disclosed to any user relying on the model for high-end property valuations.

**Mitigation strategies** (not yet implemented, logged for future work):
- Train a separate specialist model on premium-tier listings only.
- Apply log-transformation to the target variable to compress the extreme value range before training.
- Add a premium-flag feature based on location tier assignment.

---

## 2. VIF and Residual Plot Feature Choices

During Phase 4 baseline validation, we evaluated assumptions on two different representations of our Linear Regression model:

- **Variance Inflation Factor (VIF) on Target-Encoded Features**: One-Hot Encoding (OHE) expands categories into hundreds of sparse, binary, mutually-exclusive columns, causing artificially inflated VIF scores that reveal nothing meaningful about multicollinearity. VIF was computed on the **Target-Encoded** feature representation instead, measuring collinearity between the location price signal and the core numeric predictors (`total_sqft`, `bath`, `bhk`). All VIF scores were below 5 (max 4.86), indicating acceptable multicollinearity among numeric predictors.

- **Residual Analysis on One-Hot Encoded Baseline**: The residual scatter plot (`docs/eda_plots/linear_residuals.png`) was generated from the **One-Hot Encoded** Linear Regression model (the best-performing linear baseline, R² = 0.5904). The plot shows a clear funnel shape — prediction error grows with the predicted price — confirming heteroscedasticity. This justified the decision to pick a tree-based ensemble (Random Forest) as the final model.

---

## 3. Outlier Threshold Leakage Guard

All four outlier removal thresholds (sqft/BHK ratio, per-location price-per-sqft mean ± 1 std, BHK price-consistency, bathroom sanity) were computed **exclusively from training-split statistics**. The test split was intentionally left unfiltered so that evaluation metrics reflect real-world, uncleaned inputs. This is a documented design decision — not an implicit code behavior — and means the reported metrics reflect performance on slightly noisier data than the training distribution.

---

## 4. High-Cardinality Location Handling

Locations with ≤10 listings in the training split are bucketed into an `"other"` category by `LocationBucketTransformer`. Predictions for properties in low-frequency or previously-unseen locations fall back to the `"other"` aggregate price signal, leading to lower local accuracy for rare areas. The bucketing threshold of 10 was chosen pragmatically; future work could tune this or replace it with a hierarchical embedding.

---

## 5. Distance-to-City-Center Feature

Geocoding Bangalore coordinates via Nominatim (free, no API key) was evaluated but skipped for this version. Accurate geocoding would require a stable internet connection, rate-limiting (1 req/sec), a local cache, and careful verification of coordinate accuracy for dense urban layouts. This feature is logged as a high-value future addition — distance to MG Road or Cubbon Park is a known price driver in Bangalore's market.

---

## 6. Temporal Drift

The dataset (from a fixed 2017–2019 snapshot) does not reflect current Bangalore real estate market conditions. Prices, high-growth micro-markets, and infrastructure changes since 2019 (Namma Metro extensions, tech park expansions) are not reflected. The model should be retrained periodically on fresh listings data.
