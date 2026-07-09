# 90-Second Verbal Walkthrough

*Practice reading this aloud. Target: 85–95 seconds at a measured pace.*

---

## The Script

**"So the project is a Bangalore house price predictor — end-to-end, from raw CSV to a live containerized API with a web frontend.**

**The business problem**: real estate buyers and agents in Bangalore have almost no reliable way to sanity-check asking prices across the city's 500-plus micro-locations. The goal was to build something defensible — not just a tutorial notebook, but a pipeline you could hand to a new engineer and they'd know exactly what every step does and why.

**The data had real cleaning challenges.** The sqft column was a mess — range strings like '1200–1500', unit suffixes like 'Sq. Meter', and plain nulls. I wrote a `convert_sqft()` function that handles ranges by averaging the bounds and explicitly returns None for non-numeric units rather than silently coercing them — so those rows get dropped, not poisoned.

**For outlier removal**, I applied four domain-driven rules — sqft-per-BHK sanity, per-location price-per-sqft standard deviation bands, BHK price consistency (a 3-BHK shouldn't cost less per sqft than a 2-BHK in the same area), and a bathroom cap. Critically, all thresholds were computed on the training split only — the test set was left raw on purpose, so the evaluation metrics reflect real-world noise.

**For feature engineering**, I ran an actual encoding experiment: trained every model under both One-Hot Encoding and Target Encoding and compared. One-Hot won for Random Forest (R² 0.69 vs 0.66), but Target Encoding actually edged out LightGBM slightly. This matters because naive mean-encoding on the full dataset before splitting leaks target information — I used sklearn's TargetEncoder with internal cross-fitting to avoid that.

**Model comparison** — I trained six models (Linear Regression, Ridge, Lasso, Random Forest, XGBoost, LightGBM) each under both encodings, using 5-fold GridSearchCV, and tracked every run in MLflow. Random Forest with One-Hot Encoding won: R² of 0.69, RMSE of 77.65 Lakhs on the held-out test set. I didn't just pick the top metric blindly — the runner-up was LightGBM at 0.64, and Random Forest's interpretability via SHAP TreeExplainer was a genuine factor.

**For interpretability**, I added SHAP to the API response: every `/predict` call returns the top-3 contributing features for that specific prediction in real time. So the frontend shows not just a price, but *why* — property size was +18 Lakhs, location was −1.4 Lakhs.

**On the bias side**: I computed residuals grouped by location tier. The model is nearly unbiased for standard listings (mean residual −1.3 Lakhs) but systematically under-predicts premium-tier properties by an average of 219 Lakhs — about a third of their actual value. That's an honest finding, and it's in the limitations section.

**The API** is FastAPI with Pydantic v2 validation — bad inputs get a clean 422, not a stack trace. The pipeline and SHAP explainer load once at startup via the lifespan hook, not per request. And the whole thing runs in a Docker container with a non-root user and a healthcheck, with CI on GitHub Actions running lint, tests, and a full Docker build + /health + /predict smoke test on every push."**

---

## Key Numbers to Have Ready

| Fact | Value |
|:---|:---|
| Raw dataset rows | 13,320 |
| After cleaning | 13,198 |
| Training split (before outliers) | 10,558 |
| Training split (after outliers) | 6,383 (39.5% removed) |
| Test split (unfiltered) | 2,640 |
| Winning model | Random Forest + One-Hot Encoding |
| Test R² | 0.6902 |
| Test RMSE | 77.65 Lakhs |
| Test MAE | 34.46 Lakhs |
| Premium-tier mean residual | +218.72 Lakhs (under-prediction) |
| Standard-tier mean residual | −1.33 Lakhs |
| pytest coverage (core files) | 82–97% |
| CI steps | lint + test + Docker build + smoke test |
