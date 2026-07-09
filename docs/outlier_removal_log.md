# Outlier Removal Log

All outlier thresholds were calculated and applied **strictly using training-split statistics only** to prevent target and test-set leakage.

- **Initial Training Split Rows**: 10558
- **Outlier Removal Breakdown**:

| Outlier Filter Rule | Rows Removed | Remaining Rows |
| :--- | :---: | :---: |
| 1. Sqft-per-BHK Filter (total_sqft/bhk < 300) | 597 | 9961 |
| 2. Per-location Price-per-Sqft Filter (mean +/- 1 std) | 2195 | 7766 |
| 3. BHK Price-Consistency Check | 1379 | 6387 |
| 4. Bathroom Sanity Filter (bath > bhk + 2) | 4 | 6383 |

- **Total Training Rows Removed**: 4175 (39.54%)
- **Final Cleaned Training Rows**: 6383
