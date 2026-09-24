"""
main.py
=======
Main orchestrator for Module 2: Analytics & Machine Learning Pipeline.
Executes the full pipeline:
1. Data Ingestion & Profiling (sns.load_dataset with offline CSV fallback)
2. Missing Value Analysis & Cleaning Decisions
3. Univariate & Bivariate Analysis (IQR outliers, Skewness, 6x6 Heatmap)
4. Multivariate Visualizations & 2-4 Sentence Interpretations
5. Z-Score Standardization Sanity Check
6. Classification Modeling (Logistic Regression, Decision Tree with plot_tree, Random Forest)
7. Class Imbalance Comparison (Baseline vs Balanced vs SMOTE)
8. Random Forest GridSearchCV with oob_score=True
9. Multivariate Fare Regression (MAE, RMSE, R2, Adjusted R2, Heteroscedasticity Analysis)
10. Model Comparison Report & Production Deployment Recommendation
11. End-to-End Pipeline Persistence & Verification with Raw Sample
"""

import os
import sys
import time

# Ensure standard output can handle UTF-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from analysis import run_eda_pipeline
from modeling import run_modeling_pipeline


def run_full_analytics_module():
    start_time = time.time()
    print("=" * 80)
    print("ZEPTO DATA & AI PLATFORM - MODULE 2: ANALYTICS & ML PIPELINE")
    print("=" * 80)

    print("\n[PART A] Executing Exploratory Data Analysis & Statistical Story...")
    df_clean = run_eda_pipeline()

    print("\n[PART B] Executing Predictive Modeling, Tuning, and Evaluation Pipeline...")
    run_modeling_pipeline()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Module 2 Pipeline execution completed successfully in {elapsed:.2f} seconds.")
    print("All plots saved to: analytics/plots/")
    print("All reports saved to: analytics/outputs/")
    print("Saved pipeline model in: analytics/models/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_full_analytics_module()
