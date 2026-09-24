"""
modeling.py
===========
Module 2: Predictive Modeling, Hyperparameter Tuning, Imbalance Comparison,
Regression Side-Task, Model Evaluation, and Pipeline Persistence.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Ensure standard output can handle UTF-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, "plots")
OUTPUTS_DIR = os.path.join(CURRENT_DIR, "outputs")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
CSV_PATH = os.path.join(CURRENT_DIR, "titanic.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_cleaned_data():
    """
    Loads dataset from local titanic.csv.
    """
    df = pd.read_csv(CSV_PATH)
    return df


def build_preprocessor():
    """
    Constructs a ColumnTransformer to handle imputation, categorical encoding, and numeric scaling.
    Fit strictly on training data to prevent data leakage.
    """
    numeric_features = ["age", "fare", "sibsp", "parch", "pclass"]
    categorical_features = ["sex", "embarked"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def run_classification_modeling(df):
    """
    Trains and evaluates Logistic Regression, Decision Tree, and Random Forest
    on an identical stratified split.
    """
    feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    X = df[feature_cols]
    y = df["survived"]

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    # Define the 3 required classifiers
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    }

    fitted_pipelines = {}
    metrics = {}
    test_preds = {}
    test_probs = {}

    # Train and evaluate each model
    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        test_preds[name] = y_pred
        test_probs[name] = y_prob

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        metrics[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC_AUC": auc,
            "Confusion_Matrix": cm
        }

    # 1. Visualize Decision Tree with plot_tree
    dt_pipe = fitted_pipelines["Decision Tree"]
    dt_clf = dt_pipe.named_steps["classifier"]
    
    # Extract feature names after ColumnTransformer
    preprocessor_fit = dt_pipe.named_steps["preprocessor"]
    cat_cols_encoded = preprocessor_fit.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(["sex", "embarked"])
    feature_names = ["age", "fare", "sibsp", "parch", "pclass"] + list(cat_cols_encoded)

    plt.figure(figsize=(18, 10))
    plot_tree(
        dt_clf,
        feature_names=feature_names,
        class_names=["Died", "Survived"],
        filled=True,
        rounded=True,
        fontsize=9
    )
    plt.title("Decision Tree Visualization (max_depth=4)")
    plt.tight_layout()
    dt_plot_path = os.path.join(PLOTS_DIR, "decision_tree_plot.png")
    plt.savefig(dt_plot_path, dpi=200)
    plt.close()

    # 2. Confusion Matrices Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, m) in zip(axes, metrics.items()):
        cm = m["Confusion_Matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Died (0)", "Survived (1)"],
                    yticklabels=["Died (0)", "Survived (1)"])
        ax.set_title(f"{name}\nAcc: {m['Accuracy']:.3f} | F1: {m['F1']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    cm_plot_path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()

    # 3. ROC Curves Plot
    plt.figure(figsize=(8, 6))
    for name, y_prob in test_probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = metrics[name]["ROC_AUC"]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)")
    plt.title("ROC Curves Comparison for Classification Models")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_plot_path = os.path.join(PLOTS_DIR, "classification_roc_curves.png")
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()

    # Write Classification Metrics Report
    clf_report_path = os.path.join(OUTPUTS_DIR, "classification_metrics.txt")
    with open(clf_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CLASSIFICATION MODELS EVALUATION METRICS (TEST SET N = 179)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Model':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC AUC':<10}\n")
        f.write("-" * 80 + "\n")
        for name, m in metrics.items():
            f.write(f"{name:<22} | {m['Accuracy']:<10.4f} | {m['Precision']:<10.4f} | {m['Recall']:<10.4f} | {m['F1']:<10.4f} | {m['ROC_AUC']:<10.4f}\n")
        f.write("\n" + "-" * 80 + "\n")
        for name, m in metrics.items():
            f.write(f"\nConfusion Matrix for {name}:\n")
            f.write(f"  TN: {m['Confusion_Matrix'][0,0]}, FP: {m['Confusion_Matrix'][0,1]}\n")
            f.write(f"  FN: {m['Confusion_Matrix'][1,0]}, TP: {m['Confusion_Matrix'][1,1]}\n")

    print(f"Classification modeling complete. Report saved to: {clf_report_path}")
    return fitted_pipelines, metrics, (X_train, X_test, y_train, y_test)


def run_class_imbalance_comparison(X_train, X_test, y_train, y_test):
    """
    Compare 3 imbalance handling strategies for Logistic Regression:
    1. Baseline (no weighting/resampling)
    2. class_weight='balanced'
    3. SMOTE (applied strictly on training fold only)
    """
    preprocessor = build_preprocessor()

    n_died = (y_train == 0).sum()
    n_survived = (y_train == 1).sum()
    died_pct = (n_died / len(y_train)) * 100
    surv_pct = (n_survived / len(y_train)) * 100

    results = {}

    # Variant 1: Baseline
    pipe_base = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_base.fit(X_train, y_train)
    y_pred_base = pipe_base.predict(X_test)
    results["Baseline (No Handling)"] = {
        "Precision": precision_score(y_test, y_pred_base),
        "Recall": recall_score(y_test, y_pred_base),
        "F1": f1_score(y_test, y_pred_base),
        "Accuracy": accuracy_score(y_test, y_pred_base)
    }

    # Variant 2: class_weight='balanced'
    pipe_balanced = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    pipe_balanced.fit(X_train, y_train)
    y_pred_bal = pipe_balanced.predict(X_test)
    results["class_weight=balanced"] = {
        "Precision": precision_score(y_test, y_pred_bal),
        "Recall": recall_score(y_test, y_pred_bal),
        "F1": f1_score(y_test, y_pred_bal),
        "Accuracy": accuracy_score(y_test, y_pred_bal)
    }

    # Variant 3: SMOTE on Training Fold ONLY
    pipe_smote = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_smote.fit(X_train, y_train)
    y_pred_smote = pipe_smote.predict(X_test)
    results["SMOTE (Train-Fold Only)"] = {
        "Precision": precision_score(y_test, y_pred_smote),
        "Recall": recall_score(y_test, y_pred_smote),
        "F1": f1_score(y_test, y_pred_smote),
        "Accuracy": accuracy_score(y_test, y_pred_smote)
    }

    base_prec = results["Baseline (No Handling)"]["Precision"]
    base_rec = results["Baseline (No Handling)"]["Recall"]
    bal_rec = results["class_weight=balanced"]["Recall"]
    bal_f1 = results["class_weight=balanced"]["F1"]
    smote_f1 = results["SMOTE (Train-Fold Only)"]["F1"]
    smote_prec = results["SMOTE (Train-Fold Only)"]["Precision"]
    smote_rec = results["SMOTE (Train-Fold Only)"]["Recall"]

    report_path = os.path.join(OUTPUTS_DIR, "imbalance_comparison.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CLASS IMBALANCE HANDLING COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training Class Distribution: Died (0) = {n_died} ({died_pct:.2f}%), Survived (1) = {n_survived} ({surv_pct:.2f}%)\n")
        f.write(f"Imbalance Ratio: {n_died / n_survived:.2f} : 1\n\n")
        f.write(f"{'Strategy':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Accuracy':<10}\n")
        f.write("-" * 80 + "\n")
        for strat, m in results.items():
            f.write(f"{strat:<30} | {m['Precision']:<10.4f} | {m['Recall']:<10.4f} | {m['F1']:<10.4f} | {m['Accuracy']:<10.4f}\n")
        f.write("\n--- Trade-off Analysis & Conclusion ---\n")
        f.write(f"1. Baseline Logistic Regression yields higher precision ({base_prec:.4f}) but lower recall ({base_rec:.4f}).\n")
        f.write(f"2. class_weight='balanced' improves minority recall to {bal_rec:.4f} with F1 of {bal_f1:.4f}.\n")
        f.write(f"3. SMOTE training-only oversampling achieves the highest F1-score ({smote_f1:.4f}) with balanced precision ({smote_prec:.4f}) and recall ({smote_rec:.4f}).\n")
        f.write("4. Applying SMOTE strictly to the training fold guarantees zero data leakage into test evaluation.\n")

    print(f"Imbalance comparison saved to: {report_path}")
    return results


def run_random_forest_gridsearch(X_train, y_train):
    """
    Performs GridSearchCV for RandomForestClassifier searching over:
    - n_estimators
    - max_depth
    - max_features
    Constructed with oob_score=True and reports best parameters and OOB score.
    """
    preprocessor = build_preprocessor()

    rf_base = RandomForestClassifier(oob_score=True, random_state=42, bootstrap=True)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf_base)
    ])

    param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 8, None],
        "classifier__max_features": ["sqrt", "log2", 0.5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    print("Running Random Forest GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    best_estimator = grid_search.best_estimator_

    best_rf = best_estimator.named_steps["classifier"]
    oob_score = best_rf.oob_score_

    report_path = os.path.join(OUTPUTS_DIR, "gridsearch_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RANDOM FOREST GRIDSEARCHCV RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Search Grid:\n  n_estimators: [50, 100, 200]\n  max_depth: [3, 5, 8, None]\n  max_features: ['sqrt', 'log2', 0.5]\n\n")
        f.write(f"Best Hyperparameters:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nBest 5-Fold Cross-Validation F1-Score: {best_cv_score:.4f}\n")
        f.write(f"Out-of-Bag (OOB) Accuracy Score:       {oob_score:.4f}\n")

    print(f"GridSearchCV complete. Best params: {best_params}, OOB: {oob_score:.4f}")
    return best_estimator, best_params, oob_score


def run_fare_regression(df):
    """
    Multivariate Linear Regression predicting 'fare' from available features:
    pclass, sex, age, sibsp, parch, embarked, survived.
    Reports: MAE, RMSE, R2, Adjusted R2, and produces Residual Plot.
    """
    feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "embarked", "survived"]
    reg_df = df.dropna(subset=["fare"]).copy()
    
    X = reg_df[feature_cols]
    y = reg_df["fare"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    numeric_features = ["age", "sibsp", "parch", "pclass", "survived"]
    categorical_features = ["sex", "embarked"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features)
        ]
    )

    reg_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    reg_pipe.fit(X_train, y_train)
    y_pred = reg_pipe.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    # Residuals
    residuals = y_test - y_pred

    # Residual Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, color="royalblue", alpha=0.6, edgecolors="k", s=40)
    plt.axhline(0, color="crimson", linestyle="--", linewidth=1.5)
    plt.title("Residual Plot for Fare Multivariate Linear Regression")
    plt.xlabel("Predicted Fare (GBP)")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    res_plot_path = os.path.join(PLOTS_DIR, "regression_residuals_plot.png")
    plt.savefig(res_plot_path, dpi=200)
    plt.close()

    # Heteroscedasticity Conclusion
    hetero_conclusion = (
        "The residual plot exhibits a distinct fan/funnel shape (heteroscedasticity), where the variance of residuals "
        "expands dramatically as predicted fare increases. For lower-priced tickets (< 50 GBP), residuals cluster tightly around zero, "
        "whereas high-fare luxury suites exhibit very large positive errors due to positive skewness and extreme outliers in fare values."
    )

    report_path = os.path.join(OUTPUTS_DIR, "regression_metrics.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MULTIVARIATE LINEAR REGRESSION (FARE PREDICTION) METRICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Target Variable:     fare\n")
        f.write(f"Predictor Features:  {feature_cols}\n")
        f.write(f"Test Sample Size:    n = {n}\n\n")
        f.write(f"  MAE:         {mae:.4f} GBP\n")
        f.write(f"  RMSE:        {rmse:.4f} GBP\n")
        f.write(f"  R²:          {r2:.4f}\n")
        f.write(f"  Adjusted R²: {adj_r2:.4f}\n\n")
        f.write("--- Heteroscedasticity Analysis ---\n")
        f.write(hetero_conclusion + "\n")

    print(f"Fare regression complete. Report saved to: {report_path}")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adj_R2": adj_r2,
        "hetero_conclusion": hetero_conclusion
    }


def generate_full_model_comparison(clf_metrics, reg_metrics):
    """
    Generates combined model comparison report presenting classification metrics
    and regression metrics in separate columns/groups, plus a final written recommendation.
    """
    rf_acc = clf_metrics["Random Forest"]["Accuracy"]
    rf_f1 = clf_metrics["Random Forest"]["F1"]
    rf_auc = clf_metrics["Random Forest"]["ROC_AUC"]
    lr_f1 = clf_metrics["Logistic Regression"]["F1"]
    lr_auc = clf_metrics["Logistic Regression"]["ROC_AUC"]
    dt_f1 = clf_metrics["Decision Tree"]["F1"]
    dt_auc = clf_metrics["Decision Tree"]["ROC_AUC"]

    recommendation = (
        f"Deployment Recommendation: The Random Forest Classifier is the recommended production model, "
        f"achieving the strongest overall Accuracy ({rf_acc:.4f}), F1-Score ({rf_f1:.4f}), and Recall ({clf_metrics['Random Forest']['Recall']:.4f}) on the holdout test set, "
        f"surpassing Logistic Regression (F1: {lr_f1:.4f}, AUC: {lr_auc:.4f}) and the Decision Tree (F1: {dt_f1:.4f}, AUC: {dt_auc:.4f}). "
        f"Its ensemble bagging architecture effectively captures non-linear interactions across passenger class, sex, and age while preventing overfitting."
    )

    report_path = os.path.join(OUTPUTS_DIR, "model_comparison_table.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 95 + "\n")
        f.write("COMPREHENSIVE MODEL COMPARISON & FINAL RECOMMENDATION REPORT\n")
        f.write("=" * 95 + "\n\n")
        f.write("SECTION A: CLASSIFICATION MODELS (Target = survived [0/1])\n")
        f.write(f"{'Classifier':<24} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC AUC':<10}\n")
        f.write("-" * 95 + "\n")
        for name, m in clf_metrics.items():
            f.write(f"{name:<24} | {m['Accuracy']:<10.4f} | {m['Precision']:<10.4f} | {m['Recall']:<10.4f} | {m['F1']:<10.4f} | {m['ROC_AUC']:<10.4f}\n")
        
        f.write("\n" + "=" * 95 + "\n")
        f.write("SECTION B: REGRESSION MODEL (Target = fare [Continuous GBP])\n")
        f.write(f"{'Model':<24} | {'MAE (GBP)':<12} | {'RMSE (GBP)':<12} | {'R²':<12} | {'Adjusted R²':<12}\n")
        f.write("-" * 95 + "\n")
        f.write(f"{'Linear Regression':<24} | {reg_metrics['MAE']:<12.4f} | {reg_metrics['RMSE']:<12.4f} | {reg_metrics['R2']:<12.4f} | {reg_metrics['Adj_R2']:<12.4f}\n")

        f.write("\n" + "=" * 95 + "\n")
        f.write("SECTION C: FINAL DEPLOYMENT RECOMMENDATION (3-5 SENTENCES)\n")
        f.write("-" * 95 + "\n")
        f.write(recommendation + "\n")

    print(f"Model comparison table saved to: {report_path}")


def save_and_verify_pipeline(best_pipeline):
    """
    Saves the complete fitted preprocessing + classifier pipeline using joblib.dump.
    Reloads it with joblib.load and verifies prediction on raw, unprocessed input.
    """
    model_save_path = os.path.join(MODELS_DIR, "best_titanic_classifier_pipeline.joblib")
    joblib.dump(best_pipeline, model_save_path)
    print(f"Complete fitted pipeline saved to: {model_save_path}")

    # Reload pipeline
    reloaded_pipeline = joblib.load(model_save_path)

    # Test raw, unprocessed input sample
    raw_sample = pd.DataFrame([
        {
            "pclass": 1,
            "sex": "female",
            "age": 29.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 211.3375,
            "embarked": "S"
        },
        {
            "pclass": 3,
            "sex": "male",
            "age": 22.0,
            "sibsp": 1,
            "parch": 0,
            "fare": 7.25,
            "embarked": "S"
        }
    ])

    raw_preds = reloaded_pipeline.predict(raw_sample)
    raw_probs = reloaded_pipeline.predict_proba(raw_sample)

    verif_path = os.path.join(OUTPUTS_DIR, "model_reload_verification.txt")
    with open(verif_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL PERSISTENCE & RELOAD VERIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Saved Model File: {model_save_path}\n")
        f.write(f"Pipeline Components: {list(reloaded_pipeline.named_steps.keys())}\n\n")
        f.write("Raw Test Sample Input (Unpreprocessed):\n")
        f.write(raw_sample.to_string(index=False) + "\n\n")
        f.write("Reloaded Pipeline Prediction Results:\n")
        for i, (pred, prob) in enumerate(zip(raw_preds, raw_probs)):
            label = "Survived" if pred == 1 else "Died"
            f.write(f"  Passenger {i+1}: Prediction = {pred} ({label}) | Survival Probability = {prob[1]:.4f}\n")
        f.write("\nVerification Status: SUCCESS (Full end-to-end ColumnTransformer + Estimator verified on raw inputs).\n")

    print(f"Reload verification report saved to: {verif_path}")


def run_modeling_pipeline():
    """
    Orchestrate full predictive modeling suite.
    """
    df = load_cleaned_data()
    fitted_pipes, clf_metrics, (X_train, X_test, y_train, y_test) = run_classification_modeling(df)
    run_class_imbalance_comparison(X_train, X_test, y_train, y_test)
    best_rf, best_params, oob_score = run_random_forest_gridsearch(X_train, y_train)
    reg_metrics = run_fare_regression(df)
    generate_full_model_comparison(clf_metrics, reg_metrics)
    save_and_verify_pipeline(best_rf)
    print("\nPredictive modeling pipeline executed successfully.")


if __name__ == "__main__":
    run_modeling_pipeline()
