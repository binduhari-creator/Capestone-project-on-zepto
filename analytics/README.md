# Module 2: Analytics & Machine Learning Pipeline

## 1. Module Purpose
Module 2 delivers an end-to-end analytics and predictive modeling workflow for the **Zepto Data & AI Platform Capstone**. Built upon the classic Titanic passenger dataset, this module profiles and cleans data, conducts statistical and visual exploratory data analysis (univariate, bivariate, multivariate), executes an exploratory z-score sanity check, trains and evaluates classification models on a leak-free stratified pipeline, analyzes class imbalance techniques (class weighting vs. SMOTE), tunes a Random Forest model using GridSearchCV with Out-of-Bag (OOB) evaluation, performs a multivariate linear regression side-task to predict passenger fares, and persists the fitted end-to-end model pipeline with `joblib`.

---

## 2. Dataset Ingestion & Reproducible Fallback
- **Source**: `seaborn.load_dataset('titanic')`
- **Offline Fallback**: Upon initial load, the dataset is saved to `analytics/titanic.csv`. All subsequent runs load directly from this offline CSV fallback to ensure complete offline reproducibility without relying on internet connectivity.
- **Dataset Dimensions**: 891 rows, 15 columns.

---

## 3. Data Profiling & Missing-Value Strategy

### Exact Missing-Value Percentages
| Column Name | Missing Count | Missing Percentage | Action & Justification |
| :--- | :---: | :---: | :--- |
| `deck` | 688 | **77.22%** | **Drop Column**: Exceeds the 30% threshold. With >77% missing, imputation is highly speculative and would introduce significant noise. |
| `age` | 177 | **19.87%** | **Median Imputation**: Falls within the 5%–30% threshold. Imputed using median age (28.0 years) to preserve distribution shape against extreme values. |
| `embarked` / `embark_town` | 2 | **0.22%** | **Drop Rows (< 5%) / Mode Impute**: In exploratory cleaning, the 2 rows (< 5%) are dropped. In the modeling pipeline, missing entries are handled via mode imputation (`most_frequent`). |

*Full profiling details are saved in `analytics/outputs/profiling_summary.txt`.*

---

## 4. Exploratory Data Analysis (EDA)

### Univariate Analysis (`age` and `fare`)
- **Outlier Detection via IQR Rule** ($[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$):
  - **Age**: $Q_1 = 22.00$, $Q_3 = 35.00$, $IQR = 13.00$. Lower Bound: $2.50$, Upper Bound: $54.50$. Outliers: **66 passengers (7.42%)**.
  - **Fare**: $Q_1 = 7.90$, $Q_3 = 31.00$, $IQR = 23.10$. Lower Bound: $-26.75$, Upper Bound: $65.65$. Outliers: **114 passengers (12.82%)**.
- **Fare Central Tendency & Skewness**:
  - **Mean**: $32.10$ GBP
  - **Median**: $14.45$ GBP
  - **Mode**: $8.05$ GBP
  - **Skewness**: $+4.79$
  - **Skewness Conclusion**: Because $\text{Mean } (32.10) > \text{Median } (14.45) > \text{Mode } (8.05)$ and skewness is strongly positive ($+4.79$), the fare distribution is **heavily right-skewed** with a long right tail driven by first-class luxury suites.
- **Saved Plots**: `plots/univariate_age_hist.png`, `plots/univariate_age_box.png`, `plots/univariate_fare_hist.png`, `plots/univariate_fare_box.png`.

---

### Bivariate Analysis & 6×6 Correlation Matrix
- **Survival Rate Breakdowns (Boolean Masking)**:
  - **By Sex**:
    - Female: **74.04%** ($n=312$)
    - Male: **18.96%** ($n=577$)
  - **By Passenger Class**:
    - Class 1: **62.62%** ($n=214$)
    - Class 2: **47.28%** ($n=184$)
    - Class 3: **24.24%** ($n=491$)
  - **By Sex + Pclass Interaction**:
    - Female Class 1: **96.81%** | Female Class 2: **92.11%** | Female Class 3: **50.00%**
    - Male Class 1: **36.89%** | Male Class 2: **15.74%** | Male Class 3: **13.54%**
- **6×6 Numeric Correlation Heatmap**:
  - Columns evaluated: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare` *(excluding derived flags `adult_male` and `alone`)*.
  - **Top 2 Strongest Absolute Off-Diagonal Correlations**:
    1. **`pclass` and `fare`** ($r = -0.5481$, $|r| = 0.5481$): Strong negative correlation reflecting that 1st class passengers paid substantially higher fares.
    2. **`sibsp` and `parch`** ($r = +0.4145$, $|r| = 0.4145$): Moderate positive correlation indicating family groups traveling together.
- **Saved Plot**: `plots/correlation_heatmap_6x6.png`.

---

### Multivariate Data Story (4 Distinct Visualizations)
1. **Chart 1 — Survival by Class and Sex (`plots/multivariate_1_survival_pclass_sex.png`)**:
   - *Interpretation*: Demonstrates the profound intersection of gender and class. First-class women survived at 96.8% and second-class women at 92.1%, while third-class men survived at only 13.5%. The "women and children first" maritime protocol was most strictly enacted for upper-class cabins.
2. **Chart 2 — Age Distribution by Class and Survival (`plots/multivariate_2_age_pclass_survival.png`)**:
   - *Interpretation*: In 1st and 2nd class, survivors were notably younger than non-survivors, showing prioritized lifeboat boarding for children. In 3rd class, survival rates were uniformly depressed across all age brackets.
3. **Chart 3 — Fare vs. Age by Survival and Class (`plots/multivariate_3_fare_age_survival_pclass.png`)**:
   - *Interpretation*: High ticket fares in 1st class correlated with high survival regardless of passenger age. Low-fare passengers in 3rd class suffered severe mortality across all age groups.
4. **Chart 4 — Family Size vs. Survival across Classes (`plots/multivariate_4_family_size_survival_pclass.png`)**:
   - *Interpretation*: Small family units (sizes 2–4) experienced peak survival rates across all classes because family members could assist each other. Solo travelers and large families (5+) had much lower survival rates due to isolation or difficulty coordinating evacuation.

---

### Z-Score Standardization Sanity Check
- Purely an exploratory check on the full dataset before modeling:
  - **Age Raw**: Mean = $29.35$, Std = $13.02$ $\longrightarrow$ **Age Z-Score**: Mean = $0.0000$, Std = $1.0000$
  - **Fare Raw**: Mean = $32.10$, Std = $49.70$ $\longrightarrow$ **Fare Z-Score**: Mean = $0.0000$, Std = $1.0000$
- **Saved Plot**: `plots/zscore_standardization_check.png`.

---

## 5. Predictive Modeling & Evaluation

### Stratified Train/Test Split & Preprocessing Pipeline
- **Target**: `survived` ($0=$ Died, $1=$ Survived).
- **Split Ratio**: 80% Train ($n=712$), 20% Test ($n=179$) using `stratify=y` to preserve the ground truth class balance (61.66% Died, 38.34% Survived).
- **Leak-Free Preprocessing**: Built with `ColumnTransformer` + `Pipeline`, fit **strictly on training data**:
  - Numeric (`age`, `fare`, `sibsp`, `parch`, `pclass`): `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`.
  - Categorical (`sex`, `embarked`): `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`.

---

### Classification Model Results (Holdout Test Set)

| Classifier | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8437 |
| **Decision Tree (depth=4)** | 0.7877 | 0.8605 | 0.5362 | 0.6607 | 0.8152 |
| **Random Forest (100 trees)** | **0.8156** | 0.8000 | **0.6957** | **0.7442** | 0.8271 |

- **Saved Plots**: `plots/decision_tree_plot.png` (plot_tree with labeled features/classes), `plots/confusion_matrices.png`, `plots/classification_roc_curves.png`.

---

### Class Imbalance Comparison

| Strategy | Precision | Recall | F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (No Handling)** | 0.7931 | 0.6667 | 0.7244 | 0.8045 |
| **`class_weight='balanced'`** | 0.7297 | 0.7826 | 0.7552 | 0.8045 |
| **SMOTE (Train-Fold Only)** | 0.7397 | 0.7826 | **0.7606** | **0.8101** |

- **Trade-off Analysis**: The baseline model favors majority class precision at the cost of lower recall ($0.6667$). Applying class weights or SMOTE shifts the decision boundary to boost minority recall to $0.7826$. SMOTE applied strictly to the training fold achieved the highest balanced F1-score ($0.7606$) without data leakage.

---

### Random Forest Hyperparameter Tuning (GridSearchCV)
- **Search Grid**:
  - `n_estimators`: `[50, 100, 200]`
  - `max_depth`: `[3, 5, 8, None]`
  - `max_features`: `['sqrt', 'log2', 0.5]`
- **Best Parameters**: `{'max_depth': 3, 'max_features': 0.5, 'n_estimators': 50}`
- **Best 5-Fold Cross-Validation F1**: **0.7580**
- **Out-of-Bag (OOB) Accuracy (`oob_score=True`)**: **0.8174**

---

### Multivariate Fare Regression Side-Task
- **Model**: Linear Regression predicting continuous ticket `fare` from passenger attributes.
- **Evaluation Metrics (Holdout Test Set $n=179$)**:
  - **MAE**: $20.90$ GBP
  - **RMSE**: $30.53$ GBP
  - **$R^2$**: $0.3975$
  - **Adjusted $R^2$**: $0.3729$
- **Residual Plot & Heteroscedasticity**:
  - **Saved Plot**: `plots/regression_residuals_plot.png`.
  - **Conclusion**: The residual plot exhibits clear **heteroscedasticity** (fanning/funnel shape). Residual spread expands drastically for higher predicted fares due to extreme positive skewness and outliers among luxury 1st-class tickets.

---

## 6. Comprehensive Model Comparison & Recommendation

```
===============================================================================================
COMPREHENSIVE MODEL COMPARISON REPORT
===============================================================================================

SECTION A: CLASSIFICATION MODELS (Target = survived [0/1])
Classifier               | Accuracy   | Precision  | Recall     | F1-Score   | ROC AUC   
-----------------------------------------------------------------------------------------------
Logistic Regression      | 0.8045     | 0.7931     | 0.6667     | 0.7244     | 0.8437    
Decision Tree            | 0.7877     | 0.8605     | 0.5362     | 0.6607     | 0.8152    
Random Forest            | 0.8156     | 0.8000     | 0.6957     | 0.7442     | 0.8271    

===============================================================================================
SECTION B: REGRESSION MODEL (Target = fare [Continuous GBP])
Model                    | MAE (GBP)    | RMSE (GBP)   | R²           | Adjusted R² 
-----------------------------------------------------------------------------------------------
Linear Regression        | 20.8977      | 30.5328      | 0.3975       | 0.3729      
```

### Production Deployment Recommendation
> **Recommendation**: The Random Forest Classifier is the recommended model for production deployment. It achieves the highest overall Accuracy (0.8156), F1-Score (0.7442), and minority class Recall (0.6957) on the holdout test set, outperforming both Logistic Regression (F1: 0.7244) and the Decision Tree (F1: 0.6607). Its bagging ensemble mechanism effectively captures complex non-linear interactions across passenger class, sex, and age while remaining robust against single-tree overfitting.

---

## 7. Model Persistence & Reload Verification
The full end-to-end classification pipeline (ColumnTransformer preprocessor + fitted classifier) is saved to `analytics/models/best_titanic_classifier_pipeline.joblib`.

Reload verification was conducted with raw, unprocessed passenger records:
- **Female 1st Class Sample**: Prediction = `1` (**Survived**), Survival Probability = `0.9507`.
- **Male 3rd Class Sample**: Prediction = `0` (**Died**), Survival Probability = `0.1134`.
- Verified in `analytics/outputs/model_reload_verification.txt`.

---

## 8. Installation & Execution Instructions

### Install Dependencies
```bash
pip install -r analytics/requirements.txt
```

### Run Entire Analytics Module
```bash
python analytics/main.py
```

### Run Components Individually
- **Run EDA & Visualizations**:
  ```bash
  python analytics/analysis.py
  ```
- **Run Machine Learning & Modeling**:
  ```bash
  python analytics/modeling.py
  ```
