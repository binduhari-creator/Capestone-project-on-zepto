"""
analysis.py
===========
Module 2: Exploratory Data Analysis, Profiling, Cleaning, and Statistical Sanity Checks
for the Titanic dataset in the Zepto Data & AI Platform Capstone.
"""

import os
import io
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure standard output can handle UTF-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, "plots")
OUTPUTS_DIR = os.path.join(CURRENT_DIR, "outputs")
CSV_PATH = os.path.join(CURRENT_DIR, "titanic.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def load_dataset():
    """
    Loads Titanic dataset using Seaborn built-in loader with local CSV offline fallback.
    Saves titanic.csv immediately upon first load.
    """
    try:
        df = sns.load_dataset("titanic")
        df.to_csv(CSV_PATH, index=False)
        print(f"Loaded dataset via sns.load_dataset and saved offline fallback to: {CSV_PATH}")
    except Exception as e:
        print(f"sns.load_dataset failed ({e}). Loading from local fallback: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)

    return df


def profile_data(df):
    """
    Profile raw dataset: df.shape, df.info(), df.describe(), and exact missing percentages.
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    describe_df = df.describe(include="all")
    missing_series = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

    missing_df = pd.DataFrame({
        "Missing_Count": missing_series,
        "Missing_Percentage": missing_pct
    })
    missing_df = missing_df[missing_df["Missing_Count"] > 0].sort_values(by="Missing_Percentage", ascending=False)

    report_path = os.path.join(OUTPUTS_DIR, "profiling_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TITANIC DATASET PROFILING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
        f.write("--- Column Info ---\n")
        f.write(info_str + "\n\n")
        f.write("--- Missing Value Percentages ---\n")
        f.write(missing_df.to_string() + "\n\n")
        f.write("--- Summary Statistics (df.describe) ---\n")
        f.write(describe_df.to_string() + "\n\n")

    print(f"Profiling summary saved to: {report_path}")
    return missing_df


def clean_eda_dataset(df):
    """
    Clean dataset for exploratory analysis following threshold rules:
    - < 5% missing: drop rows (embarked/embark_town: 2 rows, 0.22%)
    - 5% - 30% missing: impute (age: 177 missing, 19.87% -> median age)
    - > 30% missing: drop column (deck: 688 missing, 77.22% -> drop column due to severe unreliability)
    """
    cleaned_df = df.copy()

    # Drop deck column due to 77.22% missingness
    if "deck" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=["deck"])

    # Drop rows where embarked / embark_town is missing (< 5%)
    cleaned_df = cleaned_df.dropna(subset=["embarked", "embark_town"])

    # Impute age with median (5% - 30%)
    median_age = cleaned_df["age"].median()
    cleaned_df["age"] = cleaned_df["age"].fillna(median_age)

    return cleaned_df


def univariate_analysis(df):
    """
    Univariate analysis for 'age' and 'fare':
    - Histograms & Boxplots
    - IQR-based outlier counts
    - Skewness and mean/median/mode comparison for fare
    """
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("UNIVARIATE ANALYSIS & OUTLIER REPORT")
    summary_lines.append("=" * 80 + "\n")

    # 1. Plots for Age
    plt.figure(figsize=(8, 4))
    sns.histplot(df["age"], bins=30, kde=True, color="steelblue")
    plt.title("Age Distribution (Histogram & KDE)")
    plt.xlabel("Age (years)")
    plt.ylabel("Passenger Count")
    plt.tight_layout()
    age_hist_path = os.path.join(PLOTS_DIR, "univariate_age_hist.png")
    plt.savefig(age_hist_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 3))
    sns.boxplot(x=df["age"], color="lightsteelblue")
    plt.title("Age Distribution Boxplot")
    plt.xlabel("Age (years)")
    plt.tight_layout()
    age_box_path = os.path.join(PLOTS_DIR, "univariate_age_box.png")
    plt.savefig(age_box_path, dpi=200)
    plt.close()

    # 2. Plots for Fare
    plt.figure(figsize=(8, 4))
    sns.histplot(df["fare"], bins=40, kde=True, color="teal")
    plt.title("Fare Distribution (Histogram & KDE)")
    plt.xlabel("Fare (GBP)")
    plt.ylabel("Passenger Count")
    plt.tight_layout()
    fare_hist_path = os.path.join(PLOTS_DIR, "univariate_fare_hist.png")
    plt.savefig(fare_hist_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 3))
    sns.boxplot(x=df["fare"], color="mediumaquamarine")
    plt.title("Fare Distribution Boxplot")
    plt.xlabel("Fare (GBP)")
    plt.tight_layout()
    fare_box_path = os.path.join(PLOTS_DIR, "univariate_fare_box.png")
    plt.savefig(fare_box_path, dpi=200)
    plt.close()

    # Outlier Calculation via IQR
    for col in ["age", "fare"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100

        summary_lines.append(f"--- {col.upper()} IQR Outliers ---")
        summary_lines.append(f"  Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
        summary_lines.append(f"  Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
        summary_lines.append(f"  Outliers Count: {outlier_count} ({outlier_pct:.2f}% of passengers)\n")

    # Fare Central Tendency & Skewness
    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode()[0]
    fare_skew = df["fare"].skew()

    summary_lines.append("--- FARE SKEWNESS & CENTRAL TENDENCY ---")
    summary_lines.append(f"  Mean:   {fare_mean:.2f}")
    summary_lines.append(f"  Median: {fare_median:.2f}")
    summary_lines.append(f"  Mode:   {fare_mode:.2f}")
    summary_lines.append(f"  Skewness Coefficient: {fare_skew:.4f}")
    summary_lines.append("  Skewness Interpretation:")
    summary_lines.append("  Because Mean (32.10) > Median (14.45) > Mode (8.05), and skewness is strongly positive (+4.79),")
    summary_lines.append("  the fare distribution is strongly RIGHT-SKEWED (positive skew) with a long right tail caused by luxury first-class suites.\n")

    eda_output_path = os.path.join(OUTPUTS_DIR, "eda_univariate_summary.txt")
    with open(eda_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Univariate summary saved to: {eda_output_path}")
    return {
        "fare_mean": fare_mean,
        "fare_median": fare_median,
        "fare_mode": fare_mode,
        "fare_skew": fare_skew
    }


def bivariate_analysis(df):
    """
    Bivariate analysis:
    1. Boolean masking survival rates for:
       (a) sex
       (b) pclass
       (c) sex + pclass
    2. Correlation matrix for EXACTLY: survived, pclass, age, sibsp, parch, fare (excluding adult_male, alone).
    3. Heatmap visualization and identification of top 2 strongest absolute off-diagonal correlations.
    """
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("BIVARIATE ANALYSIS & CORRELATION REPORT")
    summary_lines.append("=" * 80 + "\n")

    # (a) Survival by Sex using Boolean Masking
    female_mask = df["sex"] == "female"
    male_mask = df["sex"] == "male"
    female_survival = df[female_mask]["survived"].mean()
    male_survival = df[male_mask]["survived"].mean()

    summary_lines.append("--- 1. Survival Rate by Sex (Boolean Masking) ---")
    summary_lines.append(f"  Female Survival Rate: {female_survival:.4f} ({female_survival*100:.2f}%) [n={female_mask.sum()}]")
    summary_lines.append(f"  Male Survival Rate:   {male_survival:.4f} ({male_survival*100:.2f}%) [n={male_mask.sum()}]\n")

    # (b) Survival by Pclass using Boolean Masking
    summary_lines.append("--- 2. Survival Rate by Passenger Class (Boolean Masking) ---")
    for pclass in [1, 2, 3]:
        pclass_mask = df["pclass"] == pclass
        pclass_survival = df[pclass_mask]["survived"].mean()
        summary_lines.append(f"  Class {pclass} Survival Rate: {pclass_survival:.4f} ({pclass_survival*100:.2f}%) [n={pclass_mask.sum()}]")
    summary_lines.append("")

    # (c) Survival by Sex + Pclass using Boolean Masking (& combinations)
    summary_lines.append("--- 3. Survival Rate by Sex + Pclass (Boolean Masking) ---")
    for sex in ["female", "male"]:
        for pclass in [1, 2, 3]:
            combo_mask = (df["sex"] == sex) & (df["pclass"] == pclass)
            combo_survival = df[combo_mask]["survived"].mean()
            summary_lines.append(f"  {sex.capitalize()} in Class {pclass}: Survival Rate = {combo_survival:.4f} ({combo_survival*100:.2f}%) [n={combo_mask.sum()}]")
    summary_lines.append("")

    # 4. Correlation Matrix on EXACTLY the 6 numeric columns
    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr_matrix = df[corr_cols].corr()

    # Save heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title("Correlation Heatmap (6 Specified Numeric Features)")
    plt.tight_layout()
    heatmap_path = os.path.join(PLOTS_DIR, "correlation_heatmap_6x6.png")
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    # Find two strongest off-diagonal correlations
    off_diag_pairs = []
    for i in range(len(corr_cols)):
        for j in range(i + 1, len(corr_cols)):
            col1 = corr_cols[i]
            col2 = corr_cols[j]
            r = corr_matrix.loc[col1, col2]
            off_diag_pairs.append((col1, col2, r, abs(r)))

    off_diag_sorted = sorted(off_diag_pairs, key=lambda x: x[3], reverse=True)
    top1 = off_diag_sorted[0]
    top2 = off_diag_sorted[1]

    summary_lines.append("--- 4. Correlation Matrix (6x6) ---")
    summary_lines.append(corr_matrix.to_string() + "\n")
    summary_lines.append("--- 5. Top Two Strongest Absolute Off-Diagonal Correlations ---")
    summary_lines.append(f"  1. Pair: ({top1[0]}, {top1[1]}) | r = {top1[2]:.4f} | |r| = {top1[3]:.4f}")
    summary_lines.append("     Interpretation: Strong negative correlation between pclass and fare. First-class passengers (pclass=1) paid substantially higher fares than second and third class.")
    summary_lines.append(f"  2. Pair: ({top2[0]}, {top2[1]}) | r = {top2[2]:.4f} | |r| = {top2[3]:.4f}")
    summary_lines.append("     Interpretation: Moderate positive correlation between sibsp (siblings/spouses) and parch (parents/children), reflecting passengers traveling in cohesive multi-member family units.\n")

    bivariate_path = os.path.join(OUTPUTS_DIR, "eda_bivariate_summary.txt")
    with open(bivariate_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Bivariate summary saved to: {bivariate_path}")
    return corr_matrix, top1, top2


def multivariate_analysis(df):
    """
    Multivariate analysis: Produce 4 distinct charts with written 2-4 sentence interpretations.
    """
    # 1. Survival Rate by Pclass and Sex
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="pclass", y="survived", hue="sex", palette=["coral", "skyblue"], errorbar=None)
    plt.title("Multivariate Chart 1: Survival Rate by Passenger Class and Sex")
    plt.xlabel("Passenger Class (Pclass)")
    plt.ylabel("Survival Rate")
    plt.ylim(0, 1.05)
    for p in plt.gca().patches:
        h = p.get_height()
        if h > 0:
            plt.gca().annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h / 2),
                               ha='center', va='center', color='white', fontweight='bold')
    plt.tight_layout()
    chart1_path = os.path.join(PLOTS_DIR, "multivariate_1_survival_pclass_sex.png")
    plt.savefig(chart1_path, dpi=200)
    plt.close()

    # 2. Age Distribution across Pclass by Survival Status
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="pclass", y="age", hue="survived", palette={0: "lightcoral", 1: "mediumseagreen"})
    plt.title("Multivariate Chart 2: Age Distribution across Pclass by Survival (0=Died, 1=Survived)")
    plt.xlabel("Passenger Class")
    plt.ylabel("Age (years)")
    plt.tight_layout()
    chart2_path = os.path.join(PLOTS_DIR, "multivariate_2_age_pclass_survival.png")
    plt.savefig(chart2_path, dpi=200)
    plt.close()

    # 3. Fare vs Age by Survival and Class
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="age",
        y="fare",
        hue="survived",
        style="pclass",
        palette={0: "red", 1: "green"},
        alpha=0.7,
        s=60
    )
    plt.title("Multivariate Chart 3: Fare vs Age by Survival Status and Passenger Class")
    plt.xlabel("Age (years)")
    plt.ylabel("Fare (GBP)")
    plt.yscale("log")
    plt.tight_layout()
    chart3_path = os.path.join(PLOTS_DIR, "multivariate_3_fare_age_survival_pclass.png")
    plt.savefig(chart3_path, dpi=200)
    plt.close()

    # 4. Family Size vs Survival Rate across Pclasses
    df_fam = df.copy()
    df_fam["family_size"] = df_fam["sibsp"] + df_fam["parch"] + 1

    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=df_fam,
        x="family_size",
        y="survived",
        hue="pclass",
        marker="o",
        palette="Dark2",
        errorbar=None
    )
    plt.title("Multivariate Chart 4: Survival Rate by Family Size and Passenger Class")
    plt.xlabel("Family Size (sibsp + parch + 1)")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    chart4_path = os.path.join(PLOTS_DIR, "multivariate_4_family_size_survival_pclass.png")
    plt.savefig(chart4_path, dpi=200)
    plt.close()

    interpretations = [
        "Chart 1 (Survival by Class and Sex): Females in 1st class had an exceptional 96.8% survival rate, compared to 92.1% in 2nd and 50.0% in 3rd class. Males suffered markedly lower survival across all classes (36.9% in 1st class, dropping down to only 13.5% in 3rd class). This clearly highlights the compounding protective effect of female gender and higher socioeconomic status.",
        "Chart 2 (Age by Class and Survival): Across 1st and 2nd class, surviving passengers were noticeably younger than non-survivors, reflecting the maritime evacuation priority given to children and young mothers. In 3rd class, survival was generally depressed across all age brackets with narrow separation.",
        "Chart 3 (Fare vs Age by Survival and Class): High-fare passengers clustered in 1st class experienced overwhelmingly higher survival rates regardless of age. Conversely, low-fare passengers (primarily in 3rd class) faced heavy casualty rates across the entire age spectrum, with very few surviving adults.",
        "Chart 4 (Family Size vs Survival): Small family units (sizes 2 to 4) exhibited the highest survival rates across all passenger classes, as families could assist each other during evacuation. In contrast, solo travelers (size 1) and large families (size 5+) suffered significantly lower survival rates due to isolation or difficulties keeping large groups together."
    ]

    report_path = os.path.join(OUTPUTS_DIR, "multivariate_data_story.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MULTIVARIATE DATA STORY INTERPRETATIONS\n")
        f.write("=" * 80 + "\n\n")
        for i, text in enumerate(interpretations, 1):
            f.write(f"--- Chart {i} Interpretation ---\n{text}\n\n")

    print(f"Multivariate charts and interpretations saved to: {OUTPUTS_DIR}")
    return interpretations


def zscore_sanity_check(df):
    """
    EDA-stage sanity check: Standardize age and fare using z-score z = (x - mean) / std.
    Show before and after comparison of mean and std.
    """
    df_check = df.copy()

    # Before stats
    age_mean_pre = df_check["age"].mean()
    age_std_pre = df_check["age"].std()
    fare_mean_pre = df_check["fare"].mean()
    fare_std_pre = df_check["fare"].std()

    # Apply z-score standardization
    df_check["age_z"] = (df_check["age"] - age_mean_pre) / age_std_pre
    df_check["fare_z"] = (df_check["fare"] - fare_mean_pre) / fare_std_pre

    # After stats
    age_mean_post = df_check["age_z"].mean()
    age_std_post = df_check["age_z"].std()
    fare_mean_post = df_check["fare_z"].mean()
    fare_std_post = df_check["fare_z"].std()

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(df_check["age"], kde=True, ax=axes[0, 0], color="royalblue")
    axes[0, 0].set_title(f"Age Raw (Mean={age_mean_pre:.2f}, Std={age_std_pre:.2f})")

    sns.histplot(df_check["age_z"], kde=True, ax=axes[0, 1], color="darkorange")
    axes[0, 1].set_title(f"Age Z-Score (Mean={age_mean_post:.4f}≈0, Std={age_std_post:.4f}≈1)")

    sns.histplot(df_check["fare"], kde=True, ax=axes[1, 0], color="seagreen")
    axes[1, 0].set_title(f"Fare Raw (Mean={fare_mean_pre:.2f}, Std={fare_std_pre:.2f})")

    sns.histplot(df_check["fare_z"], kde=True, ax=axes[1, 1], color="crimson")
    axes[1, 1].set_title(f"Fare Z-Score (Mean={fare_mean_post:.4f}≈0, Std={fare_std_post:.4f}≈1)")

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "zscore_standardization_check.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    report_path = os.path.join(OUTPUTS_DIR, "zscore_sanity_check.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Z-SCORE STANDARDIZATION SANITY CHECK (EDA STAGE ONLY)\n")
        f.write("=" * 80 + "\n\n")
        f.write("NOTE: This transformation is strictly an exploratory sanity check on the full dataset\n")
        f.write("to verify standard normal scaling behavior. The predictive modeling pipeline performs\n")
        f.write("its own train-only preprocessing separately to avoid data leakage.\n\n")
        f.write("--- BEFORE STANDARDIZATION ---\n")
        f.write(f"Age:  Mean = {age_mean_pre:.4f}, Std = {age_std_pre:.4f}\n")
        f.write(f"Fare: Mean = {fare_mean_pre:.4f}, Std = {fare_std_pre:.4f}\n\n")
        f.write("--- AFTER STANDARDIZATION (z = (x - mean) / std) ---\n")
        f.write(f"Age Z-score:  Mean = {age_mean_post:.6f} (≈ 0), Std = {age_std_post:.6f} (≈ 1)\n")
        f.write(f"Fare Z-score: Mean = {fare_mean_post:.6f} (≈ 0), Std = {fare_std_post:.6f} (≈ 1)\n")

    print(f"Z-score sanity check report saved to: {report_path}")


def run_eda_pipeline():
    """
    Orchestrate full EDA pipeline.
    """
    df_raw = load_dataset()
    missing_df = profile_data(df_raw)
    df_clean = clean_eda_dataset(df_raw)
    univariate_analysis(df_clean)
    bivariate_analysis(df_clean)
    multivariate_analysis(df_clean)
    zscore_sanity_check(df_clean)
    print("\nEDA Pipeline completed successfully.")
    return df_clean


if __name__ == "__main__":
    run_eda_pipeline()
