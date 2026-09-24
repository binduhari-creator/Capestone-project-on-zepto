"""
main.py
=======
Main orchestrator for the Zepto Data & AI Platform Data Pipeline (Module 1).
Executes the full end-to-end pipeline:
1. Web Scraping & Data Extraction
2. Data Cleaning & Fixed Currency Conversion
3. Normalized SQLite Database Loading & Validation
4. SQL Query Execution & Reporting
5. Pandas SQL Reading and SQL JOIN vs. pandas.merge Equivalence Verification
"""

import os
import sys
import time

# Ensure UTF-8 output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from scraper import run_scraper_pipeline
from database import init_db, populate_database, verify_database_integrity, DEFAULT_DB_PATH
from queries import execute_and_save_queries, run_pandas_sql_demonstration, compare_sql_join_vs_pandas_merge


def run_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("ZEPTO DATA & AI PLATFORM - MODULE 1: DATA ENGINEERING PIPELINE")
    print("=" * 80)

    # 1. Scraping and Cleaning
    print("\n[STEP 1/4] Running Web Scraper and Data Cleaning...")
    cleaned_df = run_scraper_pipeline()
    total_books = len(cleaned_df)
    total_categories = cleaned_df["category"].nunique()
    print(f"-> Scraped and cleaned {total_books} books across {total_categories} categories.")

    # 2. Database Initialization and Loading
    print("\n[STEP 2/4] Initializing Database and Loading Data...")
    init_db(DEFAULT_DB_PATH)
    populate_database(cleaned_df, DEFAULT_DB_PATH)
    db_summary = verify_database_integrity(DEFAULT_DB_PATH)
    print(f"-> Database loaded: {db_summary['num_categories']} categories, {db_summary['num_books']} books.")

    # 3. SQL Query Execution
    print("\n[STEP 3/4] Executing and Saving Required SQL Queries...")
    query_results = execute_and_save_queries(DEFAULT_DB_PATH)
    print(f"-> {len(query_results)} SQL queries executed and results saved to output/.")

    # 4. Pandas Demonstrations and JOIN vs MERGE Comparison
    print("\n[STEP 4/4] Running Pandas Demonstrations & Comparison...")
    run_pandas_sql_demonstration(DEFAULT_DB_PATH)
    is_equivalent, _, _ = compare_sql_join_vs_pandas_merge(DEFAULT_DB_PATH)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY & VERIFICATION")
    print("=" * 80)
    print(f"Execution Time: {elapsed:.2f} seconds")
    print(f"Total Books Scraped: {total_books} (Requirement >= 60: {'PASS' if total_books >= 60 else 'FAIL'})")
    print(f"Total Categories: {total_categories} (Requirement >= 3: {'PASS' if total_categories >= 3 else 'FAIL'})")
    print(f"Database Integrity: {db_summary['fk_status']}")
    print(f"SQL Queries Verified: {len(query_results)} queries (Requirement >= 5: {'PASS' if len(query_results) >= 5 else 'FAIL'})")
    print(f"SQL JOIN vs pandas.merge Equivalence: {'PASS' if is_equivalent else 'FAIL'}")
    print("=" * 80)
    print("Module 1 Data Pipeline completed successfully!\n")


if __name__ == "__main__":
    run_pipeline()
