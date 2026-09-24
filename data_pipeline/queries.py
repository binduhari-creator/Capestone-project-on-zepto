"""
queries.py
==========
SQL query execution and Pandas comparison module.
Executes required SQL queries (SELECT, WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, IN, JOIN),
reads query results using pd.read_sql, and compares SQL JOIN vs pandas.merge().
"""

import os
import sys
import sqlite3
import pandas as pd
from pandas.testing import assert_frame_equal

# Ensure UTF-8 output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "zepto_catalog.db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


QUERIES = {
    "Query 1 - SELECT_WHERE": {
        "description": "Demonstrates SELECT and WHERE: Filter 5-star in-stock books.",
        "sql": """
        SELECT book_id, title, price_gbp, price_inr, rating, in_stock
        FROM books
        WHERE rating = 5 AND in_stock = 1
        ORDER BY book_id ASC;
        """
    },
    "Query 2 - ORDER_BY_LIMIT": {
        "description": "Demonstrates ORDER BY and LIMIT: Top 10 most expensive books by price_inr.",
        "sql": """
        SELECT book_id, title, price_gbp, price_inr, rating
        FROM books
        ORDER BY price_inr DESC
        LIMIT 10;
        """
    },
    "Query 3 - DISTINCT": {
        "description": "Demonstrates DISTINCT: Unique rating values among all books in stock.",
        "sql": """
        SELECT DISTINCT rating
        FROM books
        WHERE in_stock = 1
        ORDER BY rating ASC;
        """
    },
    "Query 4 - BETWEEN": {
        "description": "Demonstrates BETWEEN: Books with price_gbp between 20.00 and 40.00.",
        "sql": """
        SELECT book_id, title, price_gbp, price_inr, rating
        FROM books
        WHERE price_gbp BETWEEN 20.00 AND 40.00
        ORDER BY price_gbp ASC;
        """
    },
    "Query 5 - IN": {
        "description": "Demonstrates IN clause: Books with top-tier ratings (4 or 5 stars).",
        "sql": """
        SELECT book_id, title, rating, price_gbp, price_inr
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, price_inr DESC
        LIMIT 15;
        """
    },
    "Query 6 - JOIN": {
        "description": "Demonstrates JOIN: Books joined with categories table on category_id.",
        "sql": """
        SELECT 
            b.book_id,
            b.title,
            c.category_name,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.book_id ASC;
        """
    }
}


def execute_and_save_queries(db_path=DEFAULT_DB_PATH, output_dir=OUTPUT_DIR):
    """
    Execute all SQL queries, display summaries, and write results to output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    output_report_path = os.path.join(output_dir, "sql_queries_output.txt")
    results = {}

    with open(output_report_path, "w", encoding="utf-8") as out_file:
        out_file.write("=" * 80 + "\n")
        out_file.write("ZEPTO DATA PIPELINE - SQL QUERY DEMONSTRATION REPORT\n")
        out_file.write("=" * 80 + "\n\n")

        for key, q_data in QUERIES.items():
            name = key
            desc = q_data["description"]
            sql = q_data["sql"].strip()

            # Execute query into pandas DataFrame using pd.read_sql
            df = pd.read_sql(sql, conn)
            results[key] = df

            # Write formatted output to file
            out_file.write(f"--- {name} ---\n")
            out_file.write(f"Description: {desc}\n")
            out_file.write("SQL Statement:\n")
            out_file.write(f"{sql}\n\n")
            out_file.write(f"Result Rows: {len(df)}\n")
            out_file.write("Output Table:\n")
            out_file.write(df.to_string(index=False))
            out_file.write("\n\n" + "-" * 80 + "\n\n")

    conn.close()
    print(f"All {len(QUERIES)} SQL queries executed and saved to: {output_report_path}")
    return results


def run_pandas_sql_demonstration(db_path=DEFAULT_DB_PATH):
    """
    Demonstrates pd.read_sql on at least two queries and prints verification.
    """
    conn = sqlite3.connect(db_path)
    
    # Read Query 1 via pd.read_sql
    df_q1 = pd.read_sql(QUERIES["Query 1 - SELECT_WHERE"]["sql"], conn)
    # Read Query 2 via pd.read_sql
    df_q2 = pd.read_sql(QUERIES["Query 2 - ORDER_BY_LIMIT"]["sql"], conn)

    conn.close()

    print("\n--- pd.read_sql() Demonstration ---")
    print(f"Query 1 (5-star in-stock books) read via pd.read_sql: {len(df_q1)} rows returned.")
    print(f"Query 2 (Top 10 expensive books) read via pd.read_sql: {len(df_q2)} rows returned.")

    return df_q1, df_q2


def compare_sql_join_vs_pandas_merge(db_path=DEFAULT_DB_PATH, output_dir=OUTPUT_DIR):
    """
    Compares SQL JOIN against in-memory pandas.merge():
    1. Produce result using SQL JOIN.
    2. Separately reproduce result using pandas.merge() on books and categories DataFrames.
    3. Compare the outputs and verify equivalence.
    """
    os.makedirs(output_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)

    # 1. SQL JOIN result
    sql_join = QUERIES["Query 6 - JOIN"]["sql"]
    sql_join_df = pd.read_sql(sql_join, conn)

    # 2. In-memory pandas.merge
    books_df = pd.read_sql("SELECT book_id, title, price_gbp, price_inr, rating, in_stock, category_id FROM books;", conn)
    categories_df = pd.read_sql("SELECT category_id, category_name FROM categories;", conn)
    conn.close()

    # Merge on category_id
    pandas_merged_df = pd.merge(books_df, categories_df, on="category_id", how="inner")
    
    # Reorder and project identical columns as SQL query
    columns_order = ["book_id", "title", "category_name", "price_gbp", "price_inr", "rating", "in_stock"]
    pandas_merged_df = pandas_merged_df[columns_order].sort_values(by="book_id").reset_index(drop=True)
    sql_join_df = sql_join_df.sort_values(by="book_id").reset_index(drop=True)

    # 3. Compare outputs
    # Validate shapes and null counts
    shape_match = sql_join_df.shape == pandas_merged_df.shape
    
    # Assert exact frame equality
    assert_frame_equal(sql_join_df, pandas_merged_df, check_dtype=True)
    equivalent = True

    comparison_report_path = os.path.join(output_dir, "join_comparison.txt")
    with open(comparison_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SQL JOIN vs PANDAS MERGE COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"SQL JOIN Row Count: {len(sql_join_df)}, Column Count: {sql_join_df.shape[1]}\n")
        f.write(f"Pandas Merge Row Count: {len(pandas_merged_df)}, Column Count: {pandas_merged_df.shape[1]}\n")
        f.write(f"Shape Match: {shape_match}\n")
        f.write(f"Exact Data Equivalence (assert_frame_equal): {equivalent}\n\n")
        f.write("--- Sample SQL JOIN Output (Top 5 rows) ---\n")
        f.write(sql_join_df.head(5).to_string(index=False) + "\n\n")
        f.write("--- Sample Pandas Merge Output (Top 5 rows) ---\n")
        f.write(pandas_merged_df.head(5).to_string(index=False) + "\n\n")

    print("\n--- SQL JOIN vs pandas.merge() Comparison ---")
    print(f"SQL JOIN Shape: {sql_join_df.shape}")
    print(f"Pandas Merge Shape: {pandas_merged_df.shape}")
    print(f"Equivalence Confirmed: {equivalent} (saved to {comparison_report_path})")

    return equivalent, sql_join_df, pandas_merged_df


if __name__ == "__main__":
    execute_and_save_queries()
    run_pandas_sql_demonstration()
    compare_sql_join_vs_pandas_merge()
