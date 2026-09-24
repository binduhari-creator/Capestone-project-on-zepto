"""
database.py
===========
Database initialization and data loading module for SQLite.
Creates normalized relational tables (`categories` and `books`)
with Primary Key / Foreign Key constraints and loads cleaned catalog data.
"""

import os
import sqlite3
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "zepto_catalog.db")


def get_connection(db_path=DEFAULT_DB_PATH):
    """
    Establish a connection to the SQLite database with foreign keys enabled.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path=DEFAULT_DB_PATH, drop_existing=True):
    """
    Create normalized schema:
    1. categories (category_id PK, category_name UNIQUE)
    2. books (book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK)
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    if drop_existing:
        cursor.execute("DROP TABLE IF EXISTS books;")
        cursor.execute("DROP TABLE IF EXISTS categories;")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
            ON UPDATE CASCADE
            ON DELETE RESTRICT
    );
    """)

    conn.commit()
    conn.close()
    print(f"Database schema initialized successfully at: {db_path}")


def populate_database(df, db_path=DEFAULT_DB_PATH):
    """
    Insert cleaned DataFrame records into normalized SQLite tables:
    1. Populate categories table with unique categories
    2. Retrieve category_id mapping
    3. Insert books with category_id foreign key
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 1. Insert unique categories
    unique_categories = sorted(df["category"].dropna().unique().tolist())
    for cat in unique_categories:
        cursor.execute(
            "INSERT OR IGNORE INTO categories (category_name) VALUES (?);",
            (cat,)
        )
    conn.commit()

    # 2. Retrieve category_id mapping
    cursor.execute("SELECT category_id, category_name FROM categories;")
    cat_rows = cursor.fetchall()
    cat_to_id = {row[1]: row[0] for row in cat_rows}

    # 3. Prepare book records
    book_records = []
    for _, row in df.iterrows():
        cat_id = cat_to_id[row["category"]]
        book_records.append((
            str(row["title"]),
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(row["in_stock"]),
            int(cat_id)
        ))

    # 4. Insert books
    cursor.executemany("""
    INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (?, ?, ?, ?, ?, ?);
    """, book_records)

    conn.commit()
    conn.close()
    print(f"Loaded {len(unique_categories)} categories and {len(book_records)} books into {db_path}")


def verify_database_integrity(db_path=DEFAULT_DB_PATH):
    """
    Verify row counts, table definitions, and foreign key integrity.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Check foreign key check
    fk_check = cursor.execute("PRAGMA foreign_key_check;").fetchall()
    if fk_check:
        raise ValueError(f"Foreign key integrity violations detected: {fk_check}")

    cursor.execute("SELECT COUNT(*) FROM categories;")
    num_categories = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM books;")
    num_books = cursor.fetchone()[0]

    cursor.execute("PRAGMA table_info(categories);")
    cat_cols = cursor.fetchall()

    cursor.execute("PRAGMA table_info(books);")
    book_cols = cursor.fetchall()

    conn.close()

    summary = {
        "num_categories": num_categories,
        "num_books": num_books,
        "categories_schema": cat_cols,
        "books_schema": book_cols,
        "fk_status": "Valid (0 violations)"
    }
    return summary


if __name__ == "__main__":
    from scraper import clean_and_transform_data
    import os

    cleaned_csv = os.path.join(os.path.dirname(__file__), "data", "cleaned_books.csv")
    if os.path.exists(cleaned_csv):
        df = pd.read_csv(cleaned_csv)
    else:
        from scraper import run_scraper_pipeline
        df = run_scraper_pipeline()

    init_db()
    populate_database(df)
    summary = verify_database_integrity()
    print("\n--- Database Verification Summary ---")
    print(f"Categories Count: {summary['num_categories']}")
    print(f"Books Count: {summary['num_books']}")
    print(f"FK Integrity: {summary['fk_status']}")
