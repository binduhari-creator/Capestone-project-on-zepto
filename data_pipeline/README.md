# Module 1: Data Engineering Pipeline

## 1. Module Purpose
The purpose of Module 1 is to build an automated, end-to-end data pipeline for the **Zepto Data & AI Platform Capstone**. The pipeline autonomously scrapes e-commerce book catalog data, cleans and standardizes numeric/categorical fields, computes localized currency values using a fixed exchange rate, loads the structured data into a normalized SQLite database with relational constraints, and executes/verifies analytical SQL queries alongside Pandas DataFrame operations.

---

## 2. Technologies Used
- **Python 3.10+**: Core programming language.
- **Requests**: HTTP requests for fetching HTML web pages.
- **BeautifulSoup4 (bs4)**: HTML parsing and DOM traversal.
- **Pandas**: In-memory data manipulation, relational merging, and SQL data frame ingestion.
- **SQLite3**: Lightweight normalized relational database engine.

---

## 3. Data Source & Scraping Scope
- **Source URL**: [Books to Scrape](https://books.toscrape.com/)
- **Scraping Scope**:
  - Minimum Requirement: 60 books across at least 3 categories.
  - Actual Scraped Scope: **144 books** across **4 categories**:
    1. **Travel** (11 books)
    2. **Mystery** (32 books across 2 pages)
    3. **Historical Fiction** (26 books across 2 pages)
    4. **Sequential Art** (75 books across 4 pages)
- **Extracted Fields per Book**:
  - `title`: Book title string.
  - `price_raw`: Raw price text in GBP (e.g., `£51.77`).
  - `rating_raw`: Star rating word from DOM class attribute (`One`, `Two`, `Three`, `Four`, `Five`).
  - `availability_raw`: Availability text status (e.g., `In stock`).
  - `category`: Category name string (e.g., `Travel`, `Mystery`, etc.).

---

## 4. Cleaning & Transformation Decisions

1. **`price_gbp` (Float)**:
   - Extracted by stripping currency symbols (`£`) and extraneous whitespace using regex `[^0-9.]`.
   - Converted to `float`.
   - Missing/failed parsing fallback: Handled using median price imputation (or row drop if unrecoverable).

2. **Fixed Currency Conversion (`price_inr`)**:
   - Fixed project-defined conversion: **`1 GBP = 105.50 INR`** (No external APIs or dynamic rates).
   - Computed as: `price_inr = round(price_gbp * 105.50, 2)`.

3. **`rating` (Integer)**:
   - Text ratings mapped directly to integer values:
     - `"One"` $\rightarrow$ `1`
     - `"Two"` $\rightarrow$ `2`
     - `"Three"` $\rightarrow$ `3`
     - `"Four"` $\rightarrow$ `4`
     - `"Five"` $\rightarrow$ `5`
   - Unrecognized ratings fallback to dataset median rating.

4. **`in_stock` (Integer / Boolean)**:
   - Standardized to `1` (True) if `"in stock"` is present in the text, otherwise `0` (False).

5. **Text Hygiene**:
   - Stripped leading/trailing whitespace from `title` and `category`.
   - Enforced row validity checks (dropping any rows missing essential title/category data).

---

## 5. Database Schema & Architecture

The SQLite database (`zepto_catalog.db`) uses a 3NF normalized schema with active foreign key enforcement (`PRAGMA foreign_keys = ON;`).

```
+------------------------------------+          +--------------------------------------------+
|             categories             |          |                   books                    |
+------------------------------------+          +--------------------------------------------+
| category_id   INTEGER PRIMARY KEY  |<---------| category_id  INTEGER REFERENCES categories |
| category_name TEXT UNIQUE NOT NULL |          | book_id      INTEGER PRIMARY KEY           |
+------------------------------------+          | title        TEXT NOT NULL                 |
                                                | price_gbp    REAL NOT NULL                 |
                                                | price_inr    REAL NOT NULL                 |
                                                | rating       INTEGER NOT NULL              |
                                                | in_stock     INTEGER NOT NULL              |
                                                +--------------------------------------------+
```

---

## 6. SQL Query Demonstrations

All queries are executed against `zepto_catalog.db` and output reports are saved in `data_pipeline/output/sql_queries_output.txt`.

1. **Query 1 — SELECT & WHERE**:
   - Filters books with 5-star ratings that are currently in stock.
   - *SQL*: `SELECT book_id, title, price_gbp, price_inr, rating, in_stock FROM books WHERE rating = 5 AND in_stock = 1 ORDER BY book_id ASC;`
2. **Query 2 — ORDER BY & LIMIT**:
   - Identifies the top 10 most expensive books ranked by `price_inr DESC`.
   - *SQL*: `SELECT book_id, title, price_gbp, price_inr, rating FROM books ORDER BY price_inr DESC LIMIT 10;`
3. **Query 3 — DISTINCT**:
   - Extracts all unique rating values available across in-stock inventory.
   - *SQL*: `SELECT DISTINCT rating FROM books WHERE in_stock = 1 ORDER BY rating ASC;`
4. **Query 4 — BETWEEN**:
   - Filters books with `price_gbp` within the mid-tier price range `[20.00, 40.00]`.
   - *SQL*: `SELECT book_id, title, price_gbp, price_inr, rating FROM books WHERE price_gbp BETWEEN 20.00 AND 40.00 ORDER BY price_gbp ASC;`
5. **Query 5 — IN**:
   - Filters books with top ratings (4 or 5 stars).
   - *SQL*: `SELECT book_id, title, rating, price_gbp, price_inr FROM books WHERE rating IN (4, 5) ORDER BY rating DESC, price_inr DESC LIMIT 15;`
6. **Query 6 — Relational JOIN**:
   - Performs an `INNER JOIN` between `books` and `categories` on `category_id` to project relational attributes.
   - *SQL*: `SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating, b.in_stock FROM books b JOIN categories c ON b.category_id = c.category_id ORDER BY b.book_id ASC;`

---

## 7. Pandas Integration & SQL JOIN vs. `pandas.merge()` Comparison

1. **`pd.read_sql` Execution**:
   - Demonstrated on Query 1 (5-star books) and Query 2 (top expensive books), directly loading SQL query outputs into Pandas DataFrames.
2. **Relational Equivalence Verification**:
   - Result from SQL `JOIN` was compared against an in-memory `pd.merge(books_df, categories_df, on="category_id")`.
   - Verification was performed using `pandas.testing.assert_frame_equal`.
   - **Result**: Exactly equivalent in dimensions (144 rows, 7 columns), types, and values. Comparison details are logged to `data_pipeline/output/join_comparison.txt`.

---

## 8. Directory Structure

```
data_pipeline/
├── data/
│   ├── raw_scraped_books.csv     # Raw scraped records
│   └── cleaned_books.csv         # Cleaned and INR-converted dataset
├── output/
│   ├── sql_queries_output.txt    # Saved SQL queries and table outputs
│   └── join_comparison.txt      # SQL JOIN vs pandas.merge comparison report
├── database.py                   # Schema creation & data insertion logic
├── main.py                       # End-to-end pipeline orchestrator
├── queries.py                    # SQL query suite & Pandas verification
├── requirements.txt              # Python package dependencies
├── scraper.py                    # Web scraper and cleaning module
├── zepto_catalog.db              # SQLite normalized database
└── README.md                     # Module documentation
```

---

## 9. Installation & Running Instructions

### Install Dependencies
```bash
pip install -r data_pipeline/requirements.txt
```

### Run Full End-to-End Pipeline
```bash
python data_pipeline/main.py
```

### Run Individual Components
- **Run Scraper Only**:
  ```bash
  python data_pipeline/scraper.py
  ```
- **Run Database Loading & Schema Verification**:
  ```bash
  python data_pipeline/database.py
  ```
- **Run SQL Queries & Pandas Comparison**:
  ```bash
  python data_pipeline/queries.py
  ```
