# Zepto Data & AI Platform — Capstone Project

This repository contains the complete, production-ready implementation of the **Zepto Data & AI Platform Capstone Project**. The platform integrates three core capabilities into a single repository:

```
zepto-capstone/
├── data_pipeline/        # Module 1: Data Engineering Pipeline (Scraping, Cleaning, SQLite DB, Queries)
├── analytics/            # Module 2: Analytics & Predictive Modeling (EDA, ML, Tuning, Persistence)
├── support_assistant/    # Module 3: Grounded GenAI Support Assistant (RAG, ChromaDB, LangGraph, FastAPI)
└── README.md             # Top-level project documentation
```

---

## Module Summaries & Quickstart

### [Module 1: Data Engineering Pipeline](./data_pipeline/README.md)
An automated data pipeline that scrapes live e-commerce book catalog data from [Books to Scrape](https://books.toscrape.com/), cleans and transforms the fields, calculates fixed-rate INR conversions (`1 GBP = 105.50 INR`), loads data into a normalized SQLite relational schema (`categories` and `books` with active PK/FK constraints), and executes/verifies analytical SQL queries alongside `pandas.read_sql` and `pandas.merge` equivalence tests.

**Run Module 1:**
```bash
pip install -r data_pipeline/requirements.txt
python data_pipeline/main.py
```

---

### [Module 2: Analytics & Machine Learning Pipeline](./analytics/README.md)
A comprehensive analytics and predictive modeling workflow on the Titanic dataset featuring statistical data profiling, missing-value strategy (threshold-based), univariate/bivariate/multivariate visualizations, z-score exploratory sanity checks, leak-free classification modeling (Logistic Regression, Decision Tree with `plot_tree`, Random Forest), class-imbalance comparison (balanced weights vs. training-only SMOTE), Random Forest `GridSearchCV` with OOB scoring, multivariate fare regression with heteroscedasticity analysis, and model persistence via `joblib`.

**Run Module 2:**
```bash
pip install -r analytics/requirements.txt
python analytics/main.py
```

---

### [Module 3: Grounded GenAI Support Assistant](./support_assistant/README.md)
A policy-grounded Customer Support AI Assistant built with a local Retrieval-Augmented Generation (RAG) architecture using **SentenceTransformers** (`all-MiniLM-L6-v2`), **ChromaDB**, **LangGraph** StateGraph intent routing (`classify_intent` $\rightarrow$ `retrieve_and_answer` / `direct_answer`), structured **Pydantic** schema output (`answer`, `sources`, `confidence`), and a local **FastAPI** service (`POST /ask`). The default baseline runs 100% offline in deterministic `MOCK_LLM` mode with zero API key dependencies.

**Run Module 3:**
```bash
pip install -r support_assistant/requirements.txt
python support_assistant/test_assistant.py
# Start FastAPI service
python -m uvicorn app.main:app --app-dir support_assistant --host 0.0.0.0 --port 7860
```
## Git Workflow

The project was developed using a feature branch and merged back into the `main` branch. The repository history contains multiple commits documenting the development workflow.