"""
scraper.py
==========
Data scraper and cleaning module for the Zepto Data & AI Platform Capstone.
Scrapes book catalog information from https://books.toscrape.com/ across multiple
categories, cleans the extracted fields, and computes fixed-rate INR conversions.
"""

import os
import re
import sys
import urllib.parse
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Ensure standard output can handle UTF-8 characters on all platforms
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "https://books.toscrape.com/"
INR_PER_GBP = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


def get_category_links(base_url=BASE_URL):
    """
    Fetch category names and relative URLs from the homepage sidebar.
    """
    response = requests.get(base_url, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    categories = {}
    sidebar_links = soup.select(".side_categories ul li ul li a")
    for link in sidebar_links:
        cat_name = link.get_text(strip=True)
        cat_rel_url = link.get("href")
        cat_full_url = urllib.parse.urljoin(base_url, cat_rel_url)
        categories[cat_name] = cat_full_url

    return categories


def scrape_category_books(cat_name, start_url, max_books=None):
    """
    Scrape all books in a category, following pagination if available.
    """
    books = []
    current_url = start_url

    while current_url:
        response = requests.get(current_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        pods = soup.select("article.product_pod")
        for pod in pods:
            # 1. Title
            title_tag = pod.h3.find("a")
            title = title_tag.get("title") if title_tag and title_tag.get("title") else (title_tag.get_text(strip=True) if title_tag else "")

            # 2. Price (GBP) listed text
            price_tag = pod.select_one(".price_color")
            raw_price = price_tag.get_text(strip=True) if price_tag else ""

            # 3. Star rating text
            rating_tag = pod.select_one("p.star-rating")
            rating_text = ""
            if rating_tag:
                classes = rating_tag.get("class", [])
                for cls in classes:
                    if cls != "star-rating":
                        rating_text = cls
                        break

            # 4. Availability text
            avail_tag = pod.select_one(".availability")
            availability_text = avail_tag.get_text(strip=True) if avail_tag else ""

            books.append({
                "title": title,
                "price_raw": raw_price,
                "rating_raw": rating_text,
                "availability_raw": availability_text,
                "category": cat_name
            })

            if max_books and len(books) >= max_books:
                return books

        # Check for next page
        next_button = soup.select_one("li.next a")
        if next_button:
            next_rel = next_button.get("href")
            current_url = urllib.parse.urljoin(current_url, next_rel)
        else:
            current_url = None

    return books


def scrape_books_catalog(target_categories=None, min_books=60):
    """
    Scrapes books across multiple categories until at least min_books are collected.
    """
    all_categories = get_category_links()
    
    if target_categories:
        categories_to_scrape = {k: all_categories[k] for k in target_categories if k in all_categories}
    else:
        # Pick default top categories to ensure >= 3 categories and >= 60 books
        # Travel (11), Mystery (32), Historical Fiction (26), Sequential Art (75) = 144 books
        default_cats = ["Travel", "Mystery", "Historical Fiction", "Sequential Art"]
        categories_to_scrape = {k: all_categories[k] for k in default_cats if k in all_categories}

    all_books = []
    print(f"Starting scrape across {len(categories_to_scrape)} categories...")

    for cat_name, cat_url in categories_to_scrape.items():
        print(f"  Scraping category: {cat_name}...")
        cat_books = scrape_category_books(cat_name, cat_url)
        print(f"    -> Extracted {len(cat_books)} books.")
        all_books.extend(cat_books)

    print(f"Scraping complete. Total raw books collected: {len(all_books)}")
    return all_books


def clean_and_transform_data(raw_books):
    """
    Clean raw scraped fields and apply fixed currency conversions:
    - price_gbp: stripped of currency symbol, parsed to float
    - rating: converted from string (One..Five) to integer (1..5)
    - in_stock: boolean/integer (1 for True, 0 for False)
    - price_inr: calculated as price_gbp * 105.50
    """
    df = pd.DataFrame(raw_books)
    if df.empty:
        return df

    # 1. Clean Price GBP
    def parse_price(val):
        if pd.isna(val):
            return None
        cleaned = re.sub(r"[^0-9.]", "", str(val))
        try:
            return float(cleaned)
        except ValueError:
            return None

    df["price_gbp"] = df["price_raw"].apply(parse_price)

    # Impute missing prices with median if any parsing failed
    if df["price_gbp"].isnull().any():
        median_price = df["price_gbp"].median()
        df["price_gbp"] = df["price_gbp"].fillna(median_price)

    df["price_gbp"] = df["price_gbp"].astype(float)

    # 2. Fixed Currency Conversion: 1 GBP = 105.50 INR
    df["price_inr"] = (df["price_gbp"] * INR_PER_GBP).round(2)

    # 3. Clean Star Rating
    def parse_rating(val):
        return RATING_MAP.get(str(val).strip(), None)

    df["rating"] = df["rating_raw"].apply(parse_rating)
    if df["rating"].isnull().any():
        median_rating = int(df["rating"].dropna().median())
        df["rating"] = df["rating"].fillna(median_rating)
    
    df["rating"] = df["rating"].astype(int)

    # 4. Clean Availability / in_stock
    def parse_availability(val):
        text = str(val).lower()
        return 1 if "in stock" in text else 0

    df["in_stock"] = df["availability_raw"].apply(parse_availability).astype(int)

    # Clean title and category strings
    df = df.dropna(subset=["title", "category"])
    df["title"] = df["title"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    return df


def run_scraper_pipeline(output_dir="data_pipeline/data"):
    """
    Orchestrate scraping, cleaning, and persistence to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    raw_books = scrape_books_catalog()
    raw_df = pd.DataFrame(raw_books)
    raw_csv_path = os.path.join(output_dir, "raw_scraped_books.csv")
    raw_df.to_csv(raw_csv_path, index=False, encoding="utf-8")

    cleaned_df = clean_and_transform_data(raw_books)
    cleaned_csv_path = os.path.join(output_dir, "cleaned_books.csv")
    cleaned_df.to_csv(cleaned_csv_path, index=False, encoding="utf-8")

    print(f"Raw data saved to: {raw_csv_path}")
    print(f"Cleaned data saved to: {cleaned_csv_path}")
    return cleaned_df


if __name__ == "__main__":
    df = run_scraper_pipeline()
    print("\n--- Scraper Pipeline Summary ---")
    print(f"Total Books: {len(df)}")
    print(f"Categories: {df['category'].nunique()} ({list(df['category'].unique())})")
    print("Columns & Dtypes:")
    print(df.dtypes)
    print("\nFirst 3 Records:")
    for idx, row in df.head(3).iterrows():
        print(f"  [{row['category']}] {row['title'][:30]} | GBP {row['price_gbp']:.2f} | INR {row['price_inr']:.2f} | Rating: {row['rating']} | Stock: {row['in_stock']}")
