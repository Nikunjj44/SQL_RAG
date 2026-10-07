"""
Loads the 9 Olist CSVs into Postgres.

Run:  python -m database.load_data
Re-running is safe: it drops and recreates everything.
"""
import pandas as pd

from config import settings
from database.connection import get_engine

# (table name, csv file). Order matters: parents before children.
FILES = [
    ("category_translation", "product_category_name_translation.csv"),
    ("customers", "olist_customers_dataset.csv"),
    ("sellers", "olist_sellers_dataset.csv"),
    ("products", "olist_products_dataset.csv"),
    ("orders", "olist_orders_dataset.csv"),
    ("order_items", "olist_order_items_dataset.csv"),
    ("order_payments", "olist_order_payments_dataset.csv"),
    ("order_reviews", "olist_order_reviews_dataset.csv"),
    ("geolocation", "olist_geolocation_dataset.csv"),
]

# Read zip codes as text so leading zeros survive
ZIP_COLUMNS = {
    "customer_zip_code_prefix": str,
    "seller_zip_code_prefix": str,
    "geolocation_zip_code_prefix": str,
}

DATE_SUFFIXES = ("_date", "_timestamp", "_at")


def clean(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Table-specific fixes discovered in the exploration notebook."""
    if table == "products":
        df = df.rename(columns={
            "product_name_lenght": "product_name_length",
            "product_description_lenght": "product_description_length",
        })

    if table == "order_reviews":
        df = df.drop_duplicates(subset=["review_id", "order_id"])

    # Convert every date-like column to real datetimes
    for col in df.columns:
        if col.endswith(DATE_SUFFIXES):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def run_sql_file(conn, path):
    conn.exec_driver_sql(path.read_text())


def load():
    engine = get_engine(readonly=False)
    db_dir = settings.ROOT / "database"

    # 1. Reset the schema and create tables (such that rerrunning the script ensures clean tables everytime)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        run_sql_file(conn, db_dir / "schema.sql")

    # 2. Load each CSV (the tables we have downloaded from Kaggle - this is just for project prep)
    ## Usually the dataset will already be present at the enterprise level
    for table, filename in FILES:
        df = pd.read_csv(settings.RAW_DIR / filename, dtype=ZIP_COLUMNS)
        df = clean(table, df)
        df.to_sql(
            table,
            engine,
            if_exists="append",   # tables already exist; just insert rows
            index=False,
            chunksize=5_000,
            method="multi",
        )
        print(f"  loaded {table:22s} {len(df):>9,} rows")

    # 3. (Re)grant read-only access; dropping the schema removed old grants
    # AS in the initial step we dropped the tables
    with engine.begin() as conn:
        run_sql_file(conn, db_dir / "read_only_role.sql")
        conn.exec_driver_sql("ANALYZE;")

        ## Why exec_driver_sql instead of text()? SQLAlchemy's text() treats :word as a parameter placeholder, 
        ## which would break on SQL containing :: casts or time literals like '10:00'. 
        ## exec_driver_sql sends the SQL as-is.

    print("Done.")


if __name__ == "__main__":
    load()