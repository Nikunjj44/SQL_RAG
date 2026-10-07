"""
Runs a SELECT query as the read-only user and returns a pandas DataFrame.
"""
from decimal import Decimal

import pandas as pd

from database.connection import get_engine


def _decimals_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """Postgres NUMERIC arrives as Python Decimal; convert for pandas/JSON."""
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, Decimal)).any():
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def run_select(sql: str) -> pd.DataFrame:
    """Execute SQL on the read-only connection. Raises on database errors."""
    conn = get_engine(readonly=True).raw_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
    finally:
        conn.close()   # returns connection to the pool; uncommitted work is rolled back

    return _decimals_to_float(pd.DataFrame(rows, columns=columns))