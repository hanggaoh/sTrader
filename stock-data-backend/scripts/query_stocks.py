import os
import sys

import pandas as pd

# Add the project's 'src' directory to the Python path to allow for module imports
# like 'config' and 'data.storage'.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config import config
from data.storage import Storage


def query_stock_summary():
    """
    Queries the database for a summary of stored stock data,
    including which symbols are present, the count of records,
    and the date range for each.
    """
    print("Connecting to the database to query stock summary...")
    storage = None
    try:
        storage = Storage(config=config)

        # The SQL query to get the summary for each stock symbol
        query = """
            SELECT
                stock_symbol,
                COUNT(*) AS record_count,
                MIN(time)::date AS earliest_record,
                MAX(time)::date AS latest_record
            FROM
                stock_data
            GROUP BY
                stock_symbol
            ORDER BY
                stock_symbol;
        """

        # Use a connection from the pool to execute the query with pandas
        with storage.pool.connection() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            print("\nNo stock data found in the database.")
            return

        print("\n--- Summary of Stock Data in Database ---")
        print(df.to_string(index=False))
        print(f"\nFound data for a total of {len(df)} unique stock symbols.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure the database is running and accessible.")
    finally:
        if storage:
            storage.close()
            print("\nDatabase connection closed.")


if __name__ == "__main__":
    query_stock_summary()