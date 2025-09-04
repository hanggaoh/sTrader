import argparse
import os
import sys

import pandas as pd

# Add the project's 'src' directory to the Python path to allow for module imports
# like 'config' and 'data.storage'.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config import config
from data.storage import Storage


def query_stock_summary(storage: Storage):
    """
    Queries the database for a summary of stored stock data,
    including which symbols are present, the count of records,
    and the date range for each.

    Args:
        storage (Storage): An active storage instance for database access.
    """
    print("Querying stock summary...")
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
    with storage.pool.connection() as conn:
        df = pd.read_sql(query, conn)
    if df.empty:
        print("\nNo stock data found in the database.")
        return
    print("\n--- Summary of Stock Data in Database ---")
    print(df.to_string(index=False))
    print(f"\nFound data for a total of {len(df)} unique stock symbols.")


def check_missing_data_per_stock(storage: Storage, days_to_check: int, reference_symbol: str):
    """
    Checks for missing trading dates for all stocks over a recent period.

    It uses a reliable reference symbol to determine the actual trading days
    and then compares each stock's data against that calendar.

    Args:
        storage (Storage): An active storage instance for database access.
        days_to_check (int): The number of recent calendar days to check.
        reference_symbol (str): A liquid stock symbol to use as a market calendar.
    """
    print(f"\nChecking for missing data in the last {days_to_check} days...")
    print(f"Using '{reference_symbol}' as the reference for trading days.")
    with storage.pool.connection() as conn:
        # 1. Get the actual trading dates from the reference symbol
        ref_query = """
            SELECT DISTINCT(time::date) AS trade_date
            FROM stock_data
            WHERE stock_symbol = %s AND time >= NOW() - (%s * INTERVAL '1 day')
            ORDER BY trade_date;
        """
        ref_df = pd.read_sql(ref_query, conn, params=(reference_symbol, days_to_check))
        if ref_df.empty:
            print(f"\n[ERROR] Could not find any data for reference symbol '{reference_symbol}' in the last {days_to_check} days.")
            print("This usually means the database has not been populated with sufficient historical data.")
            print("\nSuggestion: Run a full data backfill using the API endpoint before running this check.")
            return

        expected_dates = set(ref_df['trade_date'])
        print(f"Expected trading dates: {sorted(list(expected_dates))}")

        # 2. Get all unique symbols in the database
        all_symbols_df = pd.read_sql("SELECT DISTINCT(stock_symbol) FROM stock_data;", conn)
        all_symbols = all_symbols_df['stock_symbol'].tolist()

        # 3. For each symbol, check for missing dates
        missing_data_report = {}
        for i, symbol in enumerate(all_symbols):
            if (i + 1) % 200 == 0:
                print(f"  ...checked {i + 1}/{len(all_symbols)} symbols")

            symbol_query = """
                SELECT DISTINCT(time::date) AS trade_date
                FROM stock_data
                WHERE stock_symbol = %s AND time >= NOW() - (%s * INTERVAL '1 day');
            """
            symbol_df = pd.read_sql(symbol_query, conn, params=(symbol, days_to_check))
            actual_dates = set(symbol_df['trade_date'])

            missing_dates = sorted(list(expected_dates - actual_dates))
            if missing_dates:
                missing_data_report[symbol] = missing_dates

    # 4. Print the final report
    if not missing_data_report:
        print("\n--- No missing recent dates found. Data is complete! ---")
    else:
        print("\n--- Missing Data Found ---")
        for symbol, dates in missing_data_report.items():
            print(f"Symbol {symbol} is missing dates: {[d.strftime('%Y-%m-%d') for d in dates]}")
        print("--------------------------")
        print(f"Found missing data for {len(missing_data_report)} symbols.")


def check_missing_market_days(storage: Storage, days_to_check: int):
    """
    Checks for entire market days missing from the database over a recent period.

    This is useful for detecting if the daily cron job failed to run entirely
    on a particular weekday. It works by comparing all weekdays in a date
    range against the unique dates present in the database.

    Args:
        storage (Storage): An active storage instance for database access.
        days_to_check (int): The number of recent calendar days to check.
    """
    print(f"\nChecking for missing market weekdays in the last {days_to_check} days...")

    # 1. Get all unique dates from the database for the period
    query = """
        SELECT DISTINCT(time::date) AS trade_date
        FROM stock_data
        WHERE time >= NOW() - (%s * INTERVAL '1 day')
        ORDER BY trade_date;
    """
    with storage.pool.connection() as conn:
        db_dates_df = pd.read_sql(query, conn, params=(days_to_check,))

    db_dates = set(db_dates_df['trade_date'])
    print(f"Found data for {len(db_dates)} unique dates in the database.")

    # 2. Generate all possible weekdays for the period
    import datetime
    today = datetime.date.today()
    expected_weekdays = set()
    for i in range(days_to_check):
        d = today - datetime.timedelta(days=i)
        # Monday is 0 and Sunday is 6
        if d.weekday() < 5:  # 0, 1, 2, 3, 4 are Mon-Fri
            expected_weekdays.add(d)

    # 3. Find the difference
    missing_days = sorted(list(expected_weekdays - db_dates))

    # 4. Report the results
    if not missing_days:
        print("\n--- No missing market weekdays found. It seems the cron job ran successfully. ---")
    else:
        print("\n--- Potential Missing Market Days Found ---")
        print("The following weekdays are missing from the database. This could be due to a market holiday or a failed cron job.")
        for day in missing_days:
            print(f"  - {day.strftime('%Y-%m-%d')} ({day.strftime('%A')})")
        print("-------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Query and analyze stock data in the database.")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show a summary of all stocks in the database (record count, date range)."
    )
    parser.add_argument(
        "--check-per-stock",
        action="store_true",
        help="Check for missing trading dates on a per-stock basis (requires a reliable --ref-symbol)."
    )
    parser.add_argument(
        "--check-market-days",
        action="store_true",
        help="Check for entire weekdays missing from the database (e.g., failed cron job)."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="How many days back to check for missing data. (Default: 10)"
    )
    parser.add_argument(
        "--ref-symbol",
        type=str,
        default="689009.SS",
        help="Reference symbol for market calendar. (Default: 689009.SS)"
    )
    args = parser.parse_args()

    storage = None
    try:
        storage = Storage(config=config)
        print("Database connection established.")

        if args.summary:
            query_stock_summary(storage)
        elif args.check_per_stock:
            check_missing_data_per_stock(storage, days_to_check=args.days, reference_symbol=args.ref_symbol)
        elif args.check_market_days:
            check_missing_market_days(storage, days_to_check=args.days)
        else:
            # Default action if no flag is provided
            print("No action specified. Use --summary, --check-per-stock, or --check-market-days. Showing summary by default.")
            query_stock_summary(storage)
    except Exception as e:
        print(f"A critical error occurred: {e}")
    finally:
        if storage:
            storage.close()
            print("\nDatabase connection closed.")
        print("--- Script finished ---")


if __name__ == "__main__":
    main()