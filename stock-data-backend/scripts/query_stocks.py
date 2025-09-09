import argparse
import os
import sys

import time
import requests
import json
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



def trigger_single_stock_fix(symbol: str, days_to_fetch: int) -> bool:
    api_url = "http://127.0.0.1:5000/trigger-fetch-single"  # use 127.0.0.1
    payload = {"symbol": symbol, "days": days_to_fetch}

    print(f"  Triggering API to fetch last {days_to_fetch} days...")
    try:
        # disable env proxies so call goes to localhost
        resp = requests.put(
            api_url,
            json=payload,
            timeout=10,
            headers={"Accept": "application/json"},
            proxies={"http": None, "https": None},
        )
        if resp.status_code == 202:
            print(f"  -> Successfully scheduled fix for {symbol}.")
            return True
        else:
            print(f"  -> ERROR: {resp.status_code} {resp.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  -> ERROR: Could not connect to {api_url}: {e}")
        print(f"  -> To test manually: curl --noproxy '*' -X PUT -H 'Content-Type: application/json' "
              f"-d '{json.dumps(payload)}' {api_url}")
        return False

def check_missing_data_per_stock(storage: Storage, days_to_check: int, auto_fix: bool):
    """
    Checks for missing trading dates for all stocks over a recent period.

    It works by generating a calendar of all recent weekdays (Mon-Fri) and
    then checking each stock against that calendar to find missing data points.
    Note: This may flag market holidays as "missing" days.

    Args:
        storage (Storage): An active storage instance for database access.
        days_to_check (int): The number of recent calendar days to check.
        auto_fix (bool): If True, automatically trigger API calls to fix missing data.
    """
    print(f"\nChecking for missing data in the last {days_to_check} days...")
    print("Generating a calendar of expected weekdays (Mon-Fri) to check against.")

    # 1. Generate a set of all weekdays for the period to check.
    import datetime
    today = datetime.date.today()
    expected_dates = set()
    for i in range(days_to_check):
        d = today - datetime.timedelta(days=i)
        if d.weekday() < 5:  # Monday is 0, Friday is 4
            expected_dates.add(d)

    print(f"Expected weekdays to check: {sorted(list(expected_dates))}")

    with storage.pool.connection() as conn:
        # 2. Get all unique symbols that exist in the database
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

    if not missing_data_report:
        print("\n--- No missing recent dates found. Data is complete! ---")
    else:
        if auto_fix:
            print("\n--- Missing Data Found: Automatically Triggering Fixes ---")
        else:
            print("\n--- Missing Data Found (Dry Run) ---")
        import datetime
        today = datetime.date.today()
        successful_triggers = 0
        failed_triggers = []

        for symbol, dates in missing_data_report.items():
            oldest_missing = min(dates)
            # Calculate days from today to the oldest missing date, add a buffer
            days_to_fetch = (today - oldest_missing).days + 2

            print(f"\n- Found missing dates for {symbol}: {[d.strftime('%Y-%m-%d') for d in dates]}")

            if auto_fix:
                if trigger_single_stock_fix(symbol, days_to_fetch):
                    successful_triggers += 1
                else:
                    failed_triggers.append(symbol)
                
                # Add a delay between API calls to be polite to the backend scheduler and avoid
                # triggering rate limits on the underlying data provider (yfinance).
                time.sleep(1)
            else:
                api_url = "http://localhost:5000/trigger-fetch-single"
                payload = {"symbol": symbol, "days": days_to_fetch}
                print(f"  -> To fix, run: curl -X PUT -H 'Content-Type: application/json' -d '{json.dumps(payload)}' {api_url}")

        if auto_fix:
            print("\n--- Fix Attempt Summary ---")
            print(f"Successfully triggered fixes for {successful_triggers} of {len(missing_data_report)} missing stocks.")
            if failed_triggers:
                print(f"Failed to trigger fixes for {len(failed_triggers)} stocks: {failed_triggers}")


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
        help="Check for missing weekday data on a per-stock basis."
    )
    parser.add_argument(
        "--check-market-days",
        action="store_true",
        help="Check for entire weekdays missing from the database (e.g., failed cron job)."
    )
    parser.add_argument(
        "--test-fix-api",
        action="store_true",
        help="Test the single-stock fix API endpoint with a specific symbol and number of days."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="The stock symbol to use with --test-fix-api."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Number of days to use for checks or for the test-fix API call. (Default: 10)"
    )
    parser.add_argument(
        "--no-fix",
        action="store_true",
        help="Run checks without automatically triggering fixes (dry run)."
    )
    args = parser.parse_args()

    storage = None
    try:
        storage = Storage(config=config)
        print("Database connection established.")

        if args.test_fix_api:
            if not args.symbol:
                parser.error("--symbol is required when using --test-fix-api")
            print(f"--- Testing Fix API for symbol: {args.symbol}, days: {args.days} ---")
            trigger_single_stock_fix(args.symbol, args.days)
        elif args.summary:
            query_stock_summary(storage)
        elif args.check_per_stock:
            check_missing_data_per_stock(storage, days_to_check=args.days, auto_fix=not args.no_fix)
        elif args.check_market_days:
            check_missing_market_days(storage, days_to_check=args.days)
        else:
            # Default action if no flag is provided
            print("No action specified. Use --summary, --check-per-stock, --check-market-days, or --test-fix-api. Showing summary by default.")
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