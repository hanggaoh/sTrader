import akshare as ak
import pandas as pd
import json
import os

def fetch_all_stock_symbols():
    """
    Fetches a comprehensive list of stock symbols from the Shanghai and Shenzhen exchanges
    using the akshare library.

    Returns:
        list: A list of stock symbols formatted for yfinance (e.g., '600519.SS').
    """
    print("Fetching all A-share stock lists from East Money...")
    all_symbols = []

    try:
        # Fetch all A-shares (Shanghai and Shenzhen) from the East Money source
        stock_df = ak.stock_zh_a_spot_em()

        sse_count = 0
        szse_count = 0

        # Determine the exchange based on the stock code prefix
        for code in stock_df['代码']:
            if code.startswith('6'):  # Shanghai Stock Exchange
                all_symbols.append(f"{code}.SS")
                sse_count += 1
            elif code.startswith('0') or code.startswith('3'):  # Shenzhen Stock Exchange
                all_symbols.append(f"{code}.SZ")
                szse_count += 1

        print(f"Found {sse_count} stocks on the Shanghai exchange.")
        print(f"Found {szse_count} stocks on the Shenzhen exchange.")

    except Exception as e:
        print(f"An error occurred while fetching stock data: {e}")
        return None

    return sorted(list(set(all_symbols)))  # Return a sorted list of unique symbols

def main():
    """
    Main function to run the script. Fetches symbols and saves them to the JSON file.
    """
    symbols = fetch_all_stock_symbols()
    if symbols:
        print(f"\nTotal unique symbols fetched: {len(symbols)}")
        # The JSON file is located at src/data/chinese_stocks.json
        json_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'chinese_stocks.json')
        print(f"Writing symbols to {os.path.abspath(json_path)}...")
        with open(json_path, 'w') as f:
            json.dump(symbols, f, indent=4)
        print("Successfully updated the stock list.")

if __name__ == "__main__":
    main()