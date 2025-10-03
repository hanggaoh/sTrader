import os
import sys
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.sentiment import fetch_news_for_symbol
from config import config
def test_fetch_news():
    """
    Tests the fetch_news_for_symbol function.
    """
    load_dotenv() # Load environment variables from .env file

    # --- Test Case ---
    # Use a well-known stock for testing
    test_symbol = "002806.SZ "
    test_stock_name = "002806.SZ"

    print(f"--- Testing news fetch for {test_stock_name} ({test_symbol}) ---")

    # --- Check for API Key ---
    if not config.news_api_key:
        print("ERROR: NEWS_API_KEY not found in environment variables.")
        print("Please create a .env file in the project root and add your key:")
        print("NEWS_API_KEY=your_api_key")
        return

    # --- Call the function ---
    articles = fetch_news_for_symbol(test_symbol, test_stock_name)

    # --- Print the results ---
    if articles is not None:
        print(f"Successfully fetched {len(articles)} articles.")
        if articles:
            print("--- First 3 article titles: ---")
            for i, article in enumerate(articles[:3]):
                print(f"{i+1}. {article.get('title', 'No Title')}")
    else:
        print("Failed to fetch articles. Check logs for more details.")

if __name__ == "__main__":
    test_fetch_news()
