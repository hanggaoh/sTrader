import logging
from datetime import date
from typing import Optional, List, Dict

import requests

from config import config

log = logging.getLogger(__name__)

def fetch_news_for_symbol(
    symbol: str, 
    stock_name: str, 
    from_date: Optional[date] = None, 
    to_date: Optional[date] = None
) -> Optional[List[Dict]]:
    """
    Fetches news articles for a given stock symbol by searching for its name within a date range.

    Args:
        symbol: The stock symbol (e.g., '000001.SZ').
        stock_name: The name of the stock to search for (e.g., '平安银行').
        from_date: The start date for the news search.
        to_date: The end date for the news search.

    Returns:
        A list of article dictionaries, or None if the API key is missing or an error occurs.
    """
    if not config.news_api_key:
        log.warning("NEWS_API_KEY is not set in the configuration. Cannot fetch news.")
        return None

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q=\"{stock_name}\"&"
        f"language=zh&"
        f"sortBy=publishedAt&"
        f"apiKey={config.news_api_key}"
    )

    if from_date and to_date:
        url += f"&from={from_date.isoformat()}&to={to_date.isoformat()}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        
        if data.get("status") == "ok":
            log.info(f"Successfully fetched {len(data.get('articles', []))} articles for {stock_name}.")
            return data.get('articles', [])
        else:
            log.error(f"NewsAPI returned an error: {data.get('message')}")
            return None

    except requests.exceptions.RequestException as e:
        log.error(f"Failed to fetch news for {stock_name}. Error: {e}")
        return None
