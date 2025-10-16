from unittest.mock import MagicMock, patch
import pandas as pd

from scheduler.tasks.news_fetch import NewsFetchTask
from scheduler.tasks.price_fetch import PriceFetchTask


def test_news_fetch_task_fetches_and_stores_data():
    # 1. Create mock objects for dependencies
    mock_storage = MagicMock()

    # Mock the stock list that the task will use
    stock_list = [{'symbol': 'AAPL', 'name': 'Apple Inc.'}]

    # 2. Instantiate the task we are testing
    news_fetch_task = NewsFetchTask(
        storage=mock_storage,
        stock_list=stock_list
    )

    # Mock the news fetcher function to avoid real API calls
    with patch('scheduler.tasks.news_fetch.fetch_news_for_symbol') as mock_fetch_news:
        mock_fetch_news.return_value = [{'title': 'Test News', 'publishedAt': '2025-10-13', 'content': 'Test content'}]

        # 3. Run the task's logic
        news_fetch_task.run()

    # 4. Assert that the news was fetched and stored
    assert mock_fetch_news.call_count == 1
    assert mock_storage.store_raw_news.call_count == 1


def test_price_fetch_task_fetches_and_stores_data():
    # 1. Create mock objects for dependencies
    mock_storage = MagicMock()
    mock_fetcher = MagicMock()
    stock_symbols = ['AAPL', 'GOOG']

    # 2. Instantiate the task we are testing
    price_fetch_task = PriceFetchTask(
        storage=mock_storage,
        fetcher=mock_fetcher,
        stock_symbols=stock_symbols,
        days=1
    )

    # Mock the fetcher to return some dummy data
    mock_fetcher.fetch_historical_data.return_value = pd.DataFrame({'Close': [100, 101]})

    # 3. Run the task's logic
    price_fetch_task.run()

    # 4. Assert that the fetcher and storage were called for each symbol
    assert mock_fetcher.fetch_historical_data.call_count == len(stock_symbols)
    assert mock_storage.store_historical_data.call_count == len(stock_symbols)
