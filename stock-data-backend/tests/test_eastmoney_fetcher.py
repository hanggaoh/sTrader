import unittest
from unittest.mock import patch
import pandas as pd

from data.eastmoney_fetcher import EastmoneyFetcher

class TestEastmoneyFetcher(unittest.TestCase):

    @patch('data.eastmoney_fetcher.EastmoneyFetcher._stock_news_em_all')
    def test_fetch_news_for_symbol_filters_correctly_client_side(self, mock_stock_news_em_all):
        # --- Mock Setup ---
        # This mock data simulates the API returning a wide range of dates,
        # ignoring any date filters we might have sent.
        mock_data = pd.DataFrame({
            'date': [
                '2024-01-10 12:00:00', # Before range
                '2024-01-11 12:00:00', # In range
                '2024-01-15 12:00:00', # In range
                '2024-01-17 12:00:00', # In range
                '2024-01-18 12:00:00', # After range
            ],
            'title': ['News A', 'News B', 'News C', 'News D', 'News E'],
            'content': ['Content A', 'Content B', 'Content C', 'Content D', 'Content E'],
            'art_url': ['a', 'b', 'c', 'd', 'e']
        })
        mock_stock_news_em_all.return_value = mock_data

        # --- Test Execution ---
        fetcher = EastmoneyFetcher()
        result_df = fetcher.fetch_news_for_symbol(
            symbol="000123.SZ",
            start_date="2024-01-11",
            end_date="2024-01-17"
        )

        # --- Assertions ---
        # 1. Check that the underlying API method was called with all the correct parameters.
        mock_stock_news_em_all.assert_called_once_with(
            symbol='000123',
            start_date='2024-01-11',
            end_date='2024-01-17',
            page_size=100,
            max_pages=50
        )

        # 2. Check that the client-side filtering works correctly.
        # Even though the mock returned 5 articles, we should only have 3 after filtering.
        self.assertEqual(len(result_df), 3)
        self.assertEqual(result_df['title'].iloc[0], 'News B')
        self.assertEqual(result_df['title'].iloc[-1], 'News D')

if __name__ == '__main__':
    unittest.main()
