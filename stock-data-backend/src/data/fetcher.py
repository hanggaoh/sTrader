import logging

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


class StockDataFetcher:
    """
    A class to fetch data from yfinance.
    """

    def fetch_historical_data(self, symbol: str, period: str) -> pd.DataFrame:
        """
        Fetches historical data for a given stock symbol.
        """
        ticker = yf.Ticker(symbol)

        # The history call makes the network request.
        hist = ticker.history(period=period)

        return hist
