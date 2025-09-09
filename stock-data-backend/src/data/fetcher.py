import logging

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


class StockDataFetcher:
    """
    Handles fetching data from yfinance.
    yfinance now uses curl_cffi internally to avoid rate-limiting, so we
    no longer manage a session object here.
    """

    def __init__(self):
        # No session management is needed. yfinance handles it.
        log.info("StockDataFetcher initialized.")

    def fetch_historical_data(self, symbol: str, period: str) -> pd.DataFrame:
        """
        Fetches historical data for a given stock symbol.
        """
        ticker = yf.Ticker(symbol)

        # The history call makes the network request.
        hist = ticker.history(period=period)

        return hist