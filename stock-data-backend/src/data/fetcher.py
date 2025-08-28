import yfinance as yf
import pandas as pd


class StockDataFetcher:
    """
    A class to fetch various types of stock data using the yfinance library.
    It can retrieve historical market data, company information, financial statements, and more.
    """

    def __init__(self):
        """
        Initializes the StockDataFetcher.
        """
        pass

    def _get_ticker(self, stock_symbol: str) -> yf.Ticker:
        """Helper method to create and return a yfinance Ticker object."""
        return yf.Ticker(stock_symbol)

    def fetch_stock_data(self, stock_symbol: str) -> pd.DataFrame:
        """
        Fetches the last day of historical data for a stock.
        This method is kept for compatibility with existing modules like the scheduler.
        For more flexible data fetching, use `fetch_historical_data`.
        """
        return self.fetch_historical_data(stock_symbol, period="1d", interval="1d")

    def fetch_historical_data(
        self,
        stock_symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
        """
        Fetches historical market data for a given stock symbol over a specified period.

        Args:
            stock_symbol (str): The stock ticker symbol (e.g., 'AAPL', '600519.SS').
            period (str, optional): The period of data to fetch. Valid periods: "1d", "5d",
                "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max". Defaults to "1y".
            interval (str, optional): The data interval. Valid intervals: "1m", "2m", "5m",
                "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo". Defaults to "1d".
            start (str, optional): The start date for the data ('YYYY-MM-DD'). Overrides 'period'.
                Defaults to None.
            end (str, optional): The end date for the data ('YYYY-MM-DD'). Defaults to None.

        Returns:
            pd.DataFrame: A pandas DataFrame with the historical data (OHLC, Volume).
                          Returns an empty DataFrame if no data is found.
        """
        stock = self._get_ticker(stock_symbol)
        hist_data = stock.history(
            period=period,
            interval=interval,
            start=start,
            end=end,
            auto_adjust=True,  # Adjusts for splits and dividends
        )
        return hist_data

    def get_stock_info(self, stock_symbol: str) -> dict:
        """
        Fetches a dictionary of comprehensive information for the stock.
        This includes sector, industry, business summary, market cap, P/E ratios, and more.

        Args:
            stock_symbol (str): The stock ticker symbol.

        Returns:
            dict: A dictionary of stock information. Returns an empty dict if the ticker is invalid.
        """
        stock = self._get_ticker(stock_symbol)
        # A quick check to see if the ticker is valid before returning info
        if not stock.history(period="1d").empty:
            return stock.info
        return {}

    def get_financials(self, stock_symbol: str, quarterly: bool = False) -> pd.DataFrame:
        """
        Fetches financial statements (Income Statement).

        Args:
            stock_symbol (str): The stock ticker symbol.
            quarterly (bool, optional): If True, returns quarterly financials. Defaults to False (annual).

        Returns:
            pd.DataFrame: A DataFrame of financial data.
        """
        stock = self._get_ticker(stock_symbol)
        return stock.quarterly_financials if quarterly else stock.financials

    def get_balance_sheet(self, stock_symbol: str, quarterly: bool = False) -> pd.DataFrame:
        """
        Fetches the balance sheet.

        Args:
            stock_symbol (str): The stock ticker symbol.
            quarterly (bool, optional): If True, returns quarterly balance sheet. Defaults to False (annual).

        Returns:
            pd.DataFrame: A DataFrame of the balance sheet.
        """
        stock = self._get_ticker(stock_symbol)
        return stock.quarterly_balance_sheet if quarterly else stock.balance_sheet

    def get_cash_flow(self, stock_symbol: str, quarterly: bool = False) -> pd.DataFrame:
        """
        Fetches the cash flow statement.

        Args:
            stock_symbol (str): The stock ticker symbol.
            quarterly (bool, optional): If True, returns quarterly cash flow. Defaults to False (annual).

        Returns:
            pd.DataFrame: A DataFrame of the cash flow statement.
        """
        stock = self._get_ticker(stock_symbol)
        return stock.quarterly_cashflow if quarterly else stock.cashflow

    def get_dividends(self, stock_symbol: str) -> pd.Series:
        """
        Fetches dividend history.

        Args:
            stock_symbol (str): The stock ticker symbol.

        Returns:
            pd.Series: A Series with dates as index and dividend amounts as values.
        """
        stock = self._get_ticker(stock_symbol)
        return stock.dividends

    def get_news(self, stock_symbol: str) -> list:
        """
        Fetches recent news articles related to the stock.

        Args:
            stock_symbol (str): The stock ticker symbol.

        Returns:
            list: A list of dictionaries, where each dictionary represents a news article.
        """
        stock = self._get_ticker(stock_symbol)
        return stock.news