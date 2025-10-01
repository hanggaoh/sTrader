import pytest
import pandas as pd
from unittest.mock import MagicMock

# Imports should be relative to the project root (/app), which is in the PYTHONPATH.
from data.fetcher import StockDataFetcher

# Define mock data that simulates the yfinance response for the history call.
MOCK_HISTORICAL_DATA = pd.DataFrame(
    {
        "Open": [150.0],
        "High": [152.0],
        "Low": [149.0],
        "Close": [151.5],
        "Volume": [1000000],
    },
    index=pd.to_datetime(["2023-01-01"]),
)

@pytest.fixture
def mock_yfinance(mocker):
    """
    Mocks the yfinance.Ticker class.
    
    Returns a tuple of (mock_ticker_class, mock_ticker_instance).
    """
    # The object to patch is where it is looked up. In our case, `yf.Ticker`
    # is looked up in the `src.data.fetcher` module.
    mock_ticker_class = mocker.patch("src.data.fetcher.yf.Ticker")
    
    # When `yf.Ticker("MOCK")` is called, it returns an instance. We want to control
    # that instance to mock its `history` method.
    mock_instance = MagicMock()
    mock_instance.history.return_value = MOCK_HISTORICAL_DATA
    
    # Configure the mock class to return our mock instance.
    mock_ticker_class.return_value = mock_instance
    
    return mock_ticker_class, mock_instance

@pytest.fixture
def fetcher():
    """Returns a fresh instance of StockDataFetcher for each test."""
    return StockDataFetcher()

def test_fetch_historical_data(fetcher, mock_yfinance):
    """
    Test that fetching historical data calls yfinance.Ticker(...).history(...) correctly.
    """
    mock_ticker_class, mock_instance = mock_yfinance
    
    symbol = "MOCK"
    period = "1y"
    
    # 1. Call the method under test
    data = fetcher.fetch_historical_data(symbol=symbol, period=period)

    # 2. Assert that the underlying yfinance objects were used correctly.
    # Assert that yf.Ticker was instantiated with the correct symbol.
    mock_ticker_class.assert_called_once_with(symbol)

    # Assert that the history method was called on the instance with the correct period.
    mock_instance.history.assert_called_once_with(period=period)

    # 3. Assert that the data returned is the mocked data.
    pd.testing.assert_frame_equal(data, MOCK_HISTORICAL_DATA)
