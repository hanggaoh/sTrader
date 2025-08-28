import pytest
import pandas as pd
from unittest.mock import MagicMock, PropertyMock


from data.fetcher import StockDataFetcher

# Define mock data that simulates yfinance responses
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

MOCK_INFO_DATA = {
    "symbol": "MOCK",
    "longName": "Mock Inc.",
    "sector": "Technology",
    "marketCap": 1e12,
}

MOCK_FINANCIALS_DATA = pd.DataFrame(
    {"2023-12-31": [1000], "2022-12-31": [900]}, index=["Total Revenue"]
)
MOCK_QUARTERLY_FINANCIALS_DATA = pd.DataFrame(
    {"2023-12-31": [250], "2023-09-30": [240]}, index=["Total Revenue"]
)

MOCK_DIVIDENDS_DATA = pd.Series(
    [0.5, 0.55], index=pd.to_datetime(["2023-06-01", "2023-09-01"]), name="Dividends"
)

MOCK_NEWS_DATA = [
    {
        "uuid": "1",
        "title": "Mock Stock Surges",
        "publisher": "News Co",
        "link": "http://example.com/news1",
    }
]


@pytest.fixture
def mock_yfinance_ticker(mocker):
    """Fixture to mock the yfinance.Ticker object and its methods/properties."""
    mock_ticker_instance = MagicMock()

    # Create PropertyMock objects and attach them to the mock instance
    # so we can assert on them later.
    mock_ticker_instance.mock_info = PropertyMock(return_value=MOCK_INFO_DATA)
    mock_ticker_instance.mock_financials = PropertyMock(return_value=MOCK_FINANCIALS_DATA)
    mock_ticker_instance.mock_quarterly_financials = PropertyMock(
        return_value=MOCK_QUARTERLY_FINANCIALS_DATA
    )
    mock_ticker_instance.mock_balance_sheet = PropertyMock(
        return_value=MOCK_FINANCIALS_DATA
    )
    mock_ticker_instance.mock_quarterly_balance_sheet = PropertyMock(
        return_value=MOCK_QUARTERLY_FINANCIALS_DATA
    )
    mock_ticker_instance.mock_cashflow = PropertyMock(return_value=MOCK_FINANCIALS_DATA)
    mock_ticker_instance.mock_quarterly_cashflow = PropertyMock(
        return_value=MOCK_QUARTERLY_FINANCIALS_DATA
    )
    mock_ticker_instance.mock_dividends = PropertyMock(return_value=MOCK_DIVIDENDS_DATA)
    mock_ticker_instance.mock_news = PropertyMock(return_value=MOCK_NEWS_DATA)

    # Configure the type of the mock instance to use these PropertyMocks
    type(mock_ticker_instance).info = mock_ticker_instance.mock_info
    type(mock_ticker_instance).financials = mock_ticker_instance.mock_financials
    type(
        mock_ticker_instance
    ).quarterly_financials = mock_ticker_instance.mock_quarterly_financials
    type(mock_ticker_instance).balance_sheet = mock_ticker_instance.mock_balance_sheet
    type(
        mock_ticker_instance
    ).quarterly_balance_sheet = mock_ticker_instance.mock_quarterly_balance_sheet
    type(mock_ticker_instance).cashflow = mock_ticker_instance.mock_cashflow
    type(mock_ticker_instance).quarterly_cashflow = mock_ticker_instance.mock_quarterly_cashflow
    type(mock_ticker_instance).dividends = mock_ticker_instance.mock_dividends
    type(mock_ticker_instance).news = mock_ticker_instance.mock_news

    # Mock methods
    mock_ticker_instance.history.return_value = MOCK_HISTORICAL_DATA

    # Patch the Ticker class in the fetcher module's namespace
    mocker.patch("data.fetcher.yf.Ticker", return_value=mock_ticker_instance)
    return mock_ticker_instance


@pytest.fixture
def fetcher():
    """Returns a fresh instance of StockDataFetcher for each test."""
    return StockDataFetcher()


def test_fetch_historical_data(fetcher, mock_yfinance_ticker):
    """Test fetching historical data calls yfinance correctly."""
    data = fetcher.fetch_historical_data("MOCK", period="1y", start="2023-01-01")
    mock_yfinance_ticker.history.assert_called_once_with(
        period="1y", interval="1d", start="2023-01-01", end=None, auto_adjust=True
    )
    pd.testing.assert_frame_equal(data, MOCK_HISTORICAL_DATA)


def test_fetch_stock_data(fetcher, mock_yfinance_ticker):
    """Test the compatibility method fetch_stock_data."""
    data = fetcher.fetch_stock_data("MOCK")
    mock_yfinance_ticker.history.assert_called_once_with(
        period="1d", interval="1d", start=None, end=None, auto_adjust=True
    )
    pd.testing.assert_frame_equal(data, MOCK_HISTORICAL_DATA)


def test_get_stock_info_valid(fetcher, mock_yfinance_ticker):
    """Test getting stock info for a valid ticker."""
    info = fetcher.get_stock_info("MOCK")
    assert info == MOCK_INFO_DATA
    # Check that history() was called to validate the ticker
    mock_yfinance_ticker.history.assert_called_once_with(period="1d")
    mock_yfinance_ticker.mock_info.assert_called_once()


def test_get_stock_info_invalid(fetcher, mock_yfinance_ticker):
    """Test getting stock info for an invalid ticker returns an empty dict."""
    mock_yfinance_ticker.history.return_value = pd.DataFrame()  # Simulate invalid ticker
    info = fetcher.get_stock_info("INVALID")
    assert info == {}


def test_get_financials(fetcher, mock_yfinance_ticker):
    """Test getting annual and quarterly financials."""
    # Test annual
    annual_data = fetcher.get_financials("MOCK", quarterly=False)
    pd.testing.assert_frame_equal(annual_data, MOCK_FINANCIALS_DATA)
    mock_yfinance_ticker.mock_financials.assert_called_once()

    # Test quarterly
    quarterly_data = fetcher.get_financials("MOCK", quarterly=True)
    pd.testing.assert_frame_equal(quarterly_data, MOCK_QUARTERLY_FINANCIALS_DATA)
    mock_yfinance_ticker.mock_quarterly_financials.assert_called_once()


def test_get_balance_sheet(fetcher, mock_yfinance_ticker):
    """Test getting annual and quarterly balance sheets."""
    # Test annual
    annual_data = fetcher.get_balance_sheet("MOCK", quarterly=False)
    pd.testing.assert_frame_equal(annual_data, MOCK_FINANCIALS_DATA)
    mock_yfinance_ticker.mock_balance_sheet.assert_called_once()

    # Test quarterly
    quarterly_data = fetcher.get_balance_sheet("MOCK", quarterly=True)
    pd.testing.assert_frame_equal(quarterly_data, MOCK_QUARTERLY_FINANCIALS_DATA)
    mock_yfinance_ticker.mock_quarterly_balance_sheet.assert_called_once()


def test_get_cash_flow(fetcher, mock_yfinance_ticker):
    """Test getting annual and quarterly cash flow statements."""
    # Test annual
    annual_data = fetcher.get_cash_flow("MOCK", quarterly=False)
    pd.testing.assert_frame_equal(annual_data, MOCK_FINANCIALS_DATA)
    mock_yfinance_ticker.mock_cashflow.assert_called_once()

    # Test quarterly
    quarterly_data = fetcher.get_cash_flow("MOCK", quarterly=True)
    pd.testing.assert_frame_equal(quarterly_data, MOCK_QUARTERLY_FINANCIALS_DATA)
    mock_yfinance_ticker.mock_quarterly_cashflow.assert_called_once()


def test_get_dividends(fetcher, mock_yfinance_ticker):
    """Test getting dividend data."""
    dividends = fetcher.get_dividends("MOCK")
    pd.testing.assert_series_equal(dividends, MOCK_DIVIDENDS_DATA)
    mock_yfinance_ticker.mock_dividends.assert_called_once()


def test_get_news(fetcher, mock_yfinance_ticker):
    """Test getting news data."""
    news = fetcher.get_news("MOCK")
    assert news == MOCK_NEWS_DATA
    mock_yfinance_ticker.mock_news.assert_called_once()