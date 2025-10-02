import numpy as np
import pandas as pd
import pytest

from data.storage import Storage
from ml.config import Config
from ml.preprocessor import build_features


# Ignore the UserWarning from pandas when using a raw DBAPI2 connection.
@pytest.mark.filterwarnings("ignore:pandas only supports SQLAlchemy connectable")
def test_build_and_store_features(db_storage: Storage):
    """
    An integration test for the feature engineering and storage pipeline.

    It verifies that:
    1. Raw data can be stored.
    2. Raw data can be fetched.
    3. Features can be calculated from the raw data.
    4. The calculated features can be stored in the database.
    5. The stored features can be fetched and match the calculated ones.
    """
    # 1. Setup: Insert some sample raw data into the database.
    symbol = "TEST_FEAT"
    # We need at least 50 data points for the largest rolling window (sma_50).
    # Let's use 100 to be safe.
    data = {
        'open': np.linspace(100, 150, 100),
        'high': np.linspace(101, 151, 100),
        'low': np.linspace(99, 149, 100),
        'close': np.linspace(100, 150, 100),
        'volume': np.random.randint(1000, 5000, 100),
    }
    index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))
    raw_df = pd.DataFrame(data, index=index)
    db_storage.store_historical_data(symbol, raw_df)

    # 2. Action: Fetch raw data, build features, and store them.
    # Fetch the raw data back to simulate a real-world workflow.
    with db_storage.pool.connection() as conn:
        # Use pandas to read directly from the DB connection.
        fetched_raw_df = pd.read_sql(
            "SELECT time, stock_symbol, open, high, low, close, volume FROM stock_data WHERE stock_symbol = %s",
            conn,
            params=[symbol],
            index_col="time",
        )
    
    # The dataframe needs to be prepared for the build_features function.
    fetched_raw_df = fetched_raw_df.rename(columns={"stock_symbol": "symbol"})
    fetched_raw_df.index.name = "timestamp"
    fetched_raw_df = fetched_raw_df.reset_index()
    fetched_raw_df["timestamp"] = pd.to_datetime(fetched_raw_df["timestamp"], utc=True)
    
    # The build_features function expects a sentiment column.
    fetched_raw_df['sentiment'] = 0.0

    # Create a config for the feature builder.
    cfg = Config(horizon=1)

    # Build the features.
    features_df, _ = build_features(fetched_raw_df, cfg)

    # Store the newly calculated features.
    db_storage.store_features(features_df)

    # 3. Assert: Fetch the stored features and verify their correctness.
    with db_storage.pool.connection() as conn:
        stored_features_df = pd.read_sql(
            "SELECT * FROM stock_features WHERE stock_symbol = %s ORDER BY time ASC",
            conn,
            params=[symbol],
            index_col="time",
        )
    
    # The number of rows should match the output of build_features, which drops NaNs.
    assert not stored_features_df.empty
    assert len(stored_features_df) == len(features_df)

    # Prepare the original calculated features DataFrame for comparison.
    # 1. Rename columns to match the database schema.
    features_df_to_compare = features_df.rename(columns={"timestamp": "time", "symbol": "stock_symbol"})
    # 2. Set the index to be the time column.
    features_df_to_compare = features_df_to_compare.set_index("time")
    # 3. Select only the columns that are in the database table and ensure they are in the same order.
    features_df_to_compare = features_df_to_compare[stored_features_df.columns]

    # Use the pandas testing utility for a robust, element-wise comparison.
    pd.testing.assert_frame_equal(stored_features_df, features_df_to_compare)

def test_build_features_calculation_logic():
    """
    Tests the core calculation logic of the build_features function,
    focusing on the initial values of key indicators.
    """
    # 1. Setup: Create a controlled DataFrame.
    # We need enough data to get one valid row after all NaNs are dropped.
    # The largest rolling window is 50 (sma_50), so we need at least 50 rows.
    # Let's use 60 to be safe and have a few rows to check.
    close_prices = np.arange(100, 160)
    data = {
        'symbol': ['TEST'] * 60,
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=60, freq='D')),
        'close': close_prices,
        # Add other required columns with dummy data
        'open': close_prices,
        'high': close_prices,
        'low': close_prices,
        'volume': [1000] * 60,
        'sentiment': [0.0] * 60,
    }
    input_df = pd.DataFrame(data)

    # Create a config for the feature builder.
    cfg = Config(horizon=1)

    # 2. Action: Build the features.
    features_df, _ = build_features(input_df.copy(), cfg)

    # 3. Assert: Check the first valid row of calculated features.
    # The first valid row will be at index 49 because of sma_50.
    first_valid_row = features_df.iloc[0]

    # --- Manually calculate the expected values for the first valid row ---
    # The first valid row corresponds to the original data at index 49.
    raw_data_for_calc = input_df.iloc[:50]

    # Expected SMA_5 for the 50th day (index 49)
    expected_sma_5 = raw_data_for_calc['close'].iloc[45:50].mean()

    # Expected EMA_12 for the 50th day (index 49)
    expected_ema_12 = raw_data_for_calc['close'].ewm(span=12, adjust=False).mean().iloc[49]

    # Expected EMA_26 for the 50th day (index 49)
    expected_ema_26 = raw_data_for_calc['close'].ewm(span=26, adjust=False).mean().iloc[49]
    
    # Expected MACD
    expected_macd = expected_ema_12 - expected_ema_26

    # --- Assertions ---
    # Use pytest.approx for floating point comparisons.
    assert first_valid_row['sma_5'] == pytest.approx(expected_sma_5)
    assert first_valid_row['macd'] == pytest.approx(expected_macd)
    assert first_valid_row['ema_20'] == pytest.approx(raw_data_for_calc['close'].ewm(span=20, adjust=False).mean().iloc[49])
