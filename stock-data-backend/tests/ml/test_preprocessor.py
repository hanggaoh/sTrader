import numpy as np
import pandas as pd
import pytest

from data.storage import Storage
from ml.config import Config
from ml.preprocessor import build_features, calculate_and_store_features


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

    # 4. Coerce the target dtype to match what the database returns (int64 instead of nullable Int64).
    features_df_to_compare['target'] = features_df_to_compare['target'].astype('int64')

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

@pytest.mark.filterwarnings("ignore:pandas only supports SQLAlchemy connectable")
def test_feature_calculation_with_unsorted_db_data(db_storage: Storage):
    """
    Tests that `calculate_and_store_features` correctly calculates features
    even when the underlying data in the database is not chronologically ordered.
    This specifically validates the `ORDER BY time` clause in the data fetching query.
    """
    # 1. Setup: Create and insert unsorted data
    symbol = "TEST_SORT"
    # We need enough data for feature calculation to produce at least one row after dropping NaNs.
    # The `build_features` function uses rolling windows up to 50, and a lookahead target.
    # So, we need at least 51 data points.
    num_records = 51
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_records, freq='D'))
    # Simple ascending close prices for easy manual calculation
    close_prices = np.arange(100, 100 + num_records)

    # Create a list of records and shuffle it before insertion
    records = []
    for i in range(num_records):
        records.append((dates[i], symbol, close_prices[i]))
    np.random.shuffle(records)

    # Insert the shuffled records into the database
    with db_storage.pool.connection() as conn:
        with conn.cursor() as cursor:
            for time, sym, close in records:
                cursor.execute(
                    "INSERT INTO stock_data (time, stock_symbol, open, high, low, close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (time, sym, close, close, close, close, 1000)
                )

    # 2. Action: Run the feature calculation function on the unsorted data.
    start_date = dates[0].strftime('%Y-%m-%d')
    end_date = dates[-1].strftime('%Y-%m-%d')
    calculate_and_store_features(db_storage, start_date, end_date, symbol=symbol)

    # 3. Assert: Fetch the results and verify the calculation.
    with db_storage.pool.connection() as conn:
        # After dropping NaNs, only one row should remain (for the 50th day, index 49)
        features_df = pd.read_sql("SELECT * FROM stock_features WHERE stock_symbol = %s", conn, params=[symbol])

    # There should be exactly one row of calculated features.
    assert len(features_df) == 1
    feature_row = features_df.iloc[0]

    # Manually calculate the expected SMA_5 for the 50th day (index 49).
    # This calculation is based on the *chronologically sorted* close prices.
    # Prices for days 46-50 (indices 45-49) are 145, 146, 147, 148, 149.
    expected_sma_5 = np.mean([145, 146, 147, 148, 149])

    # Check if the calculated SMA_5 from the database matches the expected value.
    # If it matches, it proves the data was sorted correctly before calculation.
    assert feature_row['sma_5'] == pytest.approx(expected_sma_5)
    print(f"\nValidation successful! SMA_5 in DB: {feature_row['sma_5']}, Expected: {expected_sma_5}")

def test_build_features_with_insufficient_data():
    """
    Tests that build_features returns an empty DataFrame when there is not
    enough data to compute the largest rolling window.
    """
    # 1. Setup: Create a DataFrame with fewer rows than the largest window (50).
    # Let's use 40 rows.
    num_records = 40
    close_prices = np.arange(100, 100 + num_records)
    data = {
        'symbol': ['TEST_INSUFFICIENT'] * num_records,
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_records, freq='D')),
        'close': close_prices,
        'open': close_prices,
        'high': close_prices,
        'low': close_prices,
        'volume': [1000] * num_records,
        'sentiment': [0.0] * num_records,
    }
    input_df = pd.DataFrame(data)

    # Create a config for the feature builder.
    cfg = Config(horizon=1)

    # 2. Action: Build the features.
    features_df, _ = build_features(input_df.copy(), cfg)

    # 3. Assert: The resulting DataFrame should be empty.
    # The dropna() call should remove all rows because sma_50 will be all NaN.
    assert features_df.empty

def test_build_features_mid_series_calculation():
    """
    Tests the calculation of features for a row in the middle of the dataset,
    ensuring the rolling windows are behaving correctly in a steady state.
    """
    # 1. Setup: Create a controlled DataFrame with 100 data points.
    num_records = 100
    close_prices = np.arange(100, 100 + num_records)
    data = {
        'symbol': ['TEST_MID'] * num_records,
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_records, freq='D')),
        'close': close_prices,
        'open': close_prices,
        'high': close_prices,
        'low': close_prices,
        'volume': [1000] * num_records,
        'sentiment': [0.0] * num_records,
    }
    input_df = pd.DataFrame(data)

    # Create a config for the feature builder.
    cfg = Config(horizon=1)

    # 2. Action: Build the features.
    features_df, _ = build_features(input_df.copy(), cfg)

    # 3. Assert: Check the calculated features for a row in the middle.
    # We will check the features for the 70th day (index 69).
    # The first valid row in features_df corresponds to index 49 of the input_df.
    # So, the 70th day (index 69) of input_df corresponds to index 20 of features_df.
    mid_series_row = features_df.iloc[20]

    # --- Manually calculate the expected values for the 70th day (index 69) ---
    raw_data_for_calc = input_df.iloc[:70]

    # Expected SMA_20 for the 70th day (index 69)
    expected_sma_20 = raw_data_for_calc['close'].iloc[50:70].mean()

    # Expected RSI for the 70th day (index 69)
    delta = raw_data_for_calc['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[69]
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[69]
    # In our simple case, loss is always 0, so RSI should be 100.
    expected_rsi = 100.0

    # --- Assertions ---
    assert mid_series_row['sma_20'] == pytest.approx(expected_sma_20)
    assert mid_series_row['rsi'] == pytest.approx(expected_rsi)

def test_build_features_with_gaps():
    """
    Tests that rolling window calculations correctly handle gaps in the data,
    such as weekends and holidays.
    """
    # 1. Setup: Create a DataFrame with a non-continuous business day index.
    num_records = 60
    close_prices = np.arange(100, 100 + num_records)
    data = {
        'symbol': ['TEST_GAPS'] * num_records,
        # Use a business day range, which will have gaps for weekends.
        'timestamp': pd.bdate_range(start='2023-01-01', periods=num_records),
        'close': close_prices,
        'open': close_prices,
        'high': close_prices,
        'low': close_prices,
        'volume': [1000] * num_records,
        'sentiment': [0.0] * num_records,
    }
    input_df = pd.DataFrame(data)

    cfg = Config(horizon=1)

    # 2. Action: Build the features.
    features_df, _ = build_features(input_df.copy(), cfg)

    # 3. Assert: Check a value immediately after a weekend gap.
    # The first valid row is at index 49. Let's check the row for index 55.
    # This corresponds to iloc[6] in the features_df.
    row_to_check = features_df.iloc[6]
    
    # Manually calculate the SMA_5 for the data point at original index 55.
    # This should be the mean of the close prices from index 51 to 55, inclusive.
    # The fact that there was a weekend between these dates should not affect the calculation.
    expected_sma_5 = input_df['close'].iloc[51:56].mean()

    assert row_to_check['sma_5'] == pytest.approx(expected_sma_5)

def test_build_features_with_multiple_symbols():
    """
    Tests that feature calculations are correctly isolated when processing
    a DataFrame with multiple stock symbols.
    """
    # 1. Setup: Create a DataFrame with two different symbols.
    num_records = 60
    # Stock A: steadily increasing price
    stock_a_prices = np.arange(100, 100 + num_records)
    # Stock B: steadily decreasing price
    stock_b_prices = np.arange(200, 200 - num_records, -1)

    data_a = {
        'symbol': ['STOCK_A'] * num_records,
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_records, freq='D')),
        'close': stock_a_prices,
        'open': stock_a_prices, 'high': stock_a_prices, 'low': stock_a_prices,
        'volume': [1000] * num_records, 'sentiment': [0.0] * num_records,
    }
    data_b = {
        'symbol': ['STOCK_B'] * num_records,
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_records, freq='D')),
        'close': stock_b_prices,
        'open': stock_b_prices, 'high': stock_b_prices, 'low': stock_b_prices,
        'volume': [2000] * num_records, 'sentiment': [0.0] * num_records,
    }

    input_df = pd.concat([pd.DataFrame(data_a), pd.DataFrame(data_b)]).sort_values('timestamp').reset_index(drop=True)

    cfg = Config(horizon=1)

    # 2. Action: Build the features on the combined DataFrame.
    features_df, _ = build_features(input_df.copy(), cfg)

    # 3. Assert: Check the calculations for each stock independently.
    # The first valid row is at index 49 for each stock.
    # In the final df, this corresponds to iloc[0] for STOCK_A and iloc[1] for STOCK_B
    # (because of the dropna and subsequent default indexing).
    
    # Separate the results for each stock
    features_a = features_df[features_df['symbol'] == 'STOCK_A']
    features_b = features_df[features_df['symbol'] == 'STOCK_B']

    # --- Manually calculate expected values for the first valid row of each stock ---
    # This corresponds to original index 49 for each stock.
    
    # For STOCK_A
    expected_sma_5_a = np.mean(stock_a_prices[45:50])
    # For STOCK_B
    expected_sma_5_b = np.mean(stock_b_prices[45:50])

    # --- Assertions ---
    assert features_a.iloc[0]['sma_5'] == pytest.approx(expected_sma_5_a)
    assert features_b.iloc[0]['sma_5'] == pytest.approx(expected_sma_5_b)
