import pandas as pd

default_api = None

def test_get_all_distinct_symbols(db_storage):
    """
    Tests that the get_all_distinct_symbols method correctly returns unique symbols
    from the database.
    """
    # 1. Setup: Insert some sample data into the test database
    # Create a sample DataFrame
    data = {
        'Open': [100, 102],
        'High': [105, 103],
        'Low': [99, 101],
        'Close': [104, 101.5],
        'Volume': [10000, 12000]
    }
    index = pd.to_datetime(['2023-01-01', '2023-01-02'])
    sample_df = pd.DataFrame(data, index=index)

    # Insert data for two different symbols
    db_storage.store_historical_data("TEST.A", sample_df)
    db_storage.store_historical_data("TEST.B", sample_df)

    # 2. Action: Call the method we want to test
    distinct_symbols = db_storage.get_all_distinct_symbols()

    # 3. Assert: Check that the results are what we expect
    # The order isn't guaranteed, so we use a set for comparison
    assert set(distinct_symbols) == {"TEST.A", "TEST.B"}
