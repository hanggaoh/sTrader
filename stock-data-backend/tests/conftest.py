import sys
import os

# Add the project root directory (/app) to the Python path.
# This ensures that pytest can find the 'src' module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.config import Config
from src.data.storage import Storage

@pytest.fixture(scope="session")
def db_connection():
    """
    Session-scoped fixture to set up the database connection and schema once.
    This is efficient as it avoids reconnecting for every test.
    """
    # Create a Config object from environment variables
    test_config = Config()

    # Instantiate the Storage class
    storage = Storage(config=test_config)

    # Set up the database schema (tables, hypertables, etc.)
    storage.setup_database()

    # Yield the configured storage object
    yield storage

    # Teardown: close the connection pool after all tests in the session are done
    storage.close()

@pytest.fixture(scope="function")
def db_storage(db_connection: Storage):
    """
    Function-scoped fixture that provides a clean database for each test.
    It depends on the session-scoped db_connection fixture and truncates
    tables before each test to ensure test isolation.
    """
    # Get a connection from the pool to truncate tables
    with db_connection.pool.connection() as conn:
        with conn.cursor() as cursor:
            # Truncate tables to ensure a clean state. RESTART IDENTITY resets
            # any auto-incrementing counters.
            cursor.execute("TRUNCATE TABLE stock_data, news_sentiment RESTART IDENTITY;")
    
    # Yield the storage object (which is the db_connection) for the test to use
    yield db_connection
