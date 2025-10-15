import logging
import os

import pytest

from config import Config
from data.storage import Storage
from app import create_app

log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def db_connection():
    """
    Session-scoped fixture to set up the connection to a dedicated test database.
    It assumes the database and user have already been created by init-db.d scripts.
    """
    # Get the config for the test database from environment variables.
    # This will correctly point to 'strader_test_db' as defined in docker-compose.yml.
    test_config = Config()

    # Connect to the test database and set up the schema (tables, hypertables, etc.).
    # This is idempotent and safe to run on an existing database.
    storage = Storage(config=test_config)
    storage.setup_database()

    yield storage

    # Teardown: close the connection pool after all tests are done.
    storage.close()


@pytest.fixture(scope="function")
def db_storage(db_connection: Storage):
    """
    Function-scoped fixture that provides a clean database for each test.
    It truncates all tables in the test database before each test run.
    """
    with db_connection.pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE stock_data, news_sentiment, stock_features RESTART IDENTITY;")
    
    yield db_connection

@pytest.fixture
def app():
    flask_app = create_app(init_scheduler=False)
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()
