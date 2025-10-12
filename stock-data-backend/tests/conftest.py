import logging
import os

import psycopg
import pytest

from config import Config
from data.storage import Storage
from app import app as flask_app

log = logging.getLogger(__name__)


def _create_test_database_if_not_exists(test_db_name: str):
    """
    Connects to the default maintenance database and creates the test database if it doesn't exist.
    """
    # For the maintenance connection, we connect to the default 'strader_db' database.
    # We build the connection string manually since we can't override the Config object easily.
    conn_info = (
        f"dbname=strader_db "
        f"user={os.getenv('DB_USER')} "
        f"password={os.getenv('DB_PASSWORD')} "
        f"host={os.getenv('DB_HOST')} "
        f"port={os.getenv('DB_PORT')}"
    )
    
    conn = None
    try:
        # Use a raw psycopg connection because CREATE DATABASE cannot be run inside a transaction.
        conn = psycopg.connect(conn_info, autocommit=True)
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (test_db_name,))
            if cursor.fetchone():
                log.info(f"Test database '{test_db_name}' already exists.")
            else:
                log.info(f"Test database '{test_db_name}' not found. Creating it...")
                cursor.execute(f'CREATE DATABASE "{test_db_name}"')
                log.info(f"Test database '{test_db_name}' created successfully.")
    except Exception as e:
        log.error(f"Failed to create test database '{test_db_name}': {e}", exc_info=True)
        raise
    finally:
        if conn:
            conn.close()


@pytest.fixture(scope="session")
def db_connection():
    """
    Session-scoped fixture to set up the connection to a dedicated test database.
    """
    # Get the config for the test database from environment variables.
    # This will correctly point to 'strader_test_db' as defined in docker-compose.yml.
    test_config = Config()
    
    # Ensure the test database exists before trying to connect to it.
    _create_test_database_if_not_exists(test_config.db_name)

    # Now, connect to the test database and set up the schema.
    storage = Storage(config=test_config)

    # Drop tables to ensure a fresh schema for each test session.
    with storage.pool.connection() as conn:
        with conn.cursor() as cursor:
            log.info("Dropping existing tables for a clean test session...")
            cursor.execute("DROP TABLE IF EXISTS stock_features;")
            cursor.execute("DROP TABLE IF EXISTS news_sentiment;")
            cursor.execute("DROP TABLE IF EXISTS stock_data;")

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
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()
