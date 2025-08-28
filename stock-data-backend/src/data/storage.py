import io
import logging

import pandas as pd
import psycopg
from psycopg_pool import ConnectionPool

from config import Config

log = logging.getLogger(__name__)

class Storage:
    """
    Manages database interactions with a TimescaleDB/PostgreSQL database.
    Handles connection, schema setup, and data storage/retrieval using a thread-safe connection pool.
    """

    def __init__(self, config: Config):
        """
        Initializes a thread-safe database connection pool.
        """
        conninfo = (
            f"dbname={config.db_name} user={config.db_user} "
            f"password={config.db_password} host={config.db_host} port={config.db_port}"
        )
        try:
            # Create a connection pool. min_size=1 to keep at least one connection open.
            # max_size should be >= number of concurrent workers in the scheduler.
            self.pool = ConnectionPool(conninfo, min_size=1, max_size=5)
        except psycopg.OperationalError as e:
            log.error(f"Could not connect to the database. Please check connection settings.", exc_info=True)
            raise e

    def setup_database(self):
        """
        Creates the necessary table for stock data and converts it into a
        TimescaleDB hypertable for efficient time-series data handling.
        """
        log.info("Setting up database schema...")
        create_table_query = """
        CREATE TABLE IF NOT EXISTS stock_data (
            time TIMESTAMPTZ NOT NULL,
            stock_symbol VARCHAR(20) NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume BIGINT,
            PRIMARY KEY (time, stock_symbol)
        );
        """
        
        create_hypertable_query = """
        SELECT create_hypertable('stock_data', 'time', if_not_exists => TRUE);
        """
        
        # Use a connection from the pool for this operation.
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                # The TimescaleDB extension needs to be enabled for create_hypertable to work.
                # The official Docker image has it enabled by default.
                cursor.execute(create_hypertable_query)
        log.info("Schema setup complete.")

    def store_historical_data(self, stock_symbol: str, data: pd.DataFrame):
        """
        Stores historical stock data from a pandas DataFrame into the database.
        It performs an "upsert" operation: inserting new rows and updating
        existing ones based on the (time, stock_symbol) primary key.

        Args:
            stock_symbol (str): The stock ticker symbol.
            data (pd.DataFrame): DataFrame with historical data. Must have a
                                 DatetimeIndex and columns for Open, High,
                                 Low, Close, and Volume.
        """
        if data.empty:
            return

        # Prepare data for bulk copy. This is significantly faster than row-by-row inserts.
        data_to_copy = data.copy()
        data_to_copy['stock_symbol'] = stock_symbol
        # Ensure column order matches the temporary table for COPY
        data_to_copy = data_to_copy[['stock_symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Use an in-memory buffer to stage the data as a CSV
        buffer = io.StringIO()
        data_to_copy.to_csv(buffer, index=True, header=False)
        buffer.seek(0)

        # Use a dedicated connection from the pool for this transaction.
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                # 1. Create a temporary table that will be automatically dropped at the end of the transaction.
                cursor.execute("""
                    CREATE TEMP TABLE temp_stock_data (
                        time TIMESTAMPTZ NOT NULL,
                        stock_symbol VARCHAR(20) NOT NULL,
                        open DOUBLE PRECISION,
                        high DOUBLE PRECISION,
                        low DOUBLE PRECISION,
                        close DOUBLE PRECISION,
                        volume BIGINT
                    ) ON COMMIT DROP;
                """)

                # 2. Use the high-performance COPY command to load data into the temp table.
                with cursor.copy("COPY temp_stock_data (time, stock_symbol, open, high, low, close, volume) FROM STDIN WITH (FORMAT CSV)") as copy:
                    copy.write(buffer.read())

                # 3. Merge data from the temp table into the main table, updating on conflict.
                # This is the standard "bulk upsert" pattern.
                cursor.execute("""
                    INSERT INTO stock_data (time, stock_symbol, open, high, low, close, volume)
                    SELECT time, stock_symbol, open, high, low, close, volume FROM temp_stock_data
                    ON CONFLICT (time, stock_symbol) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume;
                """)

    def read_historical_data(self, stock_symbol: str) -> pd.DataFrame:
        """
        Reads historical data for a given stock symbol from the database.
        """
        with self.pool.connection() as conn:
            # This method is included for completeness but not used by the scheduler.
            query = "SELECT * FROM stock_data WHERE stock_symbol = %s ORDER BY time;"
            df = pd.read_sql(query, conn, params=(stock_symbol,), index_col='time')
        return df

    def close(self):
        """Closes the database connection pool."""
        if self.pool:
            self.pool.close()
            log.info("Database connection pool closed.")