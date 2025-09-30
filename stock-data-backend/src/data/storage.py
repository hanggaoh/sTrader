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
    """

    def __init__(self, config: Config):
        conninfo = (
            f"dbname={config.db_name} user={config.db_user} "
            f"password={config.db_password} host={config.db_host} port={config.db_port}"
        )
        try:
            self.pool = ConnectionPool(conninfo, min_size=1, max_size=5)
        except psycopg.OperationalError as e:
            log.error(f"Could not connect to the database. Please check connection settings.", exc_info=True)
            raise e

    def setup_database(self):
        log.info("Setting up database schema...")
        create_stock_table_query = """
        CREATE TABLE IF NOT EXISTS stock_data (
            time TIMESTAMPTZ NOT NULL,
            stock_symbol VARCHAR(20) NOT NULL,
            open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION, close DOUBLE PRECISION, volume BIGINT,
            PRIMARY KEY (time, stock_symbol)
        );
        """
        create_stock_hypertable_query = "SELECT create_hypertable('stock_data', 'time', if_not_exists => TRUE);"

        create_sentiment_table_query = """
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id BIGSERIAL,
            published_at TIMESTAMPTZ NOT NULL,
            stock_symbol VARCHAR(20) NOT NULL,
            headline TEXT,
            source_url TEXT,
            sentiment_score DOUBLE PRECISION DEFAULT NULL,
            status VARCHAR(20) DEFAULT 'PENDING',
            CONSTRAINT news_sentiment_pk PRIMARY KEY (published_at, stock_symbol, id)
        );
        """
        create_sentiment_hypertable_query = "SELECT create_hypertable('news_sentiment', 'published_at', if_not_exists => TRUE);"
        
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                log.info("Creating and configuring 'stock_data' table...")
                cursor.execute(create_stock_table_query)
                cursor.execute(create_stock_hypertable_query)
                
                log.info("Creating and configuring 'news_sentiment' table...")
                cursor.execute(create_sentiment_table_query)
                cursor.execute(create_sentiment_hypertable_query)

        log.info("Schema setup complete.")

    def get_all_distinct_symbols(self) -> list[str]:
        """Fetches a list of all unique stock symbols from the database."""
        log.info("Querying database for all distinct stock symbols...")
        query = "SELECT DISTINCT stock_symbol FROM stock_data;"
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                symbols = [row[0] for row in results]
                log.info(f"Found {len(symbols)} distinct symbols in the database.")
                return symbols

    def store_raw_news(self, articles: list):
        """Inserts raw news articles with a 'PENDING' status, ignoring duplicates."""
        if not articles:
            return 0
        
        sql = """
            INSERT INTO news_sentiment (published_at, stock_symbol, headline, source_url)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (headline) DO NOTHING;
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, articles)
                return cursor.rowcount

    def get_pending_sentiment_articles(self, limit: int = 100) -> list:
        """Fetches articles that have not yet been processed."""
        sql = "SELECT id, headline FROM news_sentiment WHERE status = 'PENDING' LIMIT %s;"
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (limit,))
                return cursor.fetchall()

    def update_sentiment_scores(self, results: list):
        """Bulk updates articles with their calculated sentiment scores."""
        if not results:
            return
            
        sql = "UPDATE news_sentiment SET sentiment_score = %s, status = 'PROCESSED' WHERE id = %s;"
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, results)

    def store_historical_data(self, stock_symbol: str, data: pd.DataFrame):
        if data.empty:
            return
        data_to_copy = data.copy()
        data_to_copy.columns = [c.lower() for c in data_to_copy.columns]
        data_to_copy['stock_symbol'] = stock_symbol
        data_to_copy = data_to_copy[['stock_symbol', 'open', 'high', 'low', 'close', 'volume']]
        buffer = io.StringIO()
        data_to_copy.to_csv(buffer, index=True, header=False)
        buffer.seek(0)
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("CREATE TEMP TABLE temp_stock_data (LIKE stock_data) ON COMMIT DROP;")
                with cursor.copy("COPY temp_stock_data (time, stock_symbol, open, high, low, close, volume) FROM STDIN WITH (FORMAT CSV)") as copy:
                    copy.write(buffer.read())
                cursor.execute("""
                    INSERT INTO stock_data SELECT * FROM temp_stock_data
                    ON CONFLICT (time, stock_symbol) DO UPDATE SET
                        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                        close = EXCLUDED.close, volume = EXCLUDED.volume;
                """)

    def symbol_exists(self, stock_symbol: str) -> bool:
        query = "SELECT 1 FROM stock_data WHERE stock_symbol = %s LIMIT 1;"
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (stock_symbol,))
                return cursor.fetchone() is not None

    def close(self):
        if self.pool:
            self.pool.close()
            log.info("Database connection pool closed.")
