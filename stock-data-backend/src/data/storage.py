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
            self.pool = ConnectionPool(conninfo, min_size=1, max_size=5, open=True)
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
            content TEXT,
            sentiment_score DOUBLE PRECISION DEFAULT NULL,
            status VARCHAR(20) DEFAULT 'PENDING',
            CONSTRAINT news_sentiment_pk PRIMARY KEY (published_at, stock_symbol, id),
            UNIQUE (published_at, stock_symbol, headline)
        );
        """
        create_sentiment_hypertable_query = "SELECT create_hypertable('news_sentiment', 'published_at', if_not_exists => TRUE);"

        create_features_table_query = """
        CREATE TABLE IF NOT EXISTS stock_features (
            time TIMESTAMPTZ NOT NULL,
            stock_symbol VARCHAR(20) NOT NULL,
            sentiment DOUBLE PRECISION,
            sma_5 DOUBLE PRECISION, sma_10 DOUBLE PRECISION, sma_20 DOUBLE PRECISION, sma_50 DOUBLE PRECISION,
            ema_20 DOUBLE PRECISION, macd DOUBLE PRECISION, macd_signal DOUBLE PRECISION, macd_hist DOUBLE PRECISION,
            adx DOUBLE PRECISION, rsi DOUBLE PRECISION,
            return_1d DOUBLE PRECISION, return_5d DOUBLE PRECISION, return_21d DOUBLE PRECISION,
            volatility_21d DOUBLE PRECISION, atr DOUBLE PRECISION,
            skew_21d DOUBLE PRECISION, kurt_21d DOUBLE PRECISION,
            target INTEGER,
            PRIMARY KEY (time, stock_symbol)
        );
        """
        create_features_hypertable_query = "SELECT create_hypertable('stock_features', 'time', if_not_exists => TRUE);"
        
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                log.info("Creating and configuring 'stock_data' table...")
                cursor.execute(create_stock_table_query)
                cursor.execute(create_stock_hypertable_query)
                
                log.info("Creating and configuring 'news_sentiment' table...")
                cursor.execute(create_sentiment_table_query)
                cursor.execute(create_sentiment_hypertable_query)

                log.info("Creating and configuring 'stock_features' table...")
                cursor.execute(create_features_table_query)
                cursor.execute(create_features_hypertable_query)

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

    def get_stock_data_date_ranges(self) -> list[tuple]:
        """Fetches the min and max date for each symbol in the stock_data table."""
        log.info("Querying database for stock data date ranges...")
        query = "SELECT stock_symbol, MIN(time)::date, MAX(time)::date FROM stock_data GROUP BY stock_symbol;"
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                log.info(f"Found {len(results)} symbols with date ranges.")
                return results

    def get_news_summary_by_symbol(self) -> list[tuple]:
        """
        Fetches a summary of news data, grouped by stock symbol, including count and date range.

        Returns:
            A list of tuples, where each tuple contains:
            (stock_symbol, news_count, min_date, max_date)
        """
        log.info("Querying database for news summary by symbol...")
        query = """
            SELECT
                stock_symbol,
                COUNT(*) AS news_count,
                MIN(published_at)::date AS min_date,
                MAX(published_at)::date AS max_date
            FROM
                news_sentiment
            GROUP BY
                stock_symbol
            ORDER BY
                news_count DESC;
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchall()

    def store_raw_news(self, articles: list):
        """Inserts or updates raw news articles, ignoring duplicates based on a composite key."""
        if not articles:
            return 0
        
        sql = """
            INSERT INTO news_sentiment (published_at, stock_symbol, headline, source_url, content)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (published_at, stock_symbol, headline) DO UPDATE SET
                source_url = COALESCE(EXCLUDED.source_url, news_sentiment.source_url),
                content = COALESCE(EXCLUDED.content, news_sentiment.content);
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

    def store_features(self, features_df: pd.DataFrame):
        """Bulk-inserts or updates features into the stock_features table."""
        if features_df.empty:
            return

        cols = [
            'time', 'stock_symbol', 'sentiment', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_20', 
            'macd', 'macd_signal', 'macd_hist', 'adx', 'rsi', 'return_1d', 'return_5d', 'return_21d', 
            'volatility_21d', 'atr', 'skew_21d', 'kurt_21d', 'target'
        ]
        df_to_copy = features_df.rename(columns={"timestamp": "time", "symbol": "stock_symbol"})
        df_to_copy = df_to_copy[cols]

        buffer = io.StringIO()
        df_to_copy.to_csv(buffer, index=False, header=False)
        buffer.seek(0)

        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("CREATE TEMP TABLE temp_features (LIKE stock_features) ON COMMIT DROP;")
                with cursor.copy(f"COPY temp_features ({','.join(cols)}) FROM STDIN WITH (FORMAT CSV)") as copy:
                    copy.write(buffer.read())
                
                cursor.execute("""
                    INSERT INTO stock_features SELECT * FROM temp_features
                    ON CONFLICT (time, stock_symbol) DO UPDATE SET
                        sentiment = EXCLUDED.sentiment, sma_5 = EXCLUDED.sma_5, sma_10 = EXCLUDED.sma_10, 
                        sma_20 = EXCLUDED.sma_20, sma_50 = EXCLUDED.sma_50, ema_20 = EXCLUDED.ema_20, 
                        macd = EXCLUDED.macd, macd_signal = EXCLUDED.macd_signal, macd_hist = EXCLUDED.macd_hist, 
                        adx = EXCLUDED.adx, rsi = EXCLUDED.rsi, return_1d = EXCLUDED.return_1d, 
                        return_5d = EXCLUDED.return_5d, return_21d = EXCLUDED.return_21d, 
                        volatility_21d = EXCLUDED.volatility_21d, atr = EXCLUDED.atr, 
                        skew_21d = EXCLUDED.skew_21d, kurt_21d = EXCLUDED.kurt_21d, target = EXCLUDED.target;
                """)

    def symbol_exists(self, stock_symbol: str) -> bool:
        query = "SELECT 1 FROM stock_data WHERE stock_symbol = %s LIMIT 1;"
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (stock_symbol,))
                return cursor.fetchone() is not None

    def has_news_for_symbol(self, stock_symbol: str) -> bool:
        """Checks if any news articles exist for a given stock symbol."""
        query = "SELECT 1 FROM news_sentiment WHERE stock_symbol = %s LIMIT 1;"
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (stock_symbol,))
                return cursor.fetchone() is not None

    def query_features_for_symbols(self, stock_symbols: list[str]) -> pd.DataFrame:
        """
        Queries the database for all features for a given list of stock symbols.
        """
        if not stock_symbols:
            return pd.DataFrame()

        log.info(f"Querying features for {len(stock_symbols)} symbols...")
        query = "SELECT * FROM stock_features WHERE stock_symbol = ANY(%s);"
        
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (stock_symbols,))
                
                column_names = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=column_names)
                
                log.info(f"Retrieved {len(df)} feature rows for {len(stock_symbols)} symbols.")
                return df

    def close(self):
        if self.pool:
            self.pool.close()
            log.info("Database connection pool closed.")
