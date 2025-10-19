import logging
import pandas as pd
from config import config
from data.storage import Storage

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("inspect_features")

def inspect_feature_table():
    """Connects to the database and prints the last few rows of the stock_features table."""
    storage = None
    try:
        storage = Storage(config)
        log.info("Querying the last 10 rows from the 'stock_features' table...")
        with storage.pool.connection() as conn:
            # Use pandas to easily display the results
            features_df = pd.read_sql("SELECT * FROM stock_features ORDER BY time DESC LIMIT 10", conn)

        if not features_df.empty:
            log.info(f"Found {len(features_df)} rows. Displaying table contents:")
            # Configure pandas to display all columns
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(features_df)
        else:
            log.warning("The 'stock_features' table is completely empty.")

    except Exception as e:
        # Catching a general exception in case the table doesn't exist yet
        if "relation \"stock_features\" does not exist" in str(e):
            log.error("The 'stock_features' table does not seem to exist. Please run the database setup.")
        else:
            log.error("An error occurred while inspecting the feature table.", exc_info=True)
    finally:
        if storage:
            storage.close()

if __name__ == "__main__":
    inspect_feature_table()
