import logging
from datetime import date

from config import Config
from data.storage import Storage
from ml.preprocessor import calculate_and_store_features

log = logging.getLogger(__name__)

def backfill_one_symbol_feature(symbol: str):
    """
    Helper function to backfill features for a single stock.
    This function is designed to be run in a separate thread.
    """
    log.info(f"Starting feature backfill for symbol: {symbol}")
    storage = None
    try:
        # Each thread must have its own Storage instance and its own Config.
        app_config = Config()
        storage = Storage(config=app_config)

        # Define a very wide date range to ensure all data is captured.
        start_date = "1970-01-01"
        end_date = date.today().isoformat()

        calculate_and_store_features(storage, start_date, end_date, symbol=symbol)
        log.info(f"Finished feature backfill for symbol: {symbol}")
        return True, symbol
    except Exception as e:
        log.error(f"Error backfilling features for {symbol}: {e}", exc_info=True)
        return False, symbol
    finally:
        if storage:
            storage.close()
