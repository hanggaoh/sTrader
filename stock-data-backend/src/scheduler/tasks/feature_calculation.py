import logging
from datetime import date, timedelta
from typing import Optional

from config import config
from data.storage import Storage
from ml.preprocessor import calculate_and_store_features

log = logging.getLogger(__name__)


class FeatureCalculationTask:
    def __init__(self, storage: Optional[Storage] = None):
        self.storage = storage

    def run(self):
        log.info("Starting daily feature calculation job.")
        
        storage_instance = self.storage
        should_close_storage = False
        if storage_instance is None:
            log.info("No storage instance provided, creating a new one.")
            storage_instance = Storage(config)
            should_close_storage = True

        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=90)
            calculate_and_store_features(storage_instance, start_date.isoformat(), end_date.isoformat())
        except Exception as e:
            log.error(f"An error occurred during the daily feature calculation job: {e}", exc_info=True)
        finally:
            if should_close_storage and storage_instance:
                storage_instance.close()
        log.info("Daily feature calculation job finished.")
