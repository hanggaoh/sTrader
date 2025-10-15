import logging
from datetime import date, timedelta

from .base_task import ScheduledTask
from ml.preprocessor import calculate_and_store_features

log = logging.getLogger(__name__)


class FeatureCalculationTask(ScheduledTask):
    def run(self):
        log.info("Starting daily feature calculation job.")
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=90)
            calculate_and_store_features(self.storage, start_date.isoformat(), end_date.isoformat())
        except Exception as e:
            log.error(f"An error occurred during the daily feature calculation job: {e}", exc_info=True)
        log.info("Daily feature calculation job finished.")
