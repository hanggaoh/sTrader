import logging
import random
import time
from typing import List

from apscheduler.schedulers.background import BackgroundScheduler

from scheduler.tasks.base_task import ScheduledTask
from data.fetcher import StockDataFetcher
from data.storage import Storage
from scheduler.tasks.feature_calculation import FeatureCalculationTask

log = logging.getLogger(__name__)


class PriceFetchTask(ScheduledTask):
    def __init__(self, scheduler: BackgroundScheduler, storage: Storage, fetcher: StockDataFetcher, stock_symbols: List[str], days: int = 5):
        super().__init__(scheduler, storage)
        self.fetcher = fetcher
        self.stock_symbols = stock_symbols
        self.days = days

    def run(self):
        period_str = f"{self.days}d"
        total_symbols = len(self.stock_symbols)
        log.info(f"Starting master daily price fetch job for {total_symbols} symbols, fetching last {period_str}.")
        for i, symbol in enumerate(self.stock_symbols):
            if (i + 1) % 100 == 0:
                log.info(f"Daily price fetch progress: {i + 1}/{total_symbols} symbols processed.")
            self._fetch_and_store_price(symbol, period=period_str, job_type="daily")
        log.info(f"Master daily price fetch job completed for all {total_symbols} symbols.")

        # Now, trigger the feature calculation job.
        log.info("Daily fetch finished, triggering feature calculation.")
        feature_task = FeatureCalculationTask(self.scheduler, self.storage)
        self.scheduler.add_job(feature_task.run, id="immediate_feature_calculation", replace_existing=True, executor='cron_executor')

    def _fetch_and_store_price(self, symbol: str, period: str, job_type: str):
        log.info(f"Running {job_type} job for symbol: {symbol}")
        max_retries = 3
        backoff_factor = 5
        for attempt in range(max_retries):
            try:
                data = self.fetcher.fetch_historical_data(symbol, period=period)
                if not data.empty:
                    self.storage.store_historical_data(symbol, data)
                    log.info(f"Successfully stored {len(data)} records for {symbol} ({job_type}).")
                else:
                    log.warning(f"No data fetched for {symbol} ({job_type}).")
                time.sleep(random.uniform(2.0, 3.5))
                return
            except Exception as e:
                log.warning(f"Attempt {attempt + 1}/{max_retries} failed for {symbol} with error: {e}")
                if attempt + 1 < max_retries:
                    sleep_time = backoff_factor * (attempt + 1)
                    log.info(f"Will retry fetching {symbol} in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    log.error(f"All {max_retries} retries failed for {symbol}. Giving up.", exc_info=True)
        time.sleep(random.uniform(2.0, 3.5))