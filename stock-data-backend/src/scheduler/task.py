import json
import logging
import os
import random
import time

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler

from config import config
from data.fetcher import StockDataFetcher
from data.storage import Storage

log = logging.getLogger(__name__)

class TaskScheduler:
    def __init__(self):
        executors = {
            # A dedicated executor for the time-sensitive daily cron job to ensure it's never blocked.
            'cron_executor': ThreadPoolExecutor(max_workers=1),
            # The default pool for on-demand API requests.
            'default': ThreadPoolExecutor(max_workers=2),
            # A dedicated executor for the very long-running backfill task.
            # This prevents backfills from blocking the daily jobs.
            'backfill_executor': ThreadPoolExecutor(max_workers=1),
            # A dedicated, single-threaded executor for repair jobs to prevent rate-limiting.
            'repair_executor': ThreadPoolExecutor(max_workers=1)
        }

        job_defaults = {
            # If a job was supposed to run while the scheduler was down,
            # run it if the scheduler starts back up within 1 hour.
            'misfire_grace_time': 3600
        }
        self.scheduler = BackgroundScheduler(executors=executors, job_defaults=job_defaults)
        self.fetcher = StockDataFetcher()
        self.storage = Storage(config=config)

    def _get_stock_symbols(self) -> list[str]:
        """Helper method to load stock symbols from the JSON file."""
        json_path = os.path.join(os.path.dirname(__file__), '../data/chinese_stocks.json')
        try:
            with open(json_path, 'r') as f:
                symbols = json.load(f)
                if not isinstance(symbols, list):
                    log.error(f"Stock symbols file at {json_path} is not a valid JSON list.")
                    return []
                return symbols
        except FileNotFoundError:
            log.error(f"Stock symbols file not found at {json_path}")
        except json.JSONDecodeError:
            log.error(f"Could not decode JSON from stock symbols file at {json_path}. Is it formatted correctly?")
        return []

    def schedule_daily_tasks(self):
        stock_symbols = self._get_stock_symbols()
        if not stock_symbols:
            log.warning("No symbols found, skipping daily task scheduling.")
            return

        # Schedule a single master job to run daily.
        # This avoids overwhelming the scheduler with thousands of individual cron jobs.
        self.scheduler.add_job(
            self._run_daily_fetch_loop,
            'cron',
            hour=4,
            minute=0,
            # Pass a robust number of days (e.g., 5) for the scheduled cron job.
            # This ensures it can catch up even after long weekends or holidays.
            args=[stock_symbols, 5],
            id="master_daily_job",
            replace_existing=True,
            executor='cron_executor'  # Assign this job to its dedicated executor
        )
        log.info(f"Scheduled a single daily job for {len(stock_symbols)} stocks.")

    def run_daily_fetch_now(self, days=5):
        """
        Schedules a one-time, immediate job to fetch the latest daily data.
        This is useful for running on application startup to ensure data is
        fresh without waiting for the scheduled cron time.
        """
        stock_symbols = self._get_stock_symbols()
        if not stock_symbols:
            log.warning("No symbols found, skipping immediate daily fetch.")
            return 0

        log.info("Scheduling an immediate one-time run of the daily fetch job.")
        self.scheduler.add_job(
            self._run_daily_fetch_loop, args=[stock_symbols, days], id="immediate_daily_fetch", replace_existing=True
        )
        return len(stock_symbols)

    def fetch_single_stock_now(self, symbol: str, days: int = 5):
        """
        Schedules a one-time, immediate job to fetch the latest daily data
        for a single stock symbol.

        Args:
            symbol (str): The stock symbol to fetch.
            days (int): The number of recent days to fetch data for.
        """
        log.info(f"Scheduling an immediate one-time fetch for {symbol} for the last {days} days.")
        period_str = f"{days}d"
        # Use a unique job ID to allow multiple single-stock fetches to be scheduled.
        job_id = f"fetch_single_{symbol}_{int(time.time())}"
        self.scheduler.add_job(
            self._fetch_and_store,
            args=[symbol, period_str, "single-fetch"],
            id=job_id,
            replace_existing=False,  # Allow multiple ad-hoc fetches
            executor='repair_executor'  # Assign to the single-threaded executor
        )

    def schedule_backfill_tasks(self):
        """Schedules a single, one-time job to run the entire backfill process."""
        stock_symbols = self._get_stock_symbols()
        if not stock_symbols:
            return 0

        # Schedule a single job that will iterate through all stocks.
        # This avoids overwhelming the scheduler with thousands of queued jobs.
        self.scheduler.add_job(
            self._run_full_backfill_loop,
            args=[stock_symbols],
            id="master_backfill_job",
            replace_existing=True,
            executor='backfill_executor'  # Assign this job to the dedicated executor
        )
        log.info(f"Scheduled a single backfill job for {len(stock_symbols)} stocks.")
        return len(stock_symbols)

    def _run_full_backfill_loop(self, stock_symbols: list[str]):
        """
        The actual backfill process, run as a single long-running job.
        It iterates through all symbols and fetches their history, but only for
        symbols that are not already in the database.
        """
        total_symbols = len(stock_symbols)
        new_symbols_processed = 0
        log.info(f"Starting master backfill job for {total_symbols} symbols.")
        for i, symbol in enumerate(stock_symbols):
            # Log progress periodically to show the job is still running.
            if (i + 1) % 100 == 0:
                log.info(f"Backfill progress: {i + 1}/{total_symbols} symbols checked.")

            # QA Query: Check if the stock already exists in the database.
            # This prevents re-fetching the entire history for existing stocks.
            if self.storage.symbol_exists(symbol):
                log.debug(f"Skipping backfill for {symbol}; data already exists.")
                continue

            # If the symbol is new, fetch its full history.
            log.info(f"New symbol found: {symbol}. Starting full history backfill.")
            self._fetch_and_store(symbol, period="max", job_type="backfill")
            new_symbols_processed += 1
        
        log.info(f"Master backfill job completed. Checked {total_symbols} symbols, processed {new_symbols_processed} new symbols.")

    def _run_daily_fetch_loop(self, stock_symbols: list[str], days: int):
        """
        The actual daily fetch process, run as a single long-running job.
        It iterates through all symbols and fetches their recent history.

        Args:
            stock_symbols (list[str]): List of stock symbols to process.
            days (int): The number of recent days to fetch.
        """
        period_str = f"{days}d"
        total_symbols = len(stock_symbols)
        log.info(f"Starting master daily fetch job for {total_symbols} symbols, fetching last {period_str}.")
        for i, symbol in enumerate(stock_symbols):
            # Log progress periodically to show the job is still running.
            if (i + 1) % 100 == 0:
                log.info(f"Daily fetch progress: {i + 1}/{total_symbols} symbols processed.")
            
            self._fetch_and_store(symbol, period=period_str, job_type="daily")
        
        log.info(f"Master daily fetch job completed for all {total_symbols} symbols.")

    def _fetch_and_store(self, symbol: str, period: str, job_type: str):
        """Private helper to fetch, store, and log stock data."""
        log.info(f"Running {job_type} job for symbol: {symbol}")

        max_retries = 3
        backoff_factor = 5  # Start with a 5-second delay

        for attempt in range(max_retries):
            try:
                data = self.fetcher.fetch_historical_data(symbol, period=period)
                if not data.empty:
                    self.storage.store_historical_data(symbol, data)
                    log.info(f"Successfully stored {len(data)} records for {symbol} ({job_type}).")
                else:
                    log.warning(f"No data fetched for {symbol} ({job_type}). It might be a holiday or an invalid symbol.")

                # Success, so we add the politeness delay and exit the function.
                time.sleep(random.uniform(2.0, 3.5))
                return  # Exit successfully

            except Exception as e:
                log.warning(f"Attempt {attempt + 1}/{max_retries} failed for {symbol} with error: {e}")
                if attempt + 1 < max_retries:
                    # Exponential backoff: wait 5s, then 10s, etc.
                    sleep_time = backoff_factor * (attempt + 1)
                    log.info(f"Will retry fetching {symbol} in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    log.error(f"All {max_retries} retries failed for {symbol}. Giving up.", exc_info=True)

        # This part is reached only if all retries fail. We still add the politeness delay.
        time.sleep(random.uniform(2.0, 3.5))

    def start(self):
        """Starts the scheduler if it is not already running."""
        if not self.scheduler.running:
            self.scheduler.start()
            log.info("Scheduler started.")

    def stop(self):
        self.scheduler.shutdown()
        self.storage.close()