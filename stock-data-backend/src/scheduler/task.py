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
            # Limit the number of concurrent jobs to avoid overwhelming the API.
            # A lower number is safer.
            'default': ThreadPoolExecutor(max_workers=2)
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
                return json.load(f)
        except FileNotFoundError:
            log.error(f"Stock symbols file not found at {json_path}")
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
            args=[stock_symbols],
            id="master_daily_job",
            replace_existing=True
        )
        log.info(f"Scheduled a single daily job for {len(stock_symbols)} stocks.")

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
            replace_existing=True
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

    def _run_daily_fetch_loop(self, stock_symbols: list[str]):
        """
        The actual daily fetch process, run as a single long-running job.
        It iterates through all symbols and fetches their recent history.
        """
        total_symbols = len(stock_symbols)
        log.info(f"Starting master daily fetch job for {total_symbols} symbols.")
        for i, symbol in enumerate(stock_symbols):
            # Log progress periodically to show the job is still running.
            if (i + 1) % 100 == 0:
                log.info(f"Daily fetch progress: {i + 1}/{total_symbols} symbols processed.")
            
            self._fetch_and_store(symbol, period="5d", job_type="daily")
        
        log.info(f"Master daily fetch job completed for all {total_symbols} symbols.")

    def _fetch_and_store(self, symbol: str, period: str, job_type: str):
        """Private helper to fetch, store, and log stock data."""
        log.info(f"Running {job_type} job for symbol: {symbol}")
        try:
            data = self.fetcher.fetch_historical_data(symbol, period=period)
            if not data.empty:
                self.storage.store_historical_data(symbol, data)
                log.info(f"Successfully stored {len(data)} records for {symbol} ({job_type}).")
            else:
                log.warning(f"No data fetched for {symbol} ({job_type}). It might be a holiday or an invalid symbol.")
        except Exception as e:
            # exc_info=True will log the full stack trace for better debugging
            log.error(f"An error occurred during {job_type} for {symbol}: {e}", exc_info=True)
        finally:
            # Add a small, randomized delay to be polite to the API provider.
            time.sleep(random.uniform(0.5, 1.5))

    def start(self):
        """Starts the scheduler if it is not already running."""
        if not self.scheduler.running:
            self.scheduler.start()
            log.info("Scheduler started.")

    def stop(self):
        self.scheduler.shutdown()
        self.storage.close()