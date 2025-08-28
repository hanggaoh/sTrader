import json
import os
import random
import time
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor

import logger_config
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

        for symbol in stock_symbols:
            self.scheduler.add_job(
                self.fetch_and_store_data,
                'cron',
                hour=4,
                minute=0,
                args=[symbol],
                id=f"fetch_{symbol}",
                replace_existing=True
            )
        log.info(f"Scheduled daily data fetching for {len(stock_symbols)} stocks.")

    def schedule_backfill_tasks(self):
        """Schedules a one-time job for each stock to fetch its entire history."""
        stock_symbols = self._get_stock_symbols()
        if not stock_symbols:
            return 0

        for symbol in stock_symbols:
            
            self.scheduler.add_job(
                self.fetch_and_store_full_history,
                args=[symbol],
                id=f"backfill_{symbol}",
                replace_existing=True  
            )
        log.info(f"Scheduled backfill tasks for {len(stock_symbols)} stocks.")
        return len(stock_symbols)

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

    def fetch_and_store_data(self, symbol: str):
        """
        The job function that fetches recent data for a single symbol and stores it.
        """
        self._fetch_and_store(symbol, period="5d", job_type="daily")

    def fetch_and_store_full_history(self, symbol: str):
        """
        The job function that fetches the entire history for a single symbol and stores it.
        """
        self._fetch_and_store(symbol, period="max", job_type="backfill")

    def start(self):
        """Starts the scheduler if it is not already running."""
        if not self.scheduler.running:
            self.scheduler.start()
            log.info("Scheduler started.")

    def stop(self):
        self.scheduler.shutdown()
        self.storage.close()