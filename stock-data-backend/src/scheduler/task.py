import json
import logging
import os
import random
import time
from typing import List, Dict

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler

from config import config
from data.fetcher import StockDataFetcher
from data.storage import Storage
from data.eastmoney_fetcher import EastmoneyFetcher
from scheduler.tasks.price_fetch import PriceFetchTask
from scheduler.tasks.news_fetch import NewsFetchTask
from scheduler.tasks.sentiment_analysis import SentimentAnalysisTask

log = logging.getLogger(__name__)

class TaskScheduler:
    def __init__(self):
        executors = {
            'cron_executor': ThreadPoolExecutor(max_workers=3),
            'default': ThreadPoolExecutor(max_workers=2),
            'backfill_executor': ThreadPoolExecutor(max_workers=4),
            'repair_executor': ThreadPoolExecutor(max_workers=1)
        }
        job_defaults = {'misfire_grace_time': 10800}
        self.scheduler = BackgroundScheduler(executors=executors, job_defaults=job_defaults, timezone='Asia/Shanghai')
        self.fetcher = StockDataFetcher()
        self.storage = Storage(config=config)
        self.eastmoney_fetcher = EastmoneyFetcher()

    def _get_stock_list(self) -> List[Dict[str, str]]:
        log.info("Determining stock list source...")

        # Always load the JSON file to get the symbol-to-name mapping.
        stock_list_from_json = self._get_stock_list_from_json()
        if not stock_list_from_json:
            log.error("Could not load stock list from JSON. Cannot proceed with scheduling news tasks.")
            return []

        symbol_to_name_map = {stock['symbol']: stock['name'] for stock in stock_list_from_json}

        # Prioritize using symbols that are actually in our database.
        db_symbols = self.storage.get_all_distinct_symbols()
        if db_symbols:
            log.info(f"Using {len(db_symbols)} symbols from the database as the source of truth.")
            # Create the final list, ensuring we have the correct name for each symbol.
            final_stock_list = []
            for symbol in db_symbols:
                name = symbol_to_name_map.get(symbol)
                if name:
                    final_stock_list.append({'symbol': symbol, 'name': name})
                else:
                    log.warning(f"Symbol {symbol} from database not found in JSON file. Using symbol as name.")
                    final_stock_list.append({'symbol': symbol, 'name': symbol})
            return final_stock_list

        log.warning("Database contains no stock symbols. Falling back to full JSON file for initial list.")
        return stock_list_from_json

    def _get_stock_list_from_json(self) -> List[Dict[str, str]]:
        json_path = os.path.join(os.path.dirname(__file__), '../data/chinese_stocks.json')
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            if not isinstance(data, list) or not data:
                log.error(f"Stock file at {json_path} is not a valid JSON list or is empty.")
                return []
            first_element = data[0]
            if isinstance(first_element, str):
                return [{'symbol': symbol, 'name': symbol} for symbol in data]
            elif isinstance(first_element, dict) and 'symbol' in first_element:
                return [{'symbol': s['symbol'], 'name': s.get('name', s['symbol'])} for s in data]
            else:
                log.error(f"Stock file at {json_path} has an unknown or invalid format.")
                return []
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.error(f"Error processing stock file at {json_path}: {e}")
            return []

    def schedule_daily_tasks(self):
        stock_list = self._get_stock_list()
        if not stock_list:
            log.warning("No stocks found, skipping all daily task scheduling.")
            return

        stock_symbols = [s['symbol'] for s in stock_list]

        price_task = PriceFetchTask(self.scheduler, self.storage, self.fetcher, stock_symbols, days=5)
        news_task = NewsFetchTask(self.scheduler, self.storage, stock_list)
        sentiment_task = SentimentAnalysisTask(self.scheduler, self.storage)

        self.scheduler.add_job(price_task.run, 'cron', hour=15, minute=30, id="master_price_job", replace_existing=True, executor='cron_executor')
        log.info(f"Scheduled daily price fetch job for {len(stock_list)} stocks.")

        self.scheduler.add_job(news_task.run, 'cron', hour=16, minute=0, id="master_news_fetch_job", replace_existing=True, executor='cron_executor')
        log.info(f"Scheduled daily news fetch job for {len(stock_list)} stocks.")

        self.scheduler.add_job(sentiment_task.run, 'interval', minutes=10, id="sentiment_analysis_job", replace_existing=True, executor='default')
        log.info("Scheduled sentiment analysis job to run every 10 minutes.")

    def run_daily_fetch_now(self, days=5):
        stock_list = self._get_stock_list()
        if not stock_list:
            return 0
        
        stock_symbols = [s['symbol'] for s in stock_list]
        price_task = PriceFetchTask(self.scheduler, self.storage, self.fetcher, stock_symbols, days=days)

        log.info("Scheduling an immediate one-time run of the daily fetch job.")
        self.scheduler.add_job(price_task.run, id="immediate_daily_fetch", replace_existing=True)
        return len(stock_list)

    def fetch_single_stock_now(self, symbol: str, days: int = 5):
        log.info(f"Scheduling an immediate one-time fetch for {symbol} for the last {days} days.")
        period_str = f"{days}d"
        job_id = f"fetch_single_{symbol}_{int(time.time())}"
        price_task = PriceFetchTask(self.scheduler, self.storage, self.fetcher, days=days)
        self.scheduler.add_job(price_task._fetch_and_store_price, args=[symbol, period_str, "single-fetch"], id=job_id, replace_existing=False, executor='repair_executor')

    def schedule_backfill_tasks(self):
        stock_list = self._get_stock_list_from_json()
        if not stock_list:
            return 0
        price_task = PriceFetchTask(self.scheduler, self.storage, self.fetcher, stock_symbols=self._get_stock_list(), days=365 * 30)
        self.scheduler.add_job(price_task.run, args=[stock_list], id="scheduled_backfill_job", replace_existing=True, executor='backfill_executor')
        log.info(f"Scheduled a single backfill job for {len(stock_list)} stocks.")
        return len(stock_list)

    def schedule_full_news_backfill_for_all_symbols(self, skip_existing: bool = False):
        """
        Schedules a single news backfill job for all symbols, iterating through them
        in a memory-efficient way.
        """
        log.info("Scheduling a single, comprehensive news backfill job.")
        
        # This function will now run in the background via the scheduler
        self.scheduler.add_job(
            self._run_full_news_backfill_loop,
            args=[skip_existing],
            id="master_news_backfill_job",
            replace_existing=True,
            executor='backfill_executor'
        )
        
        log.info("Master news backfill job has been scheduled.")
        # We can't return a precise count anymore as it runs in the background
        return "Job Scheduled"

    def _run_full_news_backfill_loop(self, skip_existing: bool):
        """
        The actual job that iterates through all symbols and triggers their
        news backfill.
        """
        log.info("Starting full news backfill for all symbols based on stock data date ranges.")
        symbol_date_ranges = self.storage.get_stock_data_date_ranges()
        if not symbol_date_ranges:
            log.warning("No stock data found in the database. Cannot perform news backfill.")
            return

        log.info(f"Found {len(symbol_date_ranges)} symbols with date ranges to backfill news for.")
        
        for i, (symbol, start_date, end_date) in enumerate(symbol_date_ranges):
            if (i + 1) % 100 == 0:
                log.info(f"News backfill progress: {i + 1}/{len(symbol_date_ranges)} symbols processed.")

            if start_date and end_date:
                if skip_existing and self.storage.has_news_for_symbol(symbol):
                    log.info(f"Skipping news backfill for {symbol} as it already has news.")
                    continue
                
                # Directly call the backfill logic instead of creating a new job
                self._run_eastmoney_news_backfill(symbol, start_date.isoformat(), end_date.isoformat())
                
                # Add a small delay to avoid overwhelming the system
                time.sleep(random.uniform(0.5, 1.5))
        
        log.info("Completed the master news backfill job for all symbols.")

    def schedule_insufficient_news_backfill(self, threshold: int):
        """
        Schedules a news backfill for symbols with a news count below the threshold.
        """
        log.info(f"Starting backfill for symbols with fewer than {threshold} news articles.")
        insufficient_symbols = self.storage.get_symbols_with_insufficient_news(threshold)
        if not insufficient_symbols:
            log.info("No symbols found with insufficient news. Nothing to do.")
            return 0

        all_date_ranges = self.storage.get_stock_data_date_ranges()
        date_ranges_map = {symbol: (start, end) for symbol, start, end in all_date_ranges}

        scheduled_count = 0
        for symbol in insufficient_symbols:
            if symbol in date_ranges_map:
                start_date, end_date = date_ranges_map[symbol]
                if start_date and end_date:
                    job_id = f"news_backfill_for_{symbol}" # Use the same job ID to replace existing
                    self.scheduler.add_job(
                        self._run_eastmoney_news_backfill,
                        args=[symbol, start_date.isoformat(), end_date.isoformat()],
                        id=job_id,
                        replace_existing=True, # Replace any existing job for this symbol
                        executor='backfill_executor'
                    )
                    log.info(f"Scheduled repair backfill for {symbol} from {start_date} to {end_date}.")
                    scheduled_count += 1
            else:
                log.warning(f"Could not find date range for symbol {symbol}. Skipping backfill.")

        log.info(f"Completed scheduling. A total of {scheduled_count} repair backfill jobs were scheduled.")
        return scheduled_count

    def backfill_news_for_symbol(self, symbol: str, start_date: str, end_date: str):
        """Schedules a one-time job to backfill news for a single symbol using Eastmoney."""
        log.info(f"Scheduling news backfill for {symbol} from {start_date} to {end_date}.")
        job_id = f"eastmoney_news_backfill_{symbol}_{int(time.time())}"
        self.scheduler.add_job(
            self._run_eastmoney_news_backfill,
            args=[symbol, start_date, end_date],
            id=job_id,
            replace_existing=False,
            executor='backfill_executor'
        )

    def _run_eastmoney_news_backfill(self, symbol: str, start_date: str, end_date: str):
        """The actual job that fetches and stores news from Eastmoney."""
        log.info(f"Running Eastmoney news backfill for {symbol}...")
        try:
            news_df = self.eastmoney_fetcher.fetch_news_for_symbol(symbol, start_date, end_date)
            if not news_df.empty:
                records_to_store = []
                for _, row in news_df.iterrows():
                    records_to_store.append((row['datetime'], symbol, row['title'], row['url'], row['content']))
                
                if records_to_store:
                    self.storage.store_raw_news(records_to_store)
                    log.info(f"Stored {len(records_to_store)} news articles for {symbol} from Eastmoney.")
            else:
                log.info(f"No news found for {symbol} in the given date range from Eastmoney.")
        except Exception as e:
            log.error(f"Error during Eastmoney news backfill for {symbol}: {e}", exc_info=True)

    def _run_full_backfill_loop(self, stock_list: List[Dict[str, str]]):
        total_stocks = len(stock_list)
        log.info(f"Starting master backfill job for {total_stocks} stocks.")
        for i, stock in enumerate(stock_list):
            if (i + 1) % 100 == 0:
                log.info(f"Backfill progress: {i + 1}/{total_stocks} stocks checked.")
            if self.storage.symbol_exists(stock['symbol']):
                continue
            log.info(f"New symbol found: {stock['symbol']}. Starting full history backfill.")
            self._fetch_and_store_price(stock['symbol'], period="max", job_type="backfill")
        log.info(f"Master backfill job completed. Processed {total_stocks} stocks.")

    def _run_daily_fetch_loop(self, stock_symbols: list[str], days: int):
        period_str = f"{days}d"
        total_symbols = len(stock_symbols)
        log.info(f"Starting master daily price fetch job for {total_symbols} symbols, fetching last {period_str}.")
        for i, symbol in enumerate(stock_symbols):
            if (i + 1) % 100 == 0:
                log.info(f"Daily price fetch progress: {i + 1}/{total_symbols} symbols processed.")
            self._fetch_and_store_price(symbol, period=period_str, job_type="daily")
        log.info(f"Master daily price fetch job completed for all {total_symbols} symbols.")

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

    def start(self):
        if not self.scheduler.running:
            self.scheduler.start()
            log.info("Scheduler started.")

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown()
        self.storage.close()