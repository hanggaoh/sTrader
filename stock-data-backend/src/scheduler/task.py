import json
import logging
import os
import random
import time
from datetime import date, timedelta
from typing import List, Dict

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler

from config import config
from data.fetcher import StockDataFetcher
from data.storage import Storage
from data.sentiment import fetch_news_for_symbol, analyze_sentiment
from ml.preprocessor import calculate_and_store_features

log = logging.getLogger(__name__)

class TaskScheduler:
    def __init__(self):
        executors = {
            'cron_executor': ThreadPoolExecutor(max_workers=3),
            'default': ThreadPoolExecutor(max_workers=2),
            'backfill_executor': ThreadPoolExecutor(max_workers=1),
            'repair_executor': ThreadPoolExecutor(max_workers=1)
        }
        job_defaults = {'misfire_grace_time': 3600}
        self.scheduler = BackgroundScheduler(executors=executors, job_defaults=job_defaults)
        self.fetcher = StockDataFetcher()
        self.storage = Storage(config=config)

    def _get_stock_list(self) -> List[Dict[str, str]]:
        log.info("Determining stock list source...")
        db_symbols = self.storage.get_all_distinct_symbols()
        if db_symbols:
            log.info(f"Using {len(db_symbols)} symbols from the database as the source of truth.")
            return [{'symbol': symbol, 'name': symbol} for symbol in db_symbols]
        
        log.warning("Database contains no stock symbols. Falling back to JSON file for initial list.")
        return self._get_stock_list_from_json()

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

        # --- Schedule Daily Price Fetch ---
        self.scheduler.add_job(self._run_daily_fetch_loop, 'cron', hour=4, minute=0, args=[[s['symbol'] for s in stock_list], 5], id="master_price_job", replace_existing=True, executor='cron_executor')
        log.info(f"Scheduled daily price fetch job for {len(stock_list)} stocks.")

        # --- Schedule Daily Feature Calculation ---
        self.scheduler.add_job(self._run_daily_feature_calculation, 'cron', hour=4, minute=30, id="master_feature_job", replace_existing=True, executor='cron_executor')
        log.info("Scheduled daily feature calculation job.")

        # --- Schedule Daily News Fetch ---
        self.scheduler.add_job(self._run_daily_news_fetch_loop, 'cron', hour=5, minute=0, args=[stock_list], id="master_news_fetch_job", replace_existing=True, executor='cron_executor')
        log.info(f"Scheduled daily news fetch job for {len(stock_list)} stocks.")

        # --- Schedule Sentiment Analysis ---
        self.scheduler.add_job(self._run_sentiment_analysis_loop, 'cron', hour=5, minute=15, id="master_sentiment_analysis_job", replace_existing=True, executor='cron_executor')
        log.info("Scheduled sentiment analysis job.")

    def _run_daily_feature_calculation(self):
        log.info("Starting daily feature calculation job.")
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=90)
            calculate_and_store_features(self.storage, start_date.isoformat(), end_date.isoformat())
        except Exception as e:
            log.error(f"An error occurred during the daily feature calculation job: {e}", exc_info=True)
        log.info("Daily feature calculation job finished.")

    def _run_daily_news_fetch_loop(self, stock_list: List[Dict[str, str]]):
        log.info(f"Starting master news fetch job for {len(stock_list)} stocks.")
        for i, stock in enumerate(stock_list):
            if (i + 1) % 50 == 0:
                log.info(f"News fetch progress: {i + 1}/{len(stock_list)} stocks processed.")
            try:
                articles = fetch_news_for_symbol(stock['symbol'], stock['name'])
                if articles:
                    records_to_store = [(a['publishedAt'], stock['symbol'], a['title'], a['url']) for a in articles if a.get('title')]
                    if records_to_store:
                        self.storage.store_raw_news(records_to_store)
                time.sleep(random.uniform(1.0, 2.0))
            except Exception as e:
                log.error(f"Error fetching news for {stock['symbol']}: {e}", exc_info=True)
        log.info("Master news fetch job complete.")

    def _run_sentiment_analysis_loop(self):
        log.info("Starting sentiment analysis job.")
        try:
            pending_articles = self.storage.get_pending_sentiment_articles(limit=500)
            if not pending_articles:
                log.info("No pending articles to analyze.")
                return
            log.info(f"Found {len(pending_articles)} pending articles to analyze.")
            results = []
            for article_id, headline in pending_articles:
                score = analyze_sentiment(headline)
                if score is not None:
                    results.append((score, article_id))
            if results:
                self.storage.update_sentiment_scores(results)
                log.info(f"Successfully analyzed and updated {len(results)} articles.")
        except Exception as e:
            log.error(f"An error occurred during the sentiment analysis loop: {e}", exc_info=True)
        log.info("Sentiment analysis job complete.")

    def run_daily_fetch_now(self, days=5):
        stock_list = self._get_stock_list()
        if not stock_list:
            return 0
        log.info("Scheduling an immediate one-time run of the daily fetch job.")
        self.scheduler.add_job(self._run_daily_fetch_loop, args=[[s['symbol'] for s in stock_list], days], id="immediate_daily_fetch", replace_existing=True)
        return len(stock_list)

    def fetch_single_stock_now(self, symbol: str, days: int = 5):
        log.info(f"Scheduling an immediate one-time fetch for {symbol} for the last {days} days.")
        period_str = f"{days}d"
        job_id = f"fetch_single_{symbol}_{int(time.time())}"
        self.scheduler.add_job(self._fetch_and_store_price, args=[symbol, period_str, "single-fetch"], id=job_id, replace_existing=False, executor='repair_executor')

    def schedule_backfill_tasks(self):
        stock_list = self._get_stock_list_from_json()
        if not stock_list:
            return 0
        self.scheduler.add_job(self._run_full_backfill_loop, args=[stock_list], id="master_backfill_job", replace_existing=True, executor='backfill_executor')
        log.info(f"Scheduled a single backfill job for {len(stock_list)} stocks.")
        return len(stock_list)

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
