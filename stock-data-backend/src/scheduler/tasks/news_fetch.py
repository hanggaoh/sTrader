import logging
import random
import time
from typing import List, Dict

from data.storage import Storage
from data.news_api import fetch_news_for_symbol

log = logging.getLogger(__name__)


class NewsFetchTask:
    def __init__(self, storage: Storage, stock_list: List[Dict[str, str]]):
        self.storage = storage
        self.stock_list = stock_list

    def run(self):
        log.info(f"Starting master news fetch job for {len(self.stock_list)} stocks.")
        for i, stock in enumerate(self.stock_list):
            if (i + 1) % 50 == 0:
                log.info(f"News fetch progress: {i + 1}/{len(self.stock_list)} stocks processed.")
            try:
                articles = fetch_news_for_symbol(stock['symbol'], stock['name'])
                if articles:
                    # Deduplicate articles by headline before storing
                    seen_headlines = set()
                    unique_articles = []
                    for article in articles:
                        headline = article.get('title')
                        if headline and headline not in seen_headlines:
                            unique_articles.append(article)
                            seen_headlines.add(headline)

                    if len(articles) > len(unique_articles):
                        log.info(f"Removed {len(articles) - len(unique_articles)} duplicate articles for symbol {stock['symbol']}.")

                    if unique_articles:
                        records_to_store = [(a['publishedAt'], stock['symbol'], a['title'], a.get('url'), a.get('content')) for a in unique_articles if a.get('content')]
                        self.storage.store_raw_news(records_to_store)
                
                # Sleep to respect NewsAPI's rate limit on the developer plan.
                sleep_duration = random.uniform(10.0, 15.0)
                log.info(f"Sleeping for {sleep_duration:.2f} seconds to respect API rate limit.")
                time.sleep(sleep_duration)

            except Exception as e:
                log.error(f"Error fetching news for {stock['symbol']} ({stock['name']}): {e}", exc_info=True)
        log.info("Master news fetch job complete.")
