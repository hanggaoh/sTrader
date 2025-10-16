import logging
from typing import Optional

from config import config
from data.storage import Storage
from data.sentiment import analyze_sentiment

log = logging.getLogger(__name__)


class SentimentAnalysisTask:
    def __init__(self, storage: Optional[Storage] = None):
        self.storage = storage

    def run(self):
        log.info("Starting sentiment analysis job.")
        
        storage_instance = self.storage
        should_close_storage = False
        if storage_instance is None:
            log.info("No storage instance provided, creating a new one.")
            storage_instance = Storage(config)
            should_close_storage = True

        try:
            pending_articles = storage_instance.get_pending_sentiment_articles(limit=500)
            if not pending_articles:
                log.info("No pending articles to analyze.")
                return

            log.info(f"Found {len(pending_articles)} pending articles to analyze.")
            results = []
            for article_id, headline, content in pending_articles:
                score = analyze_sentiment(headline, content)
                if score is not None:
                    results.append((score, article_id))
            
            if results:
                storage_instance.update_sentiment_scores(results)
                log.info(f"Successfully analyzed and updated {len(results)} articles.")

        except Exception as e:
            log.error(f"An error occurred during the sentiment analysis loop: {e}", exc_info=True)
        finally:
            if should_close_storage and storage_instance:
                storage_instance.close()
        log.info("Sentiment analysis job complete.")
