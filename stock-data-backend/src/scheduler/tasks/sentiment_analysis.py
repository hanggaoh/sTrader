import logging

from data.storage import Storage
from data.sentiment import analyze_sentiment

log = logging.getLogger(__name__)


class SentimentAnalysisTask:
    def __init__(self, storage: Storage):
        self.storage = storage

    def run(self):
        log.info("Starting sentiment analysis job.")
        try:
            pending_articles = self.storage.get_pending_sentiment_articles(limit=500)
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
                self.storage.update_sentiment_scores(results)
                log.info(f"Successfully analyzed and updated {len(results)} articles.")
        except Exception as e:
            log.error(f"An error occurred during the sentiment analysis loop: {e}", exc_info=True)
        log.info("Sentiment analysis job complete.")
