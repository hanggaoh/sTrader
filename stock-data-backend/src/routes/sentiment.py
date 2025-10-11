import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Blueprint, jsonify, request

from config import Config
from data.sentiment import analyze_sentiment
from data.storage import Storage

log = logging.getLogger(__name__)
sentiment_bp = Blueprint('sentiment_bp', __name__)


@sentiment_bp.route('/backfill', methods=['POST'])
def run_sentiment_analysis_backfill():
    """
    Triggers a concurrent sentiment analysis backfill for pending news articles.
    Accepts an optional 'limit' in the JSON body.
    """
    log.info("Sentiment analysis backfill endpoint triggered.")

    limit = request.json.get('limit') if request.is_json else None

    config = Config()
    storage = Storage(config)

    try:
        if limit:
            pending_articles = storage.get_pending_sentiment_articles(limit=limit)
            log.info(f"Found {len(pending_articles)} articles for sentiment analysis (limit: {limit}).")
        else:
            pending_articles = storage.get_all_pending_sentiment_articles()
            log.info(f"Found {len(pending_articles)} articles for a full sentiment analysis backfill.")

        if not pending_articles:
            return jsonify({"message": "No pending articles to analyze."}), 200

        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_article = {executor.submit(analyze_sentiment, article[1]): article for article in
                                 pending_articles}
            for future in as_completed(future_to_article):
                article = future_to_article[future]
                try:
                    score = future.result()
                    results.append((score, article[0]))
                except Exception as exc:
                    log.error(f'Article ID {article[0]} generated an exception: {exc}')

        if results:
            storage.update_sentiment_scores(results)
            log.info(f"Successfully updated sentiment scores for {len(results)} articles.")

        return jsonify({
            "message": f"Sentiment analysis backfill completed for {len(results)} articles."
        }), 200

    except Exception as e:
        log.error("An error occurred during sentiment analysis backfill.", exc_info=True)
        return jsonify({"error": "An internal error occurred."}), 500
    finally:
        storage.close()
