import logging
import atexit
from datetime import date

# --- Gunicorn Logger Integration ---
if __name__ != "__main__":
    gunicorn_logger = logging.getLogger("gunicorn.error")
    root_logger = logging.getLogger()
    root_logger.handlers = gunicorn_logger.handlers
    root_logger.setLevel(logging.DEBUG)
    logging.getLogger("yfinance").setLevel(logging.INFO)
    logging.getLogger("tz_kv").setLevel(logging.WARNING)

# --- Now, import the rest of the application ---
from flask import Flask, jsonify, request, Blueprint

from scheduler.task import TaskScheduler
from routes.features import features_bp
from data.storage import Storage
from data.sentiment import bulk_analyze_sentiment
from config import Config

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
log = logging.getLogger(__name__)

# Register Blueprints
app.register_blueprint(features_bp)

# --- Scheduler Initialization ---
log.info("Initializing and starting the scheduler...")
scheduler = TaskScheduler()
scheduler.schedule_daily_tasks()
scheduler.start()
atexit.register(lambda: scheduler.stop())
log.info("Scheduler started and daily tasks scheduled.")
# --------------------------------

# --- Blueprints ---
sentiment_bp = Blueprint('sentiment_bp', __name__)

@sentiment_bp.route('/sentiment-analysis', methods=['POST'])
def run_sentiment_analysis():
    """
    Triggers sentiment analysis for pending news articles.
    """
    log.info("Sentiment analysis endpoint triggered.")
    
    config = Config()
    storage = Storage(config)
    
    try:
        # 1. Fetch pending articles
        pending_articles = storage.get_pending_sentiment_articles()
        if not pending_articles:
            log.info("No pending articles found for sentiment analysis.")
            return jsonify({"message": "No pending articles to analyze."}), 200

        log.info(f"Found {len(pending_articles)} articles for sentiment analysis.")

        # 2. Perform sentiment analysis
        # The `get_pending_sentiment_articles` returns a list of tuples (id, headline)
        sentiment_results = bulk_analyze_sentiment(pending_articles)

        # 3. Update database with scores
        if sentiment_results:
            storage.update_sentiment_scores(sentiment_results)
            log.info(f"Successfully updated sentiment scores for {len(sentiment_results)} articles.")
        
        return jsonify({
            "message": f"Sentiment analysis completed for {len(sentiment_results)} articles."
        }), 200

    except Exception as e:
        log.error("An error occurred during sentiment analysis.", exc_info=True)
        return jsonify({"error": "An internal error occurred."}), 500
    finally:
        storage.close()

app.register_blueprint(sentiment_bp, url_prefix='/data')


# --- API Endpoints ---

@app.route('/')
def index():
    return "Stock data scheduler is running."

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/backfill/prices', methods=['POST'])
def trigger_price_backfill():
    log.info("Price backfill endpoint triggered.")
    count = scheduler.schedule_backfill_tasks()
    return jsonify({
        "message": f"Successfully scheduled price backfill for {count} stocks. Check logs for progress."
    }), 202

@app.route('/backfill/news/full', methods=['POST'])
def trigger_full_news_backfill():
    log.info("Full news backfill endpoint triggered.")
    skip_existing = request.json.get('skip_existing', False) if request.is_json else False
    log.info(f"Full news backfill requested with skip_existing={skip_existing}.")
    count = scheduler.schedule_full_news_backfill_for_all_symbols(skip_existing=skip_existing)
    return jsonify({
        "message": f"Successfully scheduled news backfill for {count} stocks based on their price data range. Check logs for progress."
    }), 202

@app.route('/backfill/news', methods=['POST'])
def trigger_news_backfill_for_symbol():
    log.info("Single-symbol news backfill endpoint triggered.")
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    symbol = data.get('symbol')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')

    if not all([symbol, start_date_str, end_date_str]):
        return jsonify({"error": "Missing 'symbol', 'start_date', or 'end_date' in request body"}), 400

    try:
        # Basic date validation
        date.fromisoformat(start_date_str)
        date.fromisoformat(end_date_str)
    except ValueError:
        return jsonify({"error": "Invalid date format. Please use YYYY-MM-DD."}), 400

    scheduler.backfill_news_for_symbol(symbol=symbol, start_date=start_date_str, end_date=end_date_str)

    return jsonify({
        "message": f"Successfully scheduled news backfill for symbol {symbol} from {start_date_str} to {end_date_str}. Check logs for progress."
    }), 202

@app.route('/trigger-daily-fetch', methods=['PUT'])
def trigger_daily_fetch():
    log.info("Daily fetch endpoint triggered.")
    days = 5
    try:
        data = request.get_json(silent=True)
        if data and 'days' in data:
            parsed_days = int(data['days'])
            if parsed_days > 0:
                days = parsed_days
    except (ValueError, TypeError):
        log.warning(f"Received invalid 'days' value, using default {days}.")

    count = scheduler.run_daily_fetch_now(days=days)
    return jsonify({
        "message": f"Successfully scheduled daily fetch for {count} stocks, fetching the last {days} days. Check logs for progress."
    }), 202

@app.route('/trigger-fetch-single', methods=['PUT'])
def trigger_fetch_single():
    log.info("Single stock fetch endpoint triggered.")
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    symbol = data.get('symbol')
    days = data.get('days', 5)

    if not symbol:
        return jsonify({"error": "Missing 'symbol' in request body"}), 400

    try:
        days = int(days)
        if days <= 0:
            days = 5
    except (ValueError, TypeError):
        days = 5

    scheduler.fetch_single_stock_now(symbol=symbol, days=days)

    return jsonify({
        "message": f"Successfully scheduled a fetch for symbol {symbol}, fetching the last {days} days. Check logs for progress."
    }), 202

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(port=5000, debug=True)
