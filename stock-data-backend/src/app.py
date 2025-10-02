import logging
import atexit
import threading
from datetime import date

from pydantic import BaseModel, validator

# --- Gunicorn Logger Integration ---
if __name__ != "__main__":
    gunicorn_logger = logging.getLogger("gunicorn.error")
    root_logger = logging.getLogger()
    root_logger.handlers = gunicorn_logger.handlers
    root_logger.setLevel(logging.DEBUG)
    logging.getLogger("yfinance").setLevel(logging.INFO)
    logging.getLogger("tz_kv").setLevel(logging.WARNING)

# --- Now, import the rest of the application ---
from flask import Flask, jsonify, request

from config import Config
from data.storage import Storage
from ml.preprocessor import calculate_and_store_features
from scheduler.task import TaskScheduler

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
log = logging.getLogger(__name__)

# --- Pydantic model for request validation ---
class BackfillRequest(BaseModel):
    start_date: date
    end_date: date

    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be on or after start_date')
        return v

# --- Scheduler Initialization ---
log.info("Initializing and starting the scheduler...")
scheduler = TaskScheduler()
scheduler.schedule_daily_tasks()
scheduler.start()
atexit.register(lambda: scheduler.stop())
log.info("Scheduler started and daily tasks scheduled.")
# --------------------------------

# --- API Endpoints ---

@app.route('/')
def index():
    return "Stock data scheduler is running."

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/features/backfill', methods=['POST'])
def backfill_features():
    """
    Triggers a background task to calculate and store features for a given date range.
    """
    log.info("Feature backfill endpoint triggered.")
    
    # --- Request Validation ---
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    try:
        data = BackfillRequest(**request.get_json())
    except Exception as e:
        return jsonify({"error": "Invalid request body", "details": str(e)}), 400

    # --- Background Task Logic ---
    def worker(start, end):
        """The function that will run in the background thread."""
        log.info(f"Background worker started for feature backfill: {start} to {end}.")
        storage = None
        try:
            # Create a new Storage instance for this thread.
            app_config = Config()
            storage = Storage(config=app_config)
            calculate_and_store_features(storage, start.isoformat(), end.isoformat())
        except Exception as e:
            log.error(f"Error in feature backfill worker: {e}", exc_info=True)
        finally:
            if storage:
                storage.close()
            log.info(f"Background worker finished for feature backfill: {start} to {end}.")

    # --- Start the thread and return immediately ---
    thread = threading.Thread(target=worker, args=(data.start_date, data.end_date))
    thread.start()

    return jsonify({
        "message": f"Successfully started feature backfill from {data.start_date} to {data.end_date}. Check logs for progress."
    }), 202


@app.route('/trigger-backfill', methods=['PUT'])
def trigger_backfill():
    log.info("Backfill endpoint triggered.")
    count = scheduler.schedule_backfill_tasks()
    return jsonify({
        "message": f"Successfully scheduled backfill for {count} stocks. Check logs for progress."
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
