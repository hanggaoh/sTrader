import logging
import atexit

# --- Gunicorn Logger Integration ---
# This block MUST run before any other application modules are imported.
# This ensures that all loggers created by imported modules will inherit this configuration.
if __name__ != "__main__":
    # Get the Gunicorn logger instance
    gunicorn_logger = logging.getLogger("gunicorn.error")

    # Get the root logger and configure it to use Gunicorn's handlers and level.
    root_logger = logging.getLogger()
    root_logger.handlers = gunicorn_logger.handlers
    root_logger.setLevel(logging.DEBUG)

    # --- Silence noisy third-party loggers ---
    logging.getLogger("yfinance").setLevel(logging.INFO)
    logging.getLogger("tz_kv").setLevel(logging.WARNING)

# --- Now, import the rest of the Flask application ---
from flask import Flask, jsonify, request
from scheduler.task import TaskScheduler

# The logger in `scheduler.task` will now be created AFTER the root logger is configured.

app = Flask(__name__)

# It's still a good practice to ensure the Flask app's logger level is aligned.
app.logger.setLevel(logging.DEBUG)

# Use a logger for this file. It will also inherit the root configuration.
log = logging.getLogger(__name__)

# --- Scheduler Initialization ---
log.info("Initializing and starting the scheduler...")
scheduler = TaskScheduler()
scheduler.schedule_daily_tasks()
scheduler.start()
atexit.register(lambda: scheduler.stop())
log.info("Scheduler started and daily tasks scheduled.")
# --------------------------------

# A simple route to confirm the web server is running.
@app.route('/')
def index():
    return "Stock data scheduler is running."

@app.route('/health')
def health_check():
    """A simple health check endpoint that returns a 200 OK status."""
    return jsonify({"status": "healthy"}), 200

@app.route('/trigger-backfill', methods=['PUT'])
def trigger_backfill():
    """
    Triggers a one-time background task to fetch and store the full
    historical data for all stocks in the JSON file.
    """
    log.info("Backfill endpoint triggered.")
    count = scheduler.schedule_backfill_tasks()
    return jsonify({
        "message": f"Successfully scheduled backfill for {count} stocks. Check logs for progress."
    }), 202

@app.route('/trigger-daily-fetch', methods=['PUT'])
def trigger_daily_fetch():
    """
    Triggers a one-time background task to fetch and store recent
    historical data for all stocks.
    Accepts an optional 'days' parameter in the JSON body.
    """
    log.info("Daily fetch endpoint triggered.")
    
    days = 5
    try:
        data = request.get_json(silent=True)
        if data and 'days' in data:
            parsed_days = int(data['days'])
            if parsed_days > 0:
                days = parsed_days
            else:
                log.warning(f"Received non-positive 'days' value ({data['days']}), using default {days}.")
    except (ValueError, TypeError):
        log.warning(f"Received invalid 'days' value, using default {days}.")

    count = scheduler.run_daily_fetch_now(days=days)
    return jsonify({
        "message": f"Successfully scheduled daily fetch for {count} stocks, fetching the last {days} days. Check logs for progress."
    }), 202

@app.route('/trigger-fetch-single', methods=['PUT'])
def trigger_fetch_single():
    """
    Triggers a one-time background task to fetch and store recent
    historical data for a single stock.
    Accepts 'symbol' and 'days' in the JSON body.
    """
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
    # This block is for local development only.
    # For local, we use basicConfig. The scheduler is already started above.
    logging.basicConfig(level=logging.INFO)
    app.run(port=5000, debug=True)
