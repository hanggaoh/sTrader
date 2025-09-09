from flask import Flask, jsonify, request
from scheduler.task import TaskScheduler
import os
import atexit
import logging
import logger_config

app = Flask(__name__)

log = logging.getLogger(__name__)

# Instantiate the scheduler once.
scheduler = TaskScheduler()

# Ensure the scheduler and its database connection are shut down gracefully on exit.
atexit.register(lambda: scheduler.stop())

# A simple route to confirm the web server is running.
@app.route('/')
def index():
    return "Stock data scheduler is running."

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
        # Use silent=True to avoid an exception if the body is not JSON or mimetype is wrong.
        # It will return None in that case, and the default 'days' will be used.
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
    # Schedule the daily recurring tasks
    scheduler.schedule_daily_tasks()

    # Start the scheduler. When Flask is in debug mode, it uses a reloader
    # that will restart this process on code changes. The scheduler will be
    # correctly instantiated and started in the new child process.
    scheduler.start()

    # For development, it's helpful to run with debug=True.
    # For production, use a proper WSGI server like Gunicorn or uWSGI.
    app.run(port=5000, debug=True)