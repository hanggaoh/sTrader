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
    
    # Get 'days' from request body, default to 5 if not provided or invalid
    days = 5
    if request.is_json:
        days = request.get_json().get('days', 5)
    
    try:
        days = int(days)
        if days <= 0:
            days = 5 # Reset to default if non-positive
    except (ValueError, TypeError):
        days = 5 # Reset to default if not a valid integer

    count = scheduler.run_daily_fetch_now(days=days)
    return jsonify({
        "message": f"Successfully scheduled daily fetch for {count} stocks, fetching the last {days} days. Check logs for progress."
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