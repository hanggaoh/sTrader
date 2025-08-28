from flask import Flask, jsonify
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