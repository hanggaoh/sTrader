import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from flask import Blueprint, jsonify, request
from pydantic import BaseModel, field_validator

from config import Config
from data.storage import Storage
from ml.preprocessor import calculate_and_store_features
from utils.backfill import backfill_one_symbol_feature

log = logging.getLogger(__name__)

features_bp = Blueprint('features_bp', __name__)

class BackfillRequest(BaseModel):
    start_date: date
    end_date: date

    @field_validator('end_date')
    def end_date_must_be_after_start_date(cls, v, info):
        if 'start_date' in info.data and v < info.data['start_date']:
            raise ValueError('end_date must be on or after start_date')
        return v

@features_bp.route('/features/backfill', methods=['POST'])
def backfill_features():
    """
    Triggers a background task to calculate and store features for a given date range.
    """
    log.info("Feature backfill endpoint triggered.")
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    try:
        data = BackfillRequest(**request.get_json())
    except Exception as e:
        return jsonify({"error": "Invalid request body", "details": str(e)}), 400

    def worker(start, end):
        """The function that will run in the background thread."""
        log.info(f"Background worker started for feature backfill: {start} to {end}.")
        storage = None
        try:
            app_config = Config()
            storage = Storage(config=app_config)
            calculate_and_store_features(storage, start.isoformat(), end.isoformat())
        except Exception as e:
            log.error(f"Error in feature backfill worker: {e}", exc_info=True)
        finally:
            if storage:
                storage.close()
            log.info(f"Background worker finished for feature backfill: {start} to {end}.")

    thread = threading.Thread(target=worker, args=(data.start_date, data.end_date))
    thread.start()

    return jsonify({
        "message": f"Successfully started feature backfill from {data.start_date} to {data.end_date}. Check logs for progress."
    }), 202

@features_bp.route('/features/full-backfill', methods=['POST'])
def full_backfill_features():
    """
    Triggers a full feature backfill for all stocks in parallel.
    """
    log.info("Parallel full feature backfill endpoint triggered.")

    def worker():
        log.info("Background worker started for parallel full feature backfill.")
        main_storage = None
        try:
            app_config = Config()
            main_storage = Storage(config=app_config)
            symbols = main_storage.get_all_distinct_symbols()
            log.info(f"Found {len(symbols)} symbols to backfill in parallel.")

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(backfill_one_symbol_feature, symbols))
            
            successful_jobs = sum(1 for r in results if r[0])
            failed_symbols = [s for success, s in results if not success]

            log.info(f"Parallel backfill finished. Success: {successful_jobs}/{len(symbols)}.")
            if failed_symbols:
                log.warning(f"The following symbols failed to backfill: {failed_symbols}")

        except Exception as e:
            log.error(f"Error in parallel full feature backfill worker: {e}", exc_info=True)
        finally:
            if main_storage:
                main_storage.close()
            log.info("Parallel full feature backfill worker finished.")

    thread = threading.Thread(target=worker)
    thread.start()

    return jsonify({
        "message": "Successfully started parallel full feature backfill for all stocks. Check logs for progress."
    }), 202
