from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Add src directory to path to allow for imports
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
except NameError:
    sys.path.append(os.path.abspath(os.getcwd()))

from config import config
from data.storage import Storage
from ml.utils import Config, build_features, attach_sentiment, fetch_data, train_loop
from ml.inference import predict_next

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s')
logger = logging.getLogger("model_evaluation")

def run_evaluation(cfg: Config, split_date: str):
    """Runs a time-based backtest of the training and prediction pipeline."""
    storage = Storage(config)
    
    sql_query = "SELECT time, stock_symbol, open, high, low, close, volume FROM stock_data"
    params = []
    if cfg.symbols:
        placeholders = ",".join(["%s"] * len(cfg.symbols))
        sql_query += f" WHERE stock_symbol IN ({placeholders})"
        params.extend(cfg.symbols)

    full_df = fetch_data(storage, sql_query, logger, params=params)
    
    if full_df.empty:
        logger.error(f"No data found for the specified symbols: {cfg.symbols}. Aborting evaluation.")
        
        # --- Diagnostic Step ---
        logger.info("Running a diagnostic query to find all available symbols in the database...")
        try:
            with storage.pool.connection() as conn, conn.cursor() as cursor:
                cursor.execute("SELECT DISTINCT stock_symbol FROM stock_data ORDER BY stock_symbol")
                available_symbols = [row[0] for row in cursor.fetchall()]
            
            if available_symbols:
                logger.info(f"Found {len(available_symbols)} symbols in the database. Here are the first 20: {available_symbols[:20]}")
                logger.info("Please try running the evaluation again with one of the symbols listed above.")
            else:
                logger.error("Diagnostic query found NO symbols in the 'stock_data' table. Please ensure your database is populated.")
        except Exception as e:
            logger.error(f"Diagnostic query failed: {e}")
        # --- End Diagnostic Step ---
        storage.close()
        return

    split_datetime = pd.to_datetime(split_date, utc=True)
    train_df = full_df[full_df['timestamp'] < split_datetime].copy()
    test_df = full_df[full_df['timestamp'] >= split_datetime].copy()

    if train_df.empty or test_df.empty:
        logger.error(f"Not enough data to perform a train/test split at {split_date}. Aborting.")
        storage.close()
        return

    logger.info(f"Training on {len(train_df)} rows (before {split_date}). Testing on {len(test_df)} rows (on/after {split_date}).")

    train_df = attach_sentiment(train_df, cfg, logger)
    train_df, feats = build_features(train_df, cfg, logger)

    logger.info("--- Starting Backtest Training Phase ---")
    model, scalers, _ = train_loop(cfg, train_df, feats, logger)
    logger.info("--- Backtest Training Phase Complete ---")

    logger.info("--- Starting Walk-Forward Prediction Phase ---")
    predictions = []
    actuals = []
    artifacts = {"model": model, "scalers": scalers, "config": cfg}

    for symbol in test_df['symbol'].unique():
        symbol_test_df = test_df[test_df['symbol'] == symbol]
        symbol_full_hist = full_df[full_df['symbol'] == symbol]

        for i in range(len(symbol_test_df)):
            current_timestamp = symbol_test_df.iloc[i]['timestamp']
            end_idx_loc = symbol_full_hist.index.get_loc(symbol_full_hist[symbol_full_hist['timestamp'] == current_timestamp].index[0])
            start_idx = max(0, end_idx_loc - cfg.window)
            prediction_data = symbol_full_hist.iloc[start_idx:end_idx_loc]

            if len(prediction_data) < cfg.window:
                logger.warning(f"Skipping prediction for {symbol} at {current_timestamp} due to insufficient historical data ({len(prediction_data)} < {cfg.window} rows).")
                continue

            pred = predict_next(artifacts, prediction_data)
            actual = symbol_test_df.iloc[i]["close"]
            
            predictions.append(pred)
            actuals.append(actual)

    if not predictions:
        logger.error("No predictions were made. Cannot calculate test score.")
        storage.close()
        return

    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    logger.info("--- Walk-Forward Prediction Complete ---")
    logger.info(f"Backtest Mean Squared Error (MSE): {mse:.6f}")
    logger.info(f"Backtest Root Mean Squared Error (RMSE): {rmse:.6f}")

    storage.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=str, nargs='+', default=['000001.SZ'], help="List of stock symbols to evaluate.")
    p.add_argument("--split-date", type=str, default='2024-09-01', help="Date to split train/test data (e.g., '2024-09-01').")
    p.add_argument("--epochs", type=int, default=10, help="Number of epochs for the backtest training.")
    args, _ = p.parse_known_args()

    cfg = Config(symbols=args.symbols, epochs=args.epochs)

    run_evaluation(cfg, args.split_date)

if __name__ == "__main__":
    main()
