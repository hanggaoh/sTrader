from __future__ import annotations

from functools import partial
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from data.storage import Storage
from ml.config import Config
from ml.features import (
    add_sentiment_features,
    add_trend_indicators,
    add_momentum_indicators,
    add_volatility_indicators,
    add_distributional_features,
    add_volume_and_volatility_features,
    add_target_variable,
)

log = logging.getLogger(__name__)

class FeatureBuilder:
    """A class to orchestrate the creation of features in a modular way."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.feature_functions = []
        self.base_feats = [
            'open', 'high', 'low', 'close', 'volume', 
            'sentiment', 'sentiment_3d_avg', 'sentiment_7d_avg', 'sentiment_momentum',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_20', 
            'macd', 'macd_signal', 'macd_hist',
            'adx',
            'rsi', 
            'return_1d', 'return_5d', 'return_21d',
            'volatility_21d',
            'atr',
            'skew_21d',
            'kurt_21d', 'obv', 'bb_percent_b', 'bb_width'
        ]

    def register_feature(self, func, **kwargs):
        """Registers a feature calculation function."""
        self.feature_functions.append(partial(func, **kwargs))

    def build(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Builds all registered features on the given DataFrame."""
        log.info("Building features...")
        df = df.copy()

        for func in self.feature_functions:
            df = func(df)
        
        feats = self.cfg.features or self.base_feats
        # Filter for only the features that were actually created
        final_feats = [f for f in feats if f in df.columns]
        df = df.dropna(subset=final_feats).copy()
        log.info(f"Finished building features. Final dataset shape: {df.shape}")
        return df, final_feats

def _calculate_daily_sentiment(storage: Storage, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches all scored news articles within a date range and calculates the daily average sentiment for each stock.
    """
    log.info(f"Calculating daily sentiment from {start_date} to {end_date}.")
    sql = """
        SELECT published_at, stock_symbol, sentiment_score
        FROM news_sentiment
        WHERE status = 'PROCESSED' AND sentiment_score IS NOT NULL
          AND published_at BETWEEN %s AND %s
    """
    with storage.pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, [start_date, end_date])
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            sentiment_df = pd.DataFrame(data, columns=columns)

    if sentiment_df.empty:
        log.info("No processed sentiment data found for the given date range.")
        return pd.DataFrame(columns=['date', 'symbol', 'sentiment'])

    sentiment_df['date'] = pd.to_datetime(sentiment_df['published_at']).dt.date
    daily_sentiment = sentiment_df.groupby(['date', 'stock_symbol'])['sentiment_score'].mean().reset_index()
    daily_sentiment.rename(columns={'stock_symbol': 'symbol', 'sentiment_score': 'sentiment'}, inplace=True)
    
    log.info(f"Successfully calculated {len(daily_sentiment)} daily sentiment scores.")
    return daily_sentiment

def build_features(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    """Initializes and runs the FeatureBuilder with a standard set of features."""
    epsilon = 1e-10
    
    builder = FeatureBuilder(cfg)
    builder.register_feature(add_sentiment_features)
    builder.register_feature(add_trend_indicators)
    builder.register_feature(add_momentum_indicators, epsilon=epsilon)
    builder.register_feature(add_volatility_indicators, epsilon=epsilon)
    builder.register_feature(add_distributional_features)
    builder.register_feature(add_volume_and_volatility_features, epsilon=epsilon)
    
    return builder.build(df)

def calculate_and_store_features(storage: Storage, start_date: str, end_date: str, symbol: str = None):
    """
    Fetches raw data for a date range, calculates features, and stores them in the database.
    """
    log.info(f"Starting feature calculation for {start_date} to {end_date}.")
    
    # 1. Fetch raw data from the database
    sql = """
        SELECT time, stock_symbol, open, high, low, close, volume 
        FROM stock_data 
        WHERE time BETWEEN %s AND %s
    """
    params = [start_date, end_date]
    if symbol:
        sql += " AND stock_symbol = %s"
        params.append(symbol)
    sql += " ORDER BY time ASC"

    with storage.pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            raw_df = pd.DataFrame(data, columns=columns)

    if raw_df.empty:
        log.warning("No raw data found for the specified date range. Aborting.")
        return

    # 2. Prepare the DataFrame for feature building
    raw_df = raw_df.rename(columns={"stock_symbol": "symbol", "time": "timestamp"})
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)
    raw_df['date'] = raw_df['timestamp'].dt.date

    # 3. Calculate and merge daily sentiment scores
    daily_sentiment_df = _calculate_daily_sentiment(storage, start_date, end_date)
    if not daily_sentiment_df.empty:
        raw_df = pd.merge(raw_df, daily_sentiment_df, on=['date', 'symbol'], how='left')
        raw_df['sentiment'] = raw_df['sentiment'].fillna(0.0)
    else:
        raw_df['sentiment'] = 0.0

    # 4. Build features
    cfg = Config(horizon=1)
    features_df, _ = build_features(raw_df, cfg)

    if features_df.empty:
        log.warning("Feature calculation resulted in an empty DataFrame. Nothing to store.")
        return

    # 5. Store the new features in the database
    log.info(f"Storing {len(features_df)} rows of calculated features...")
    storage.store_features(features_df)
    log.info("Feature calculation and storage complete.")
