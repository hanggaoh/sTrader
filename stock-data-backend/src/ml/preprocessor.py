from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from data.storage import Storage
from ml.config import Config

log = logging.getLogger(__name__)

def build_features(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    log.info("Building features...")
    df = df.copy()

    # --- Trend Indicators ---
    df['sma_5'] = df.groupby('symbol')['close'].transform(lambda s: s.rolling(5).mean())
    df['sma_10'] = df.groupby('symbol')['close'].transform(lambda s: s.rolling(10).mean())
    df['sma_20'] = df.groupby('symbol')['close'].transform(lambda s: s.rolling(20).mean())
    df['sma_50'] = df.groupby('symbol')['close'].transform(lambda s: s.rolling(50).mean())
    df['ema_20'] = df.groupby('symbol')['close'].transform(lambda s: s.ewm(span=20, adjust=False).mean())
    
    # MACD
    ema_12 = df.groupby('symbol')['close'].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema_26 = df.groupby('symbol')['close'].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df.groupby('symbol')['macd'].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # --- Momentum Indicators ---
    delta = df.groupby('symbol')['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['return_1d'] = df.groupby('symbol')['close'].pct_change(1)
    df['return_5d'] = df.groupby('symbol')['close'].pct_change(5)
    df['return_21d'] = df.groupby('symbol')['close'].pct_change(21)

    # --- Volatility Indicators ---
    df['volatility_21d'] = df.groupby('symbol')['return_1d'].transform(lambda s: s.rolling(21).std())
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df.groupby('symbol')['close'].shift())
    low_close = np.abs(df['low'] - df.groupby('symbol')['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, adjust=False).mean()

    # ADX
    plus_dm = df.groupby('symbol')['high'].diff()
    minus_dm = df.groupby('symbol')['low'].diff().mul(-1)
    plus_dm[plus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < 0] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / tr14)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / tr14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.ewm(alpha=1/14).mean()

    # --- Distributional Features ---
    df['skew_21d'] = df.groupby('symbol')['return_1d'].transform(lambda s: s.rolling(21).skew())
    df['kurt_21d'] = df.groupby('symbol')['return_1d'].transform(lambda s: s.rolling(21).kurt())

    # --- Target Variable (Classification) ---
    # Target is 1 if the price goes up in the next `horizon` periods, 0 otherwise.
    # We must handle the NaN from the shift at the end of the series explicitly.
    next_close = df.groupby('symbol')['close'].shift(-cfg.horizon)
    df['target'] = (next_close > df['close'])
    # Where the next close is not available, the target is undefined (NaN).
    df.loc[next_close.isna(), 'target'] = np.nan
    # Use a nullable integer type to preserve NaNs.
    df['target'] = df['target'].astype(pd.Int64Dtype())
    
    # --- Diagnostic Logging: Target Distribution ---
    # Check if the target variable is imbalanced. If it's all 0s or 1s, the model can't learn.
    target_dist = df['target'].value_counts(normalize=True)
    log.info(f"Target variable distribution:\n{target_dist}")
    if len(target_dist) < 2: 
        log.warning("The target variable has only one class! The model will not be able to learn.")
    else:
        majority_frac = target_dist.max()
        log.info(f"Sanity Check: Majority class fraction is {majority_frac:.4f}. A good model must beat this accuracy.")
        
    base_feats = [
        # Price & Volume
        'open', 'high', 'low', 'close', 'volume', 
        # Sentiment
        'sentiment', 
        # Trend
        'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_20', 
        'macd', 'macd_signal', 'macd_hist',
        'adx',
        # Momentum
        'rsi', 
        'return_1d', 'return_5d', 'return_21d',
        # Volatility
        'volatility_21d',
        'atr',
        # Distribution
        'skew_21d',
        'kurt_21d',
    ]
    feats = cfg.features or base_feats
    df = df.dropna(subset=feats + ["target"]).copy()
    log.info(f"Finished building features. Final dataset shape: {df.shape}")
    return df, feats

def calculate_and_store_features(storage: Storage, start_date: str, end_date: str, symbol: str = None):
    """
    Fetches raw data for a date range, calculates features, and stores them in the database.
    """
    log.info(f"Starting feature calculation for {start_date} to {end_date}.")
    
    # 1. Fetch raw data from the database
    with storage.pool.connection() as conn:
        sql = """
            SELECT time, stock_symbol, open, high, low, close, volume 
            FROM stock_data 
            WHERE time BETWEEN %s AND %s
        """
        params = [start_date, end_date]
        if symbol:
            sql += " AND stock_symbol = %s"
            params.append(symbol)

        # Add the ORDER BY clause at the end
        sql += " ORDER BY time ASC"

        raw_df = pd.read_sql(sql, conn, params=params)

    if raw_df.empty:
        log.warning("No raw data found for the specified date range. Aborting.")
        return

    # 2. Prepare the DataFrame for feature building
    raw_df = raw_df.rename(columns={"stock_symbol": "symbol", "time": "timestamp"})
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)
    
    # build_features expects a sentiment column. We'll join it in the future,
    # but for now, we'll use a placeholder.
    raw_df['sentiment'] = 0.0

    # 3. Build features
    # We can use a default config for the feature calculation.
    cfg = Config(horizon=1)
    features_df, _ = build_features(raw_df, cfg)

    if features_df.empty:
        log.warning("Feature calculation resulted in an empty DataFrame. Nothing to store.")
        return

    # 4. Store the new features in the database
    log.info(f"Storing {len(features_df)} rows of calculated features...")
    storage.store_features(features_df)
    log.info("Feature calculation and storage complete.")
