from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

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
    df["target"] = (df.groupby('symbol')['close'].shift(-cfg.horizon) > df['close']).astype(int)
    
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
