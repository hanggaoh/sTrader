import numpy as np
import pandas as pd

def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds sentiment-based features."""
    df['sentiment_3d_avg'] = df.groupby('symbol')['sentiment'].transform(lambda s: s.rolling(3).mean())
    df['sentiment_7d_avg'] = df.groupby('symbol')['sentiment'].transform(lambda s: s.rolling(7).mean())
    df['sentiment_momentum'] = df.groupby('symbol')['sentiment'].diff()
    return df

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds trend-based technical indicators like SMA, EMA, and MACD."""
    df['sma_5'] = df.groupby('symbol')['close'].transform(lambda s: s.rolling(5).mean())
    df['sma_10'] = df.groupby('symbol')['close'].transform(lambda s: s.rolling(10).mean())
    df['sma_20'] = df.groupby('symbol')['close'].transform(lambda s: s.rolling(20).mean())
    df['sma_50'] = df.groupby('symbol')['close'].transform(lambda s: s.rolling(50).mean())
    df['ema_20'] = df.groupby('symbol')['close'].transform(lambda s: s.ewm(span=20, adjust=False).mean())

    ema_12 = df.groupby('symbol')['close'].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema_26 = df.groupby('symbol')['close'].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df.groupby('symbol')['macd'].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df

def add_momentum_indicators(df: pd.DataFrame, epsilon: float = 1e-10) -> pd.DataFrame:
    """Adds momentum-based technical indicators like RSI and returns."""
    delta = df.groupby('symbol')['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + epsilon)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['return_1d'] = df.groupby('symbol')['close'].pct_change(1)
    df['return_5d'] = df.groupby('symbol')['close'].pct_change(5)
    df['return_21d'] = df.groupby('symbol')['close'].pct_change(21)
    return df

def add_volatility_indicators(df: pd.DataFrame, epsilon: float = 1e-10) -> pd.DataFrame:
    """Adds volatility-based technical indicators like ATR and ADX."""
    df['volatility_21d'] = df.groupby('symbol')['return_1d'].transform(lambda s: s.rolling(21).std()).fillna(0)
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df.groupby('symbol')['close'].shift())
    low_close = np.abs(df['low'] - df.groupby('symbol')['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, adjust=False).mean()

    plus_dm = df.groupby('symbol')['high'].diff()
    minus_dm = df.groupby('symbol')['low'].diff().mul(-1)
    plus_dm[plus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < 0] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr14 = tr.rolling(14).sum()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / (tr14 + epsilon))
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / (tr14 + epsilon))
    
    di_sum = plus_di + minus_di
    dx = (100 * np.abs(plus_di - minus_di) / (di_sum + epsilon)).fillna(0)
    df['adx'] = dx.ewm(alpha=1/14).mean()
    return df

def add_distributional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds distributional features like skew and kurtosis."""
    df['skew_21d'] = df.groupby('symbol')['return_1d'].transform(lambda s: s.rolling(21).skew()).fillna(0)
    df['kurt_21d'] = df.groupby('symbol')['return_1d'].transform(lambda s: s.rolling(21).kurt()).fillna(0)
    return df

def add_target_variable(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Adds the target variable for classification."""
    next_close = df.groupby('symbol')['close'].shift(-horizon)
    df['target'] = np.where(next_close.isna(), np.nan, (next_close > df['close']))
    df['target'] = df['target'].astype(pd.Int64Dtype())
    return df
