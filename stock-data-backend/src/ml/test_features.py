import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.features import (
    add_sentiment_features,
    add_trend_indicators,
    add_momentum_indicators,
    add_volatility_indicators,
    add_distributional_features,
    add_target_variable
)

class TestFeatures(unittest.TestCase):

    def setUp(self):
        """Set up a sample DataFrame for testing."""
        data = {
            'symbol': ['AAPL'] * 25 + ['GOOG'] * 25,
            'close': np.linspace(100, 150, 50),
            'high': np.linspace(102, 152, 50),
            'low': np.linspace(98, 148, 50),
            'sentiment': np.random.randn(50),
            'return_1d': np.random.randn(50) # Pre-populate for volatility/distributional tests
        }
        self.df = pd.DataFrame(data)
        # Ensure return_1d has some NaNs to test fillna
        self.df.loc[0, 'return_1d'] = np.nan

    def test_add_sentiment_features(self):
        df = add_sentiment_features(self.df.copy())
        self.assertIn('sentiment_3d_avg', df.columns)
        self.assertIn('sentiment_7d_avg', df.columns)
        self.assertIn('sentiment_momentum', df.columns)
        self.assertTrue(df['sentiment_momentum'].iloc[0:1].isna().all())
        self.assertFalse(df['sentiment_3d_avg'].iloc[2:].isna().any())

    def test_add_trend_indicators(self):
        df = add_trend_indicators(self.df.copy())
        self.assertIn('sma_5', df.columns)
        self.assertIn('macd', df.columns)
        self.assertTrue(df['sma_5'].iloc[0:4].isna().all())
        self.assertFalse(df['sma_5'].iloc[4:].isna().any())

    def test_add_momentum_indicators(self):
        df = add_momentum_indicators(self.df.copy())
        self.assertIn('rsi', df.columns)
        self.assertIn('return_1d', df.columns)
        self.assertTrue(df['rsi'].iloc[0:14].isna().all()) # RSI has a window of 14
        self.assertFalse(df['return_1d'].iloc[1:].isna().any())

    def test_add_volatility_indicators(self):
        # This function needs 'return_1d' to be calculated first
        temp_df = self.df.copy()
        temp_df['return_1d'] = temp_df.groupby('symbol')['close'].pct_change(1)
        df = add_volatility_indicators(temp_df)
        self.assertIn('volatility_21d', df.columns)
        self.assertIn('atr', df.columns)
        self.assertIn('adx', df.columns)
        # .fillna(0) is used, so there should be no NaNs in volatility_21d
        self.assertFalse(df['volatility_21d'].isna().any())

    def test_add_distributional_features(self):
        # This function also needs 'return_1d'
        temp_df = self.df.copy()
        temp_df['return_1d'] = temp_df.groupby('symbol')['close'].pct_change(1)
        df = add_distributional_features(temp_df)
        self.assertIn('skew_21d', df.columns)
        self.assertIn('kurt_21d', df.columns)
        # .fillna(0) is used, so no NaNs
        self.assertFalse(df['skew_21d'].isna().any())
        self.assertFalse(df['kurt_21d'].isna().any())

    def test_add_target_variable(self):
        df = add_target_variable(self.df.copy(), horizon=5)
        self.assertIn('target', df.columns)
        # Last 5 rows for each symbol should be NaN
        self.assertTrue(df.groupby('symbol')['target'].tail(5).isna().all())

if __name__ == '__main__':
    unittest.main()
