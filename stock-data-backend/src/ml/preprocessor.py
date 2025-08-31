import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    """
    Prepares historical stock data for LSTM model training.
    This includes scaling the data and creating sequences.
    """

    def __init__(self, look_back: int = 60):
        """
        Args:
            look_back (int): The number of previous time steps to use as
                             input variables to predict the next time period.
        """
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocess(self, data: pd.DataFrame):
        """
        Scales the 'close' price and creates training sequences.

        Args:
            data (pd.DataFrame): DataFrame with historical stock data,
                                 expecting a 'close' column.

        Returns:
            A tuple containing:
            - np.ndarray: Training data sequences (X_train).
            - np.ndarray: Target values (y_train).
            - MinMaxScaler: The scaler object used for transformation.
        """
        # We are interested in the 'close' price
        close_prices = data['close'].values.reshape(-1, 1)

        # Scale the data
        scaled_data = self.scaler.fit_transform(close_prices)

        # Create training sequences
        X_train = []
        y_train = []

        for i in range(self.look_back, len(scaled_data)):
            X_train.append(scaled_data[i - self.look_back:i, 0])
            y_train.append(scaled_data[i, 0])

        # Convert to numpy arrays
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Reshape data for LSTM [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        return X_train, y_train, self.scaler