#%%
import pymysql
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class DataStore:
    def __init__(self):
        self.connection = pymysql.connect(
            db=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 3306))
        )

    def store_data(self, stock_symbol, data):
        cursor = self.connection.cursor()
        for date, row in data.iterrows():
            cursor.execute(
                """
                INSERT INTO stock_data (stock_symbol, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                open=VALUES(open), high=VALUES(high), low=VALUES(low), close=VALUES(close), volume=VALUES(volume)
                """,
                (stock_symbol, date.date(), row['Open'], row['High'], row['Low'], row['Close'], row['Volume'])
            )
        self.connection.commit()
        cursor.close()
    def read_data(self, stock_symbol):
        query = """
        SELECT date, open, high, low, close, volume
        FROM stock_data
        WHERE stock_symbol = %s
        ORDER BY date
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (stock_symbol,))
        result = cursor.fetchall()
        cursor.close()
        return pd.DataFrame(result, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    def close_connection(self):
        self.connection.close()

#%%
# Example usage
if __name__ == "__main__":
    store = DataStore()
    # Assuming 'data' is a DataFrame with stock data
    # Corrected usage
    store.store_data("AAPL", data)
    
    # Read data for a specific stock symbol
    df = store.read_data("AAPL")
    print(df)
    store.close_connection()

# %%
