#%%

import pymysql
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.db_name = os.getenv("DB_NAME")
        self.connection = pymysql.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 3306))
        )
        self.cursor = self.connection.cursor()

    def create_database(self):
        self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_name};")
        self.cursor.execute(f"USE {self.db_name};")
        self.connection.commit()

    def delete_table(self, table_name):
        self.cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        self.connection.commit()

    def create_schema(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS stock_data (
            id SERIAL PRIMARY KEY,
            stock_symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            open DECIMAL(10, 2),
            high DECIMAL(10, 2),
            low DECIMAL(10, 2),
            close DECIMAL(10, 2),
            volume BIGINT,
            UNIQUE (stock_symbol, date)
        );
        """
        self.cursor.execute(create_table_query)
        self.connection.commit()

    def close_connection(self):
        self.cursor.close()
        self.connection.close()

if __name__ == "__main__":
    db_manager = DatabaseManager()
    db_manager.create_database()
    db_manager.delete_table("stock_data")  # Drop the stock_data table if it exists
    db_manager.create_schema()  # Create the schema
    db_manager.close_connection()

# %%
