#%%
# !pip install yfinance

#%%

import yfinance as yf

class DataDownloader:
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol

    def fetch_data(self):
        stock = yf.Ticker(self.stock_symbol)
        hist_data = stock.history(period="max")
        return hist_data


if __name__ == "__main__":
    downloader = DataDownloader("AAPL")
    data = downloader.fetch_data()
    print(data.head())
