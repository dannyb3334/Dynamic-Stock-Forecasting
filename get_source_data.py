import os
import pandas as pd
import yfinance as yf
import datetime

"""
Data classes require returned columns in the format:
index = 'time' and columns = ['open', 'high', 'low', 'close', 'volume']
"""

class DataProvider:
    """
    Base class for data providers. Subclasses should implement the `fetch_by_ticker` method.
    """

    def fetch_by_ticker(ticker):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def crop_open_market_data(data):
        """
        Filters the data to include only the times between market open (09:30) and close (16:00).
        """
        return data.between_time('09:30', '16:00')


class Databento(DataProvider):
    """
    Data provider for fetching data from local CSV files.
    """

    def _find_csv_with_keyword(keyword, folder='XNAS-20250204-NEXW4MSSYB/decompressed/'):
        """
        Searches for a CSV file in the specified folder that contains the given keyword.
        """
        for filename in os.listdir(folder):
            if filename.endswith('.csv') and keyword.upper() in filename:
                return os.path.join(folder, filename)
        return None
    
    def fetch_by_ticker(ticker):
        """
        Fetches historical data for the given ticker from a CSV file.
        """
        history = Databento._find_csv_with_keyword(ticker) # Expected for Databento
        if history:
            data = pd.read_csv(history)
            # Convert 'ts_event' to datetime and set it as the index
            data['date'] = pd.to_datetime(data['ts_event']).dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            data.drop(columns=['ts_event'], inplace=True)
            data.set_index('date', inplace=True)
            # Select and rename required columns
            data = data[['close', 'volume', 'open', 'high', 'low']]
            # Filter data to market open hours
            data = Databento.crop_open_market_data(data)
            data.reset_index(inplace=True)
            return data
        else:
            print(f"No CSV file found containing the keyword '{ticker}' in the 'decompressed' folder.")
            return None


class YahooFinance(DataProvider):
    """
    Data provider for fetching data from Yahoo Finance.
    """
    # See https://pypi.org/project/yfinance/ for more details on the library
    def fetch_by_ticker(ticker, start_date='', end_date='', interval='1m'):
        """
        Fetches historical data for the given ticker from Yahoo Finance.
        """
        # If no start and end date are provided, fetch data for the last day
        if start_date == '' and end_date == '':
            start_date = end_date - datetime.timedelta(days=1)
            end_date = datetime.datetime.now()
            data = yf.download(ticker, interval='1m', start=start_date, end=end_date)
        else:
            data = yf.download(ticker, interval=interval, start=start_date, end=end_date)

        if not data.empty:
            # Reset index and convert datetime to Eastern timezone
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Datetime']).dt.tz_convert('US/Eastern')
            # Flatten multi-level columns and rename them
            data.columns = data.columns.get_level_values(0)
            data.index.name = 'Date'
            # Select and rename required columns
            data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Date']]
            data.columns = [col.lower() for col in data.columns]
            data = data[['date', 'close', 'volume', 'open', 'high', 'low']]
            return data
        else:
            print(f"No data found for ticker '{ticker}' on Yahoo Finance.")
            return None


if __name__ == "__main__":
    # Example usage: Fetch data for Apple, (AAPL) using Yahoo Finance
    ticker = 'AAPL'
    data = YahooFinance.fetch_by_ticker(ticker)
    if data is not None:
        print(data.head())
