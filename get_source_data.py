import os
import pandas as pd

""" Data classes require returned columns in format index= 'time' and columns= ['open', 'high', 'low', 'close', 'volume'] """

class Databento:
    def _find_csv_with_keyword(keyword, folder='XNAS-20250204-NEXW4MSSYB/decompressed/'):
        # List all files in the 'decompressed' folder
        for filename in os.listdir(folder):
            # Check if the file is a .csv and contains the keyword
            if filename.endswith('.csv') and keyword.upper() in filename:
                return os.path.join(folder, filename)
        return None
    
    def fetch_by_ticker(ticker):
        history = Databento._find_csv_with_keyword(ticker)
        if history:
            data = pd.read_csv(history)
            # Rename the 'ts_event' column to 'time' and set it as the index
            data['time'] = pd.to_datetime(data['ts_event']).dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            data.set_index('time', inplace=True)
            data.drop(columns=['ts_event'], inplace=True)
            # Select only the required columns and capitalize their first letter
            data = data[['close','volume', 'open', 'high', 'low']]

            data = Databento.crop_open_market_data(data)

            return data
        else:
            print(f"No CSV file found containing the keyword '{ticker}' in the 'decompressed' folder.")
            return None

    def crop_open_market_data(data):
        # Define the market open and close times
        market_open = data.index.normalize() + pd.Timedelta(hours=9, minutes=30)
        market_close = data.index.normalize() + pd.Timedelta(hours=16)

        # Filter the data to include only the times between market open and close
        data = data.between_time('09:30', '16:00')

        return data
        
