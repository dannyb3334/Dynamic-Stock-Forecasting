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
            data['time'] = pd.to_datetime(data['ts_event'])
            data.set_index('time', inplace=True)
            data.drop(columns=['ts_event'], inplace=True)
            # Select only the required columns and capitalize their first letter
            data = data[['open', 'high', 'low', 'close', 'volume']]
            return data
        else:
            print(f"No CSV file found containing the keyword '{ticker}' in the 'decompressed' folder.")
            return None

