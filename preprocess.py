import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

from get_source_data import Databento
import indicators

class DataProcessor:
    def __init__(self, tickers, train_split_amount=0.8, val_split_amount=0.1, lead=1, lag=12, inference=False):
        self.tickers = tickers
        if not inference:
            self.train_split_amount = train_split_amount
            self.val_split_amount = val_split_amount
        self.lead = lead
        self.lag = lag
        assert lead > 0, 'Lead must be a positive integer'
        assert lag > 0, 'Lag must be a positive integer'

    def process_ticker(self, ticker, ticker_index):
        # Fetch data for the given ticker
        ticker_df = Databento.fetch_by_ticker(ticker)
        if ticker_df is None:
            return None

        # Remove holes in dataset
        ticker_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        ticker_df.dropna(inplace=True)
        ticker_df = ticker_df.head(10000)

        # Apply indicators to the dataframe
        self.apply_indicators(ticker_df)
        ticker_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        ticker_df.dropna(inplace=True)

        # Ensure the number of data points is a multiple of lag
        if len(ticker_df) % self.lag != 0:
            ticker_df = ticker_df[(len(ticker_df) % self.lag):]

        # Create sequences and labels
        X, y = self.create_sequences_and_labels(ticker_df, ticker_index)
        return self.split_and_scale_data(X, y)

    def create_sequences_and_labels(self, ticker_df, ticker_index):
        X, y = [], []
        for j in range(self.lag, len(ticker_df)-self.lead+1):
            X.append(ticker_df.iloc[j - self.lag:j].values.flatten())
            y.append((ticker_df.iloc[j + self.lead - 1]['close'], ticker_index))
        return np.array(X), np.array(y)
    
    def apply_indicators(self, ticker_df):
        print('Calculating indicators...')
        print(f"Length of data before applying indicators: {len(ticker_df)}")
        # Apply indicators from the indicators module
        ticker_df['day'] = indicators.day_of_week(ticker_df)
        ticker_df['hour'] = indicators.hour_of_day(ticker_df)
        ticker_df['month'] = indicators.month(ticker_df)
        ticker_df['ema'] = indicators.ema(ticker_df['close'])
        ticker_df['sma'] = indicators.sma(ticker_df['close'])
        ticker_df['roc'] = indicators.roc(ticker_df['close'])
        ticker_df['percent_change'] = indicators.percent_change(ticker_df['close'])
        ticker_df['difference'] = indicators.difference(ticker_df['close'])
        ticker_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        ticker_df.dropna(inplace=True)
        print(f"Length of data after applying indicators: {len(ticker_df)}")        

    def split_and_scale_data(self, X, y):
        if not self.inference:
            length = len(y)
            train_split = int(length * self.train_split_amount)
            val_split = int(length * self.val_split_amount)

            # Scale training data
            train_scalerX = StandardScaler()
            train_scalerY = StandardScaler()
            train_X = train_scalerX.fit_transform(X[:train_split])
            train_y = train_scalerY.fit_transform(y[:train_split])

            # Scale validation data
            val_scalerX = StandardScaler()
            val_scalerY = StandardScaler()
            val_X = val_scalerX.fit_transform(X[train_split:train_split + val_split])
            val_y = val_scalerY.fit_transform(y[train_split:train_split + val_split])

            # Scale test data
            test_scalerX = StandardScaler()
            test_scalerY = StandardScaler()
            test_X = test_scalerX.fit_transform(X[train_split + val_split:])
            test_y = test_scalerY.fit_transform(y[train_split + val_split:])

            return {
                'train': {'X': train_X, 'y': train_y, 'scalerY': train_scalerY},
                'val': {'X': val_X, 'y': val_y, 'scalerY': val_scalerY},
                'test': {'X': test_X, 'y': test_y, 'scalerY': test_scalerY}
            }
        else:
            # Scale inference data
            scalerX = StandardScaler()
            scalerY = StandardScaler()
            X = scalerX.fit_transform(X)
            y = scalerY.fit_transform(y)
            return {
                'test': {'X': X, 'y': y, 'scalerY': scalerY}
            }

    def save_data_splits(self, ticker, data_splits):
        # Create directories if they don't exist
        os.makedirs('feature_dataframes', exist_ok=True)
        # Save the dataframe
        with open(f'feature_dataframes/{ticker}_features.pkl', 'wb') as f:
            pickle.dump(data_splits, f)

    def process_all_tickers(self):
        for i, ticker in enumerate(self.tickers):
            data_splits = self.process_ticker(ticker, i)
            if data_splits:
                self.save_data_splits(ticker, data_splits)
        
        return len(data_splits['train']['X'][0])


if __name__ == "__main__":
    tickers = ['SPY']
    processor = DataProcessor(tickers)
    processor.process_all_tickers()
