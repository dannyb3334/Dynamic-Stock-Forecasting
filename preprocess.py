import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

from get_source_data import *
import indicators
from typing import List

class DataProcessor:
    """DataProcessor class to handle the preprocessing of stock data for model training and inference.
    
    This class fetches stock data, applies various indicators, adjusts for stock splits, creates sequences and labels,
    splits the data into training, validation, and test sets, scales the data, and saves the processed data.
    """

    def __init__(self, provider:str, tickers:List[str], train_split_amount:float=0.8, val_split_amount:float=0.1, lead:int=2,
                 lag:int=12, inference:bool=False, col_to_predict:str='close'):
        """Initialize the DataProcessor"""
        # Check for valid input
        assert lead > 0, 'Lead must be a positive integer'
        assert lag > 0, 'Lag must be a positive integer'
        provider = eval(provider)
        #assert provider in globals(), f"{provider} is not a valid class in get_source_data"
        self.provider = provider
        self.tickers = tickers
        self.col_to_predict = col_to_predict
        self.inference = inference
        self.lead = lead
        self.lag = lag

        # Set the split amounts if not in inference mode
        if not inference:
            self.train_split_amount = train_split_amount
            self.val_split_amount = val_split_amount

    def process_ticker(self, ticker, ticker_index):
        """Process a single ticker"""
        print(f"Processing {ticker}")
        # Fetch data for the given ticker
        ticker_df = self.provider.fetch_by_ticker(ticker)
        if ticker_df is None:
            return None

        # Remove holes in dataset
        ticker_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        ticker_df.dropna(inplace=True)
        ticker_df = ticker_df.tail(10000)
        print(f"Length of data before applying features: {len(ticker_df)}")

        # Create labels
        self.apply_indicators(ticker_df)
        assert self.col_to_predict in ticker_df.columns, \
                    f"{self.col_to_predict} is not a feature column in the dataframe"
        categorical_cols = self.apply_indicators(ticker_df)
        print(ticker_df.columns)

        # Ensure the number of data points is a multiple of lag
        if len(ticker_df) % self.lag != 0:
            ticker_df = ticker_df[(len(ticker_df) % self.lag):]
        
        print(f"Length of data after applying features: {len(ticker_df)}")
        # Remove 'Date' column if it exists
        # Separate the 'Date' column if it exists
        date_df = ticker_df[['date']].copy().astype(int)
        date_df = date_df['date'].tolist()
        ticker_df.drop(columns=['date'], inplace=True)

        num_cols = len(ticker_df.columns.tolist())

        # Create sequences and labels
        print(f"Creating sequences and labels")
        X, y = self.create_sequences_and_labels(ticker_df, ticker_index, num_cols, categorical_cols, date_df)
        print(f"Splitting and scaling data")
        data_splits = self.split_and_scale_data(X, y)
        return data_splits
    
    def adjust_for_stock_splits(self, ticker_df):
        """Adjust the stock data for stock splits"""
        ticker_df['pct_change'] = ticker_df['close'][::-1].pct_change()[::-1]

        indices_to_check = ticker_df.index[ticker_df['pct_change'] > 1].tolist()[::-1]
        for idx in indices_to_check:
            split_value = int(ticker_df.iloc[idx]['pct_change']) + 1
            ticker_df.loc[0:idx, ['open', 'high', 'low', 'close']] /= split_value
            ticker_df.loc[0:idx, ['volume']] *= split_value        

    def create_sequences_and_labels(self, ticker_df, ticker_index, num_cols, categorical_cols, date_df):
        """Create sequences and labels for the model"""
        X, y = [], []
        sin_vector = np.array([[np.sin(2 * np.pi * i / self.lag)]*num_cols for i in range(self.lag + self.lead)])

        for j in range(self.lag, len(ticker_df)-self.lead+1):
            # Create a zero-padded array for the lead
            lead_zero_pad =  np.zeros(num_cols * self.lead)
            # Fill known categorical values for the lead
            for col in categorical_cols:
                for lead_step in range(1, self.lead + 1):
                    index_to_fill = (num_cols * (lead_step - 1)) + ticker_df.columns.get_loc(col)
                    lead_zero_pad[index_to_fill] = ticker_df.loc[ticker_df.index[j + lead_step - 1], col]

            # Create the sequence
            X.append(np.concatenate([
                ticker_df.iloc[j - self.lag:j].values.flatten(), # Flatten the lagged data
                lead_zero_pad # Add the zero-padded lead
                ]) + sin_vector.flatten() # Apply sinuoidal positional encoding
            )
            y.append((ticker_df.iloc[j + self.lead - 1][self.col_to_predict], date_df[j], ticker_index)) # Value to predict

        return np.array(X), np.array(y)
    
    def apply_categoricals(self, ticker_df):
        """Apply categorical indicators to the dataframe"""
        # COS and SIN
        # Time positional encoding
        ticker_df['minute_cos'] = indicators.minute_of_day_cos(ticker_df)
        ticker_df['minute_sin'] = indicators.minute_of_day_sin(ticker_df)

        ticker_df['hour_cos'] = indicators.hour_of_day_cos(ticker_df)
        ticker_df['hour_sin'] = indicators.hour_of_day_sin(ticker_df)

        ticker_df['day_cos'] = indicators.day_of_week_cos(ticker_df)
        ticker_df['day_sin'] = indicators.day_of_week_sin(ticker_df)

        ticker_df['month_cos'] = indicators.month_cos(ticker_df)
        ticker_df['month_sin'] = indicators.month_sin(ticker_df)

        cat_cols = ['minute_cos', 'minute_sin', 'hour_cos', 'hour_sin', \
                    'day_cos', 'day_sin', 'month_cos', 'month_sin']
        return cat_cols
    
    def apply_indicators(self, ticker_df):
        """Apply statistical indicators to the dataframe"""
        ticker_df['ema'] = indicators.ema(ticker_df['close'])
        ticker_df['sma'] = indicators.sma(ticker_df['close'])
        ticker_df['roc'] = indicators.roc(ticker_df['close'])
        ticker_df['percent_change'] = indicators.percent_change(ticker_df['close'])
        ticker_df['difference'] = indicators.difference(ticker_df['close'])

        ticker_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        ticker_df.dropna(inplace=True)

        indicator_cols = ['ema', 'sma', 'roc', 'percent_change', 'difference']
        return indicator_cols

    def split_and_scale_data(self, X, y):
        """Split and scale the data"""
        if not self.inference:
            # Split data
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
        """Save the data splits"""
        # Create directories if they don't exist
        os.makedirs('feature_dataframes', exist_ok=True)
        # Save the dataframe
        with open(f'feature_dataframes/{ticker}_features.pkl', 'wb') as f:
            pickle.dump(data_splits, f)

    def process_all_tickers(self):
        """Process all tickers"""
        for i, ticker in enumerate(self.tickers):
            data_splits = self.process_ticker(ticker, i)
            if data_splits:
                self.save_data_splits(ticker, data_splits)
        
        return len(data_splits['test']['X'][0])
