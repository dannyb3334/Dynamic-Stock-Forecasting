import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

from get_source_data import Databento
import indicators

class DataProcessor:
    def __init__(self, tickers, train_split_amount=0.8, val_split_amount=0.1, lead=2, lag=12, inference=False, col_to_predict='close'):
        self.tickers = tickers
        self.col_to_predict = col_to_predict
        self.inference = inference
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
        #ticker_df = ticker_df.head(7000)
        print(f"Length of data before adding features: {len(ticker_df)}")

        # Apply indicators to the dataframe
        categorical_cols = self.apply_indicators(ticker_df)
        assert self.col_to_predict in ticker_df.columns, f"{self.col_to_predict} is not a column in the dataframe"
        ticker_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        ticker_df.dropna(inplace=True)

        # Ensure the number of data points is a multiple of lag
        if len(ticker_df) % self.lag != 0:
            ticker_df = ticker_df[(len(ticker_df) % self.lag):]
        
        print(f"Length of data after applying features: {len(ticker_df)}")
        cols = ticker_df.columns.tolist()
        num_cols = len(cols)
        print(cols)

        # Create sequences and labels
        X, y = self.create_sequences_and_labels(ticker_df, ticker_index, num_cols, categorical_cols)
        return self.split_and_scale_data(X, y)

    def create_sequences_and_labels(self, ticker_df, ticker_index, num_cols, categorical_cols):
        X, y = [], []
        for j in range(self.lag, len(ticker_df)-self.lead+1):
            # Create a zero-padded array for the lead
            lead_zero_pad =  np.zeros(num_cols * self.lead)
            # Fill known categorical values for the lead
            for col in categorical_cols:
                for lead_step in range(1, self.lead + 1):
                    index_to_fill = (num_cols * (lead_step - 1)) + ticker_df.columns.get_loc(col)
                    lead_zero_pad[index_to_fill] = ticker_df.loc[ticker_df.index[j + lead_step - 1], col]

            X.append(np.concatenate([
                ticker_df.iloc[j - self.lag:j].values.flatten(), # Flatten the lagged data
                lead_zero_pad # Add the zero-padded lead
            ]))
            y.append((ticker_df.iloc[j + self.lead - 1][self.col_to_predict], ticker_index)) # Value to predict
        return np.array(X), np.array(y)
    
    def apply_indicators(self, ticker_df):
        print('Calculating indicators...')
        # Categorical
        # COS
        ticker_df['minute_cos'] = indicators.minute_of_day_cos(ticker_df)
        ticker_df['hour_cos'] = indicators.hour_of_day_cos(ticker_df)
        ticker_df['day_cos'] = indicators.day_of_week_cos(ticker_df)
        ticker_df['month_cos'] = indicators.month_cos(ticker_df)
        # SIN
        ticker_df['minute_sin'] = indicators.minute_of_day_sin(ticker_df)
        ticker_df['hour_sin'] = indicators.hour_of_day_sin(ticker_df)
        ticker_df['day_sin'] = indicators.day_of_week_sin(ticker_df)
        ticker_df['month_sin'] = indicators.month_sin(ticker_df)
        cat_cols = ['minute_cos', 'hour_cos', 'day_cos', 'month_cos',
                'minute_sin', 'hour_sin', 'day_sin', 'month_sin']
        # Statistical
        ticker_df['ema'] = indicators.ema(ticker_df['close'])
        ticker_df['sma'] = indicators.sma(ticker_df['close'])
        ticker_df['roc'] = indicators.roc(ticker_df['close'])
        ticker_df['percent_change'] = indicators.percent_change(ticker_df['close'])
        ticker_df['difference'] = indicators.difference(ticker_df['close'])
        ticker_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        ticker_df.dropna(inplace=True)  
        return cat_cols     

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
