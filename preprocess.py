import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

from get_source_data import *
import features
from typing import List
import glob
from tqdm import tqdm

class DataProcessor:
    """DataProcessor class to handle the preprocessing of stock data for model training and inference.
    
    This class fetches stock data, applies various indicators, adjusts for stock splits, creates sequences and labels,
    splits the data into training, validation, and test sets, scales the data, and saves the processed data.
    """

    def __init__(self, provider:str, tickers:List[str], train_split_amount:float=0.8, val_split_amount:float=0.1, lead:int=5,
                 lag:int=30, inference:bool=False, col_to_predict:str='close', tail:int=0, window_size:int=15, step:int=0):
        """Initialize the DataProcessor"""
        # Check for valid input
        assert train_split_amount + val_split_amount <= 1, 'Train and validation split amounts must sum to 1 or less'
        assert window_size >= 1, 'Window size must be a positive integer'
        assert lead > 0, 'Lead must be a positive integer'
        assert lag > 0, 'Lag must be a positive integer'
        assert tail >= 0, 'Tail must be a non-negative integer'
        assert step >= 0, 'Skip must be a non-negative integer'
        provider = eval(provider)
        #assert provider in globals(), f"{provider} is not a valid class in get_source_data"
        self.step = step
        self.window_size = window_size
        self.provider = provider
        self.tickers = tickers
        self.tail = tail
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

        # Sample data on the set interval
        if self.step:
            prev_col = len(ticker_df) -1
            for col in range(len(ticker_df)-self.step, 0, -self.step):
                segment = ticker_df.iloc[col:prev_col + 1]
                # OHLCV aggregation
                ticker_df.at[prev_col, 'high'] = segment['high'].max()
                ticker_df.at[prev_col, 'low'] = segment['low'].min()
                ticker_df.at[prev_col, 'volume'] = segment['volume'].sum()
                ticker_df.at[prev_col, 'open'] = ticker_df.at[col, 'open']
                prev_col = col - 1
            ticker_df = ticker_df.iloc[self.step - 1 + len(ticker_df) % self.step:: self.step]

        # Crop the dataframe to the specified tail length
        if self.tail > 0:
            ticker_df = ticker_df.tail(self.tail)
            
        # Reset index count to 0
        ticker_df.reset_index(drop=True, inplace=True)

        # Adjust for stock splits
        self.adjust_for_stock_splits(ticker_df)

        print(f"Length of data before applying features: {len(ticker_df)}")

        # Create labels
        self.create_features(ticker_df)
        assert self.col_to_predict in ticker_df.columns and not self.col_to_predict.startswith('cat_'), \
                f"{self.col_to_predict} is not a valid feature column in the dataframe"
        
        # Extend the dataframe with zeros and fill in lead time
        # zero_rows = pd.DataFrame(0, index=np.arange(self.lead), columns=ticker_df.columns)
        # last_date = pd.to_datetime(ticker_df['date'].iloc[-1])
        # next_dates = pd.date_range(start=last_date, periods=6, freq='T').tolist()[1:]
        # zero_rows['date'] = next_dates
        # ticker_df = pd.concat([ticker_df, zero_rows], ignore_index=True)
        
        # Ensure the number of data points is a multiple of lag
        if len(ticker_df) % self.lag != 0:
            ticker_df = ticker_df[(len(ticker_df) % self.lag):]
        
        print(f"Length of data after applying features: {len(ticker_df) - self.lead}")
    

        # Separate the date column from the rest of the dataframe
        date_df = ticker_df[['date']].copy().astype(int)
        date_df = date_df['date'].tolist()
        ticker_df.drop(columns=['date'], inplace=True)

        # Separate the 'close' column for later profit calculation
        close_df = ticker_df[['close']].copy()
        close_df = close_df['close'].tolist()

        # Identify categorical columns for lead time
        categorical_cols = [col for col in ticker_df.columns if col.startswith('cat_')]

        print(f"Number of features: {len(ticker_df.columns.tolist())}")
        print(ticker_df.columns.tolist())

        # Create sequences and labels
        print(f"Creating sequences and labels")
        X, y = self.create_sequences_and_labels(ticker_df, ticker_index, categorical_cols, date_df, close_df)

        # Create the data splits
        data_splits = self.create_data_splits(X, y)
        return data_splits
    
    def adjust_for_stock_splits(self, ticker_df):
        """Adjust the stock data for stock splits"""
        # Identify the rows where the percentage change is greater than 100%
        ticker_df['temp_pct_change'] = ticker_df['close'][::-1].pct_change()[::-1]
        indices_to_check = ticker_df.index[ticker_df['temp_pct_change'] > 1].tolist()[::-1]
        # Adjust the OHLCV that preceed the split
        for idx in indices_to_check:
            split_value = int(ticker_df.iloc[idx]['temp_pct_change']) + 1
            ticker_df.loc[0:idx, ['close', 'volume', 'open', 'high', 'low']] /= split_value
            ticker_df.loc[0:idx, ['volume']] *= split_value    
        # Drop the 'temp_pct_change' column
        ticker_df.drop(columns=['temp_pct_change'], inplace=True)
    
    def create_sequences_and_labels(self, ticker_df, ticker_index, categorical_cols, date_df, close_df):
        """Create sequences and labels for the model"""
        X, Y = [], []  # Sequence and label arrays
        scaler = StandardScaler()
        window_to_avg = (self.lag + self.lead) * self.window_size # Ensure window_to_avg is a multiple of the sequence length
        print(f"Window to average: {window_to_avg}")

        # Convert pandas to numpy, keep track of column locations
        feature_cols = [col for col in ticker_df.columns if col not in categorical_cols and col != self.col_to_predict]
        feature_locs = [ticker_df.columns.get_loc(col) for col in feature_cols]
        categorical_locs = [ticker_df.columns.get_loc(col) for col in categorical_cols]
        target_loc = ticker_df.columns.get_loc(self.col_to_predict)

        zero_padded_array = np.zeros((self.lead, len(ticker_df.columns)), dtype=np.float64)
        ticker_df = ticker_df.values    

        # Step through the dataframe to create sequences, using a sliding window approach for normalization
        for j in tqdm(range(self.lag + window_to_avg, len(ticker_df) - self.lead + 1), desc=f"Processing sequences for {ticker_index}"):
            # Create zero-padded lead array
            zero_padded_lead = zero_padded_array.copy()
            # Fill in the lead time with the appropriate values
            zero_padded_lead[:, categorical_locs] = ticker_df[j:j + self.lead, categorical_locs]  
            # NumPy-based approach
            lagged_sequences_array = ticker_df[j - self.lag:j].copy()
            windowed_array = ticker_df[j - self.lag - window_to_avg:j]
            scaler.fit(windowed_array[:, feature_locs])

            lagged_sequences_array[:, feature_locs] = scaler.transform(lagged_sequences_array[:, feature_locs])
            # Standardize the target column
            mean, std = np.mean(windowed_array[:, target_loc]), np.std(windowed_array[:, target_loc], ddof=1) # Match the behavior of pandas
            lagged_sequences_array[:, target_loc] = (lagged_sequences_array[:, target_loc] - mean) / std
            # Apply the same transformation to the target value
            y = (ticker_df[j + self.lead - 1, target_loc] - mean) / std

            # Create the sequence
            X.append(np.concatenate([
                        lagged_sequences_array,
                        zero_padded_lead
                        ]).flatten())

            # Create the label
            Y.append((y, date_df[j], close_df[j + self.lead - 1], mean, std, ticker_index))
        
        # Convert to NumPy arrays
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")

        return X, Y
    
    def create_features(self, ticker_df):
        """Apply statistical and categorical features to the dataframe"""
        # Apply specified features to the dataframe
        features.apply_features(ticker_df)
        # Clean the data
        ticker_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        ticker_df.dropna(inplace=True)        

    def create_data_splits(self, X, y):
        """Split and scale the data"""
        if not self.inference:
            # Split data
            length = len(y)
            train_split = int(length * self.train_split_amount)
            val_split = int(length * self.val_split_amount)

            train_X = X[:train_split]
            train_y = y[:train_split]

            val_X = X[train_split:train_split + val_split]
            val_y = y[train_split:train_split + val_split]

            test_X = X[train_split + val_split:]
            test_y = y[train_split + val_split:]

            return {
                'train': {'X': train_X, 'y': train_y},
                'val': {'X': val_X, 'y': val_y},
                'test': {'X': test_X, 'y': test_y}
            }
        else:
            return {
                'test': {'X': X, 'y': y}
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
            del data_splits # Free up memory

if __name__ == "__main__":
    print("Example Usage")
    provider = 'Databento'
    processor = DataProcessor(provider, ['SPY'], lag=30, lead=5, train_split_amount=0.85, val_split_amount=0.15, col_to_predict='percent_change')
    columns = processor.process_all_tickers()
    print("Data processing complete.")