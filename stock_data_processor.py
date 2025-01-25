import yfinance as yf
import pandas as pd
import os

def calculate_indicators(data: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Calculates various technical indicators for stock data.

    Args:
        data (pd.DataFrame): Stock data.
        tickers (list): List of stock tickers.
        source (str): Data source ('yfinance' or 'databendo').
        condensed (bool): Whether to flatten MultiIndex columns (for 'yfinance').

    Returns:
        pd.DataFrame: Data with technical indicators added.
    """
    for ticker in tickers:
        # Calculate moving averages
        period=14
        data[f'{ticker}_SMA'] = data[f'{ticker}_Close'].rolling(window=period).mean()
        data[f'{ticker}_EMA'] = data[f'{ticker}_Close'].ewm(span=period, adjust=False).mean()
        
        # Calculate MACD (Moving Average Convergence Divergence)
        ema_12 = data[f'{ticker}_Close'].ewm(span=12, adjust=False).mean()
        ema_26 = data[f'{ticker}_Close'].ewm(span=26, adjust=False).mean()
        data[f'{ticker}_MACD'] = ema_12 - ema_26
        data[f'{ticker}_MACD_Signal'] = data[f'{ticker}_MACD'].ewm(span=9, adjust=False).mean()

        # Calculate Stochastic Oscillator
        period=14
        low_min = data[f'{ticker}_Low'].rolling(window=period).min()
        high_max = data[f'{ticker}_High'].rolling(window=period).max()
        data[f'{ticker}_Stochastic_Oscillator'] = 100 * ((data[f'{ticker}_Close'] - low_min) / (high_max - low_min))

        # Calculate 50-day and 200-day Moving Average Cross
        data[f'{ticker}_50MA-200MA'] = (data[f'{ticker}_Close'].rolling(window=50).mean() - \
                        data[f'{ticker}_Close'].rolling(window=200).mean())
        
        # Calculate On-Balance Volume (OBV)
        data[f'{ticker}_OBV'] = (data[f'{ticker}_Volume'] * ((data[f'{ticker}_Close'].diff() > 0) * 2 - 1)).fillna(0).cumsum()

        # Calculate Accumulation/Distribution Line (ADL)
        adl = ((data[f'{ticker}_Close'] - data[f'{ticker}_Low']) - (data[f'{ticker}_High'] - data[f'{ticker}_Close'])) / (data[f'{ticker}_High'] - data[f'{ticker}_Low']) * data[f'{ticker}_Volume']
        data[f'{ticker}_ADL'] = adl.cumsum()

        # Calculate Average Directional Index (ADX)
        def calculate_adx(data, window=14):
            high = data[f'{ticker}_High']
            low = data[f'{ticker}_Low']
            close = data[f'{ticker}_Close']
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(window=window).mean()
            plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / atr)
            minus_di = abs(100 * (minus_dm.ewm(alpha=1/window).mean() / atr))
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            adx = dx.ewm(alpha=1/window).mean()
            return adx

        data[f'{ticker}_ADX'] = calculate_adx(data)

        # Calculate Aroon Indicator (25 period)
        def calculate_aroon(data, period=25):
            aroon_up = data[f'{ticker}_High'].rolling(window=period + 1).apply(lambda x: x.argmax() / period * 100, raw=True)
            aroon_down = data[f'{ticker}_Low'].rolling(window=period + 1).apply(lambda x: x.argmin() / period * 100, raw=True)
            return aroon_up, aroon_down

        data[f'{ticker}_Aroon_Up'], data[f'{ticker}_Aroon_Down'] = calculate_aroon(data)

        # Calculate Relative Strength Index (RSI) (period 14)
        delta = data[f'{ticker}_Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data[f'{ticker}_RSI_14'] = 100 - (100 / (1 + rs))

        data = data.replace([float('inf'), float('-inf')], pd.NA).dropna()

    return data

def is_valid_ticker(ticker: str) -> bool:
    """
    Validates if a ticker exists on Yahoo Finance.

    Args:
        ticker (str): Stock ticker.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        yf.Ticker(ticker).info
        return True
    except Exception as e:
        return False
    
def fetch_stock_data_yfinance(tickers: list) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance for the given list of tickers.

    Args:
        tickers (list): A list of stock tickers to fetch data for.

    Returns:
        pd.DataFrame: A DataFrame containing the stock data for each ticker,
                      with columns formatted as 'ticker_column_name' and rows with daily data.
                      Binary columns for Dividends and Stock Splits are also converted to 0/1.
    """
    dataset = yf.download(tickers, interval='1d', actions=True, group_by='ticker')
    dataset.columns = ['{}_{}'.format(ticker, col) for ticker, col in dataset.columns]
    
    # Convert binary columns to 0/1 for Dividends and Stock Splits
    binary_columns = ['Dividends', 'Stock Splits']
    for column in dataset.columns:
        if column in binary_columns:
            dataset[column] = (dataset[column] > 0.0).astype(float)

    dataset.reset_index(inplace=True)
    dataset.rename(columns={'Date': 'Datetime'}, inplace=True)
    dataset = dataset.replace([float('inf'), float('-inf')], pd.NA).dropna()
    
    return dataset

def fetch_stock_data_csv(tickers: list, paths: list) -> pd.DataFrame:
    """
    Fetch stock data from CSV files for the given tickers and corresponding file paths.

    Args:
        tickers (list): A list of stock tickers to fetch data for.
        paths (list): A list of file paths corresponding to each ticker's data.

    Returns:
        pd.DataFrame: A DataFrame containing the stock data for each ticker,
                      with columns formatted as 'ticker_column_name'.
                      Data is merged based on the Datetime index, and invalid rows are dropped.
    """
    data_frames = []
    for i in range(len(tickers)):
        ticker_data_raw = pd.read_csv(paths[i])
        ticker_data_raw['Datetime'] = pd.to_datetime(ticker_data_raw['ts_event'])
        ticker_data_raw.set_index('Datetime', inplace=True)
        ticker_data = ticker_data_raw[['open', 'high', 'low', 'close', 'volume']]
        ticker_data.columns = ['{}_{}'.format(tickers[i], col.capitalize()) for col in ticker_data.columns]
        data_frames.append(ticker_data)
    
    dataset = pd.concat(data_frames, axis=1, join='inner')
    dataset.reset_index(inplace=True)
    dataset = dataset.replace([float('inf'), float('-inf')], pd.NA).dropna()
    
    return dataset

def main(tickers: list, filenames: list = None):
    """
    Main function to fetch stock data either from Yahoo Finance or CSV files,
    calculate stock indicators, and save the processed data.

    Args:
        tickers (list): A list of stock tickers to fetch data for.
        filenames (list, optional): A list of file paths for the CSV files containing stock data.
                                    If provided, must match the number of tickers.

    Raises:
        ValueError: If a ticker is invalid or there is a mismatch between tickers and filenames.
        FileNotFoundError: If any of the specified files do not exist.
    """
    if filenames:
        assert len(tickers) == len(filenames), "Number of tickers and filenames must match."
        dataset = fetch_stock_data_csv(tickers, filenames)
    else:
        for ticker in tickers:
            if not is_valid_ticker(ticker):
                raise ValueError(f"Invalid ticker {ticker}.")

        dataset = fetch_stock_data_yfinance(tickers)

    data_with_stats = calculate_indicators(dataset, tickers)
    data_with_stats = data_with_stats.iloc[-15000:]
    data_with_stats.to_pickle('./stock_dataset.pkl')
    # data_with_stats.to_csv('./stock_dataset.csv', index=False)
    print(data_with_stats)
    print('Saved stock data to stock_dataset.pkl')

if __name__ == "__main__":
    main(['SPY'], ['./xnas-itch-20180501-20250120.ohlcv-1m.SPY.csv'])