import pandas as pd
import numpy as np

# statistical
def bollinger_bands(data, window=10, num_of_std=2):
    """
    Calculate Bollinger Bands.
    """
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band

def rsi(data, window=10):
    """
    Calculate Relative Strength Index.
    """
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def roc(data, periods=10):
    """
    Calculate Rate of Change.
    """
    roc = ((data - data.shift(periods)) / data.shift(periods)) * 100
    return roc

def adx(data, window=14):
    """
    Calculate Average Directional Index.
    """
    high = data['high']
    low = data['low']
    close = data['close']
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

def adl(data):
    """
    Calculate Accumulation/Distribution Line.
    """
    adl = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low']) * data['volume']
    return adl

def macd(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculate Moving Average Convergence Divergence.
    """
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def ema(data, window=10):
    """
    Calculate Exponential Moving Average.
    """
    return data.ewm(span=window, adjust=False).mean()

def sma(data, window=10):
    """
    Calculate Simple Moving Average.
    """
    return data.rolling(window=window).mean()

def difference(data, periods=1):
    """
    Calculate Difference.
    """
    return data.diff(periods)

def percent_change(data, periods=1):
    """
    Calculate Percent Change.
    """
    return data.pct_change(periods) * 1000

def log_percent_change(data, periods=1):
    """
    Calculate Log Return.
    """
    return np.log(data.pct_change(periods) + 1) * 1000

def momentum(data, window=10):
    """
    Calculate Momentum.
    """
    return data.diff(window)

def volatility(data, window=10):
    """
    Calculate Volatility.
    """
    return data.rolling(window=window).std()

def skewness(data, window=10):
    """
    Calculate Skewness.
    """
    return data.rolling(window=window).skew()

def kurtosis(data, window=10):
    """
    Calculate Kurtosis.
    """
    return data.rolling(window=window).kurtosis()

def z_score(data, window=10):
    """
    Calculate Z-Score.
    """
    mean = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    return (data - mean) / std

def cci(data, window=20):
    """
    Calculate Commodity Channel Index.
    """
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    sma = typical_price.rolling(window=window).mean()
    mad = (typical_price - sma).abs().rolling(window=window).mean()
    cci = (typical_price - sma) / (0.015 * mad)
    return cci

def williams_r(data, window=14):
    """
    Calculate Williams %R.
    """
    highest_high = data['high'].rolling(window=window).max()
    lowest_low = data['low'].rolling(window=window).min()
    williams_r = (highest_high - data['close']) / (highest_high - lowest_low) * -100
    return williams_r

def stochastic_oscillator(data, k_window=14, d_window=3):
    """
    Calculate Stochastic Oscillator.
    """
    lowest_low = data['low'].rolling(window=k_window).min()
    highest_high = data['high'].rolling(window=k_window).max()
    k = (data['close'] - lowest_low) / (highest_high - lowest_low) * 100
    d = k.rolling(window=d_window).mean()
    return k, d

def average_true_range(data, window=14):
    """
    Calculate Average True Range.
    """
    high = data['high']
    low = data['low']
    close = data['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

# categorical
def minute_of_day_cos(data):
    """
    Extract Minute of the day as a feature.
    """
    minutes = ((data.index.hour - 9) * 60) + data.index.minute - 30
    return np.cos(2 * np.pi * minutes / 390)

def hour_of_day_cos(data):
    """
    Extract Hour of the Day as a feature.
    """
    hours = data.index.hour - 9
    return np.cos(2 * np.pi * hours / 7)

def day_of_week_cos(data):
    """
    Extract Day of the Week as a feature.
    """
    days = data.index.dayofweek
    return np.cos(2 * np.pi * days / 4)

def month_cos(data):
    """
    Extract Month as a feature.
    """
    months = data.index.month - 1
    return np.cos(2 * np.pi * months / 11)

def minute_of_day_sin(data):
    """
    Extract Minute of the day as a feature.
    """
    minutes = ((data.index.hour - 9) * 60) + data.index.minute - 30
    return np.sin(2 * np.pi * minutes / 390)

def hour_of_day_sin(data):
    """
    Extract Hour of the Day as a feature.
    """
    hours = data.index.hour - 9
    return np.sin(2 * np.pi * hours / 7)

def day_of_week_sin(data):
    """
    Extract Day of the Week as a feature.
    """
    days = data.index.dayofweek
    return np.sin(2 * np.pi * days / 4)

def month_sin(data):
    """
    Extract Month as a feature.
    """
    months = data.index.month - 1
    return np.sin(2 * np.pi * months / 11)

def golden_cross(data, short_window=50, long_window=200):
    """
    Calculate Golden Cross.
    """
    short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['close'].ewm(span=long_window, adjust=False).mean()
    return short_ema > long_ema

def death_cross(data, short_window=50, long_window=200):
    """
    Calculate Death Cross.
    """
    short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['close'].ewm(span=long_window, adjust=False).mean()
    return short_ema < long_ema

def apply_features(ticker_df):
    """
    Apply selected features to the ticker_df.
    """
    # Statistical features
    close = ticker_df['close']
    ticker_df['roc'] = roc(close)
    ticker_df['difference'] = difference(close)
    ticker_df['percent_change'] = percent_change(close)
    ticker_df['log_return'] = log_percent_change(close)
    ticker_df['sma'] = sma(close)
    ticker_df['ema'] = ema(close)
    ticker_df['momentum'] = momentum(close)
    ticker_df['cci'] = cci(ticker_df)
    ticker_df['golden_cross'] = golden_cross(ticker_df)
    ticker_df['death_cross'] = death_cross(ticker_df)
    k, d = stochastic_oscillator(ticker_df)
    ticker_df['stochastic_k'] = k
    ticker_df['stochastic_d'] = d
    ticker_df['atr'] = average_true_range(ticker_df)

    return ticker_df

def apply_categorical_features(ticker_df):
    """
    Apply categorical features to the ticker_df.
    """
    # Categorical features
    # COS and SIN
    # Time positional encoding
    ticker_df.set_index('date', inplace=True)
    ticker_df['cat_minute_cos'] = minute_of_day_cos(ticker_df)
    ticker_df['cat_minute_sin'] = minute_of_day_sin(ticker_df)

    ticker_df['cat_hour_cos'] = hour_of_day_cos(ticker_df)
    ticker_df['cat_hour_sin'] = hour_of_day_sin(ticker_df)

    ticker_df['cat_day_cos'] = day_of_week_cos(ticker_df)
    ticker_df['cat_day_sin'] = day_of_week_sin(ticker_df)

    ticker_df['cat_month_cos'] = month_cos(ticker_df)
    ticker_df['cat_month_sin'] = month_sin(ticker_df)
    ticker_df.reset_index(inplace=True)

    return ticker_df