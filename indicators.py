import pandas as pd
import numpy as np

# statistical
def bollinger_bands(data, window=10, num_of_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band

def rsi(data, window=10):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def roc(data, periods=10):
    """Calculate Rate of Change."""
    roc = ((data - data.shift(periods)) / data.shift(periods)) * 100
    return roc

def adx(data, window=14):
    """Calculate Average Directional Index."""
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
    """Calculate Accumulation/Distribution Line."""
    adl = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low']) * data['volume']
    return adl

def macd(data, short_window=12, long_window=26, signal_window=9):
    """Calculate Moving Average Convergence Divergence."""
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def ema(data, window=10):
    """Calculate Exponential Moving Average."""
    return data.ewm(span=window, adjust=False).mean()

def sma(data, window=10):
    """Calculate Simple Moving Average."""
    return data.rolling(window=window).mean()

def difference(data, periods=1):
    """Calculate Difference."""
    return data.diff(periods)

def roc(data, periods=10):
    """Calculate Rate of Change."""
    roc = ((data - data.shift(periods)) / data.shift(periods)) * 100
    return roc

def percent_change(data, periods=1):
    """Calculate Percent Change."""
    return data.pct_change(periods) * 100



# categorical
def minute_of_day_cos(data):
    """Extract Minute of the day as a feature."""
    minutes = ((data.index.hour - 9) * 60) + data.index.minute - 30
    return np.cos(2 * np.pi * minutes / 390)

def hour_of_day_cos(data):
    """Extract Hour of the Day as a feature."""
    hours = data.index.hour - 9
    return np.cos(2 * np.pi * hours / 7)

def day_of_week_cos(data):
    """Extract Day of the Week as a feature."""
    days = data.index.dayofweek
    return np.cos(2 * np.pi * days / 4)

def month_cos(data):
    """Extract Month as a feature."""
    months = data.index.month - 1
    return np.cos(2 * np.pi * months / 11)

def minute_of_day_sin(data):
    """Extract Minute of the day as a feature."""
    minutes = ((data.index.hour - 9) * 60) + data.index.minute - 30
    return np.sin(2 * np.pi * minutes / 390)

def hour_of_day_sin(data):
    """Extract Hour of the Day as a feature."""
    hours = data.index.hour - 9
    return np.sin(2 * np.pi * hours / 7)

def day_of_week_sin(data):
    """Extract Day of the Week as a feature."""
    days = data.index.dayofweek
    return np.sin(2 * np.pi * days / 4)

def month_sin(data):
    """Extract Month as a feature."""
    months = data.index.month - 1
    return np.sin(2 * np.pi * months / 11)
