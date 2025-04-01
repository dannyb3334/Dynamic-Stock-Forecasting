import pandas as pd
import numpy as np
# statistical
def bollinger_bands(data, window=20, num_of_std=2):
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
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
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
    return adx, plus_di, minus_di

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

def atr(data, window=14):
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
