"""
Inspired by: https://github.com/virattt/ai-hedge-fund.git
"""
import pandas as pd
import numpy as np
import math
from technicals import (
    ema,
    atr,
    adx,
    bollinger_bands,
    rsi,
    z_score,
)

# Signals

def trend_signals(data):
    """
    Calculate a trend signal based on EMA and ADX.

    Returns:
        signal: pd.Series with trend signals
        adx: pd.DataFrame with ADX values
    """
    # Calculate EMAs for multiple timeframes
    close_df = data['close']
    ema_8 = ema(close_df, 8)
    ema_21 = ema(close_df, 21)
    ema_55 = ema(close_df, 55)

    # Calculate ADX for trend strength
    window = 14
    adx_values = []
    for i in range(len(data)):
        if i < window - 1:
            adx_values.append(np.nan)
        else:
            window_data = data.iloc[i - window + 1 : i + 1]
            adx_val, _, _ = adx(window_data, window)
            adx_values.append(adx_val.iloc[-1])
    adx_values = pd.Series(adx_values, index=data.index)

    # Determine trend direction and strength
    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55

    # Combine signals with confidence weighting
    trend_strength = adx_values / 100.0
    condition = (short_trend & medium_trend) | (~short_trend & ~medium_trend)
    signal = pd.Series(np.where(condition, trend_strength, 0.5), index=close_df.index)

    return signal, adx_values


def golden_cross_signal(data, short_window=50, long_window=200):
    """
    Calculate Golden Cross. Positive signal when 'golden', negative when 'death'.
    """
    short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['close'].ewm(span=long_window, adjust=False).mean()
    return short_ema > long_ema

def mean_reversion_signal(close_df):
    """
    Mean reversion strategy using statistical measures and Bollinger Bands.

    Returns:
        signal: pd.Series with mean reversion signals
        z_score_50: pd.Series with z-scores
        price_vs_bb: pd.Series with price vs Bollinger Bands
        rsi_14: pd.Series with RSI(14)
        rsi_28: pd.Series with RSI(28)
    """
    # Calculate z-score of price relative to moving average
    z_score_50 = z_score(close_df, 50)

    # Calculate Bollinger Bands
    bb_upper, bb_lower = bollinger_bands(close_df, window=20, num_of_std=2)

    # Calculate RSI with multiple timeframes
    rsi_14 = rsi(close_df, 14)
    rsi_28 = rsi(close_df, 28)

    # Mean reversion signals
    price_vs_bb = (close_df - bb_lower) / (bb_upper - bb_lower)

    # Create signal
    condition = ((z_score_50 < -2) & (price_vs_bb < 0.2)) | ((z_score_50 > 2) & (price_vs_bb > 0.8))
    signal = pd.Series(np.where(condition, np.minimum(abs(z_score_50) / 4, 1.0), 0.5), index=close_df.index)

    return signal, z_score_50, price_vs_bb, rsi_14, rsi_28

def momentum_signals(data):
    """
    Multi-factor momentum strategy.

    Returns:
        signal: pd.Series with momentum signals
        mom_1m: pd.Series with 1-month momentum
        mom_3m: pd.Series with 3-month momentum
        mom_6m: pd.Series with 6-month momentum
        volume_momentum: pd.Series with volume momentum
    """
    # Price momentum
    returns = data["close"].pct_change()
    mom_1m = returns.rolling(21).sum()
    mom_3m = returns.rolling(63).sum()
    mom_6m = returns.rolling(126).sum()

    # Volume momentum
    volume_ma = data["volume"].rolling(21).mean()
    volume_momentum = data["volume"] / volume_ma

    # Relative strength
    # (would compare to market/sector in real implementation)

    # Calculate momentum score
    momentum_score = (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m)

    # Volume confirmation
    volume_confirmation = volume_momentum > 1.0

    # Create signal
    condition = ((momentum_score > 0.05) & (volume_confirmation)) | ((momentum_score < -0.05) & (volume_confirmation))
    signal = pd.Series(np.where(condition, np.minimum(abs(momentum_score) * 5, 1.0), 0.5), index=data.index)

    return signal, mom_1m, mom_3m, mom_6m, volume_momentum

def volatility_signals(data):
    """
    Volatility-based trading strategy.

    Returns:
        signal: pd.Series with volatility signals
        hist_vol: pd.Series with historical volatility
        vol_regime: pd.Series with volatility regime
        vol_z_score: pd.Series with z-score of volatility
        atr_ratio: pd.Series with ATR ratio
    """
    # Calculate various volatility metrics
    returns = data['close'].pct_change()

    # Historical volatility
    hist_vol = returns.rolling(21).std() * math.sqrt(252)

    # Volatility regime detection
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma

    # Volatility mean reversion
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

    # ATR ratio
    atr_value = atr(data)
    atr_ratio = atr_value / data['close']

    # Generate signal based on volatility regime
    condition = ((vol_regime < 0.8) & (vol_z_score < -1)) | ((vol_regime > 1.2) & (vol_z_score > 1))
    signal = pd.Series(np.where(condition, np.minimum(abs(vol_z_score) / 3, 1.0), 0.5), index=data.index)

    return signal, hist_vol, vol_regime, vol_z_score, atr_ratio



def stat_arb_signals(close_df):
    """
    Statistical arbitrage signals based on price action analysis

    Returns:
        signal: pd.Series with statistical arbitrage signals
        hurst: pd.Series with Hurst exponent values
        skew: pd.Series with skewness values
        kurt: pd.Series with kurtosis values
    """
    # Calculate price distribution statistics
    returns = close_df.pct_change()

    # Skewness and kurtosis
    skew = returns.rolling(63).skew()
    kurt = returns.rolling(63).kurt()

    # Test for mean reversion using Hurst exponent

    lags = range(2, 20)
    # Add small epsilon to avoid log(0)
    tau = [max(1e-8, np.sqrt(np.std(np.subtract(close_df[lag:], close_df[:-lag])))) for lag in lags]

    # Return the Hurst exponent from linear fit
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = reg[0]  # Hurst exponent is the slope
    except (ValueError, RuntimeWarning):
        # Return 0.5 (random walk) if calculation fails
        hurst =  0.5

    # Create signal
    condition = ((hurst < 0.4) & (skew > 1)) | ((hurst < 0.4) & (skew < -1))
    signal = pd.Series(np.where(condition, (0.5 - hurst) * 2, 0.5), index=close_df.index)

    return signal, hurst, skew, kurt

def breakout_signal(data, window=20):
    """
    Breakout trading signal:
    Generates a bullish signal when the closing price breaks above the previous window high,
    and a bearish signal when it falls below the previous window low.

    Returns:
        signal: pd.Series with breakout signals (1 for bullish, -1 for bearish, 0 otherwise)
    """
    high_rolling = data['high'].rolling(window).max()
    low_rolling = data['low'].rolling(window).min()
    close = data['close']
    prev_high = high_rolling.shift(1)
    prev_low = low_rolling.shift(1)

    bullish = close > prev_high
    bearish = close < prev_low

    signal = pd.Series(0, index=data.index)
    signal[bullish] = 1
    signal[bearish] = -1
    return signal

def volume_spike_signal(data, window=20, threshold=2):
    """
    Volume spike signal:
    Signals when current volume significantly exceeds its moving average.

    Returns:
        signal: pd.Series with volume spike signals (1 when spike, 0 otherwise)
    """
    volume_ma = data['volume'].rolling(window).mean()
    spike = data['volume'] > (threshold * volume_ma)
    signal = pd.Series(0, index=data.index)
    signal[spike] = 1
    return signal