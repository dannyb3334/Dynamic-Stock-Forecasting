from categoricals import *
from technicals import *
from trading_signals import *

def apply_features(ticker_df):
    """
    Apply selected features to the ticker_df.
    """
    # Statistical features
    close = ticker_df['close']
    ticker_df['log_return'] = log_percent_change(close)
    ticker_df['rsi_14'] = rsi(close, 14)
    ticker_df['rsi_28'] = rsi(close, 28)
    ticker_df['bb_upper'], ticker_df['bb_lower'], = bollinger_bands(close)
    ticker_df['percent_change'] = percent_change(close)

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
