import numpy as np

# categoricals
def minute_of_day_cos(data):
    """
    Extract Minute of the day as a periodical categorical feature.
    """
    minutes = ((data.index.hour - 9) * 60) + data.index.minute - 30
    return np.cos(2 * np.pi * minutes / 390)

def hour_of_day_cos(data):
    """
    Extract Hour of the Day as a periodical categorical feature.
    """
    hours = data.index.hour - 9
    return np.cos(2 * np.pi * hours / 7)

def day_of_week_cos(data):
    """
    Extract Day of the Week as a periodical categorical feature.
    """
    days = data.index.dayofweek
    return np.cos(2 * np.pi * days / 4)

def month_cos(data):
    """
    Extract Month as a periodical categorical feature.
    """
    months = data.index.month - 1
    return np.cos(2 * np.pi * months / 11)

def minute_of_day_sin(data):
    """
    Extract Minute of the day as a periodical categorical feature.
    """
    minutes = ((data.index.hour - 9) * 60) + data.index.minute - 30
    return np.sin(2 * np.pi * minutes / 390)

def hour_of_day_sin(data):
    """
    Extract Hour of the Day as a periodical categorical feature.
    """
    hours = data.index.hour - 9
    return np.sin(2 * np.pi * hours / 7)

def day_of_week_sin(data):
    """
    Extract Day of the Week as a periodical categorical feature.
    """
    days = data.index.dayofweek
    return np.sin(2 * np.pi * days / 4)

def month_sin(data):
    """
    Extract Month as a periodical categorical feature.
    """
    months = data.index.month - 1
    return np.sin(2 * np.pi * months / 11)