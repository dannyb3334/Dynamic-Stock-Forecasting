import logging
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error

from DSIPTS.dsipts import TimeSeries
from model_definitions import define_Autoformer, define_RNN
import cv2

torch.cuda.set_per_process_memory_fraction(0.9, device=None)

# -------------------- Utility Functions --------------------

def set_logging_config():
    """
    Configures logging to write logs to a file and stdout.
    """
    file_handler = logging.FileHandler(filename='tmp.log')
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )


def show_plots():
    img1 = plt.imread('training_loss_plot.png')
    img2 = plt.imread('scatter_plot.png')
    img3 = plt.imread('prediction_plot.png')
    img4 = plt.imread('error_plot.png')

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(img1)
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Training Loss vs Val Loss')

    axes[0, 1].imshow(img2)
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Scatter Plot Prediction on Test Set')

    axes[1, 0].imshow(img3)
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Linear Plot Prediction on Test Set')

    axes[1, 1].imshow(img4)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Error margins on lags')

    plt.tight_layout()
    plt.show()


def create_time_features(dataset: pd.DataFrame):
    """
    Create time features from the Datetime column.

    Args:
        dataset (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: The dataset with time features.
    """
    dataset['day'] = dataset['Datetime'].dt.day
    dataset['month'] = dataset['Datetime'].dt.month
    dataset['hour'] = dataset['Datetime'].dt.hour

    # Normalize cyclic features using sine and cosine transformations
    # For day of the month (assuming 31 days in the longest month):
    dataset['day_sin'] = np.sin(2 * np.pi * dataset['day'] / 31)
    dataset['day_cos'] = np.cos(2 * np.pi * dataset['day'] / 31)

    # For month of the year (12 months in a year):
    dataset['month_sin'] = np.sin(2 * np.pi * dataset['month'] / 12)
    dataset['month_cos'] = np.cos(2 * np.pi * dataset['month'] / 12)

    # For hour of the day (24 hours in a day):
    dataset['hour_sin'] = np.sin(2 * np.pi * dataset['hour'] / 24)
    dataset['hour_cos'] = np.cos(2 * np.pi * dataset['hour'] / 24)

    return dataset

# -------------------- Core Functions --------------------

def load_timeseries(dataset: pd.DataFrame, past_vars: list, target_var: str) -> TimeSeries:
    """
    Load a dataset into a TimeSeries object and enrich it with covariates.

    Args:
        dataset (pd.DataFrame): The input dataset.
        past_vars (list): List of past variables to include.
        target_var (str): Target variable name.

    Returns:
        TimeSeries: A TimeSeries object with enriched data.
    """
    ts = TimeSeries('stockdata')
    ts.load_signal(
        dataset,
        target_variables=[target_var],
        past_variables=past_vars,
        check_holes_and_duplicates=False,
        future_variables=['day_sin', 'day_cos', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos']
    )
    return ts


def train_model(ts: TimeSeries, split_params: dict):
    """
    Train a model on the given TimeSeries object.

    Args:
        ts (TimeSeries): The TimeSeries object.
        split_params (dict): Parameters for train-validation split.
    """
    ts.set_verbose(True)
    ts.train_model(
        dirpath='checkpoints/',
        split_params=split_params,
        batch_size=128,
        max_epochs=25,
        gradient_clip_val=0.0,
        gradient_clip_algorithm='value',
        precision=32,
        auto_lr_find=False,
    )
    ts.losses.plot()
    plt.savefig('training_loss_plot.png')


def evaluate_model(lag: int, res: pd.DataFrame, has_quantiles: bool):
    """
    Evaluate the model using R-squared and Mean Absolute Error (MAE).

    Args:
        lag (int): Prediction lag.
        res (pd.DataFrame): Results DataFrame containing actual and predicted values.
        has_quantiles (bool): Whether quantiles are used in predictions.
    """
    if has_quantiles:
        y = res[res.lag == lag].y_median
        res[res.lag == lag].plot.scatter(x='y', y='y_median')
    else:
        y = res[res.lag == lag].y_pred
        res[res.lag == lag].plot.scatter(x='y', y='y_pred')
    
    r2 = r2_score(res[res.lag == lag].y, y)
    mae = mean_absolute_error(res[res.lag == lag].y, y)
    print(f"R-Squared: {r2:.4f}, Mean Absolute Error: {mae:.4f}")
    plt.savefig('scatter_plot.png')


def plot_error_lag(res: pd.DataFrame, has_quantiles: bool):
    pred = 'y_median' if has_quantiles else 'y_pred'
    error = res.groupby('lag').apply(lambda x: np.nanmean((x.y - x[pred]) ** 2)).reset_index().rename(columns={0: 'error'})
    print(error)
    plt.figure(figsize=(7, 11))
    plt.plot(error.lag, error.error, 'p')
    plt.plot(error.lag, error.error)
    plt.savefig('error_plot.png')


def visualize_predictions(res: pd.DataFrame, lag: int, has_quantiles: bool):
    """
    Visualize the model's predictions versus actual data.

    Args:
        res (pd.DataFrame): Results DataFrame.
        lag (int): Prediction lag.
        has_quantiles (bool): Whether quantiles are used in predictions.
    """
    
    plt.plot(res[res.lag == lag].time, res[res.lag == lag].y, label='real')
    if has_quantiles:
        plt.plot(res[res.lag == lag].time, res[res.lag == lag].y_median, label='Prediction Median')
    else:
        plt.plot(res[res.lag == lag].time, res[res.lag == lag].y_pred, label='Prediction')

    plt.title(f'Prediction on test set for lag={lag}')
    plt.legend()
    plt.savefig('prediction_plot.png')


# -------------------- Main Execution --------------------

def main(target_col: str, past_steps: int, future_steps: int, allow_logging: bool = True):

    if allow_logging:
        set_logging_config()

    # Load the dataset
    dataset = pd.read_pickle('./stock_dataset.pkl').dropna()

    # Create time features
    dataset = create_time_features(dataset)

    # Prepare the dataset
    dataset.reset_index(inplace=True)

    dataset.rename(columns={'Datetime': 'time', target_col: 'y'}, inplace=True)
    if 'index' in dataset.columns:
        dataset.drop(columns=['index'], inplace=True)
    logging.info(f'Dataset loaded with shape {dataset.shape}')

    past_vars = list(set(dataset.columns) - {'time', 'y'})
    dataset = dataset.replace([float('inf'), float('-inf')], pd.NA).dropna()
    print(dataset)
    print(dataset.shape)

    # Load the dataset into a TimeSeries object (DSIPTS)
    ts = load_timeseries(dataset, past_vars, 'y')

    has_quantiles = True
    logging.info(f'Using quantiles: {has_quantiles}')

    # Define and train the model
    ts = define_RNN(ts, past_steps, future_steps, has_quantiles)

    split_params = {
        'perc_train': 0.7,
        'perc_valid': 0.15,
        'past_steps': past_steps,
        'future_steps': future_steps,
        'shift': 0,
        'starting_point': None,
        'skip_step': 1
    }

    train_model(ts, split_params)
    logging.info('Training completed.')

    # Save the model
    ts.save(f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{target_col}")

    # Evaluate the model on test data
    res = ts.inference_on_set(batch_size=128, set='test')
    res = res.replace([float('inf'), float('-inf')], pd.NA).dropna()
    #res.to_csv('predictions.csv', index=False)
    evaluate_model(future_steps, res, has_quantiles)
    plot_error_lag(res, has_quantiles)
    visualize_predictions(res, future_steps, has_quantiles)
    model_vs_hold_return(res, future_steps, has_quantiles)
    show_plots()
    
if __name__ == '__main__':
    #past_steps=1200
    #future_steps=60
    #main('SPY_Close', past_steps=past_steps, future_steps=future_steps, allow_logging=False)
    res = pd.read_csv('predictions.csv')
    a = model_vs_hold_return(res, 10, True)