import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import numpy as np
import wandb
import pandas as pd
import torch

def evaluate(df, wandb=False):
    """
    Evaluate the model's predictions and compare them with true values.
    Includes metrics calculation, strategy backtesting, and visualization.

    Args:
        df (pd.DataFrame): DataFrame containing 'True Values', 'Predictions', and 'Close' columns.
        wandb (bool): Whether to log metrics to Weights & Biases (wandb).

    Returns:
        None
    """
    # Extract true and predicted values
    test_true = df['True Values'].values
    test_pred = df['Predictions'].values

    # Calculate evaluation metrics
    mae = mean_absolute_error(test_true, test_pred)
    mse = mean_squared_error(test_true, test_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(test_true, test_pred)
    nll = -np.mean(np.log(np.maximum(1e-9, np.abs(test_pred - test_true))))

    # Log metrics to wandb or print them
    metrics = {
        "Validation NLL": nll,
        "Prediction MAE": mae,
        "Prediction MSE": mse,
        "Prediction RMSE": rmse,
        "Prediction R2 Score": r2
    }
    if wandb:
        wandb.log(metrics)
    else:
        print(metrics)

    # Backtest a simple SMA crossover strategy
    active_position = False
    df['Strategy'] = np.nan
    equity = df.iloc[0]['Close']
    df.iloc[0]['Strategy'] = equity

    # Iterate through the DataFrame to simulate the strategy
    for index, row in df.iterrows():
        if index == 0:
            continue
        
        # Determine whether to hold a position based on predictions
        active_position = row['Predictions'] > 0
        
        # Update strategy equity if holding a position
        if active_position:
            equity *= (row['True Values'] / 1000) + 1  # Compounding ROI
        df.loc[index, 'Strategy'] = equity

    # Print final values for comparison
    print("Hold value:", df.iloc[-1]['Close'])
    print("Strategy value:", df.iloc[-1]['Strategy'])

    # Remove duplicate indices and reset index
    df = df[~df.index.duplicated(keep='first')].copy()
    df.reset_index(inplace=True)

    # Save strategy vs. close price plot
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['Strategy'], label='Strategy')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Strategy vs. Close Price')
    plt.legend()
    plt.savefig('plots/strategy_vs_close_price.png')
    plt.close()

    # Save predictions vs. true values plot
    plt.figure(figsize=(14, 7))
    plt.plot(df['True Values'], label='True Values')
    plt.plot(df['Predictions'], label='Predictions')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Predictions vs. True Values')
    plt.legend()
    plt.savefig('plots/predictions_vs_true_values.png')
    plt.close()


def test(model, data_loader, device):
    """
    Test the model on a given dataset and return a DataFrame with predictions and true values.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').

    Returns:
        pd.DataFrame: DataFrame containing predictions, true values, and additional metadata.
    """
    model.eval()  # Set the model to evaluation mode
    y_pred = []
    y_true = [[], [], [], []]  # To store true values and metadata

    with torch.no_grad():  # Disable gradient computation
        for inputs, targets, m in data_loader:
            # Prepare inputs and metadata
            inputs = inputs.to(device).view(-1, model.seq_len, model.features)
            targets = targets.numpy()
            m = m.numpy()

            # Get model predictions
            outputs = model(inputs).detach().cpu().numpy()

            # Unscale predictions and targets
            unscaled_outputs = outputs * m[:, 3] + m[:, 2]
            unscaled_targets = targets * m[:, 3] + m[:, 2]

            # Append predictions and true values
            y_pred.extend(unscaled_outputs)
            y_true[0].extend(unscaled_targets)
            y_true[1].extend(m[:, 0])  # Datetime
            y_true[2].extend(m[:, 1])  # Close price
            y_true[3].extend(m[:, 4])  # Ticker

    # Convert results to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Create a DataFrame with predictions and true values
    df = pd.DataFrame({
        "Predictions": y_pred,
        "True Values": y_true[0],
        "Datetime": y_true[1],
        "Close": y_true[2],
        "Ticker": y_true[3]
    })
    print(df.columns)

    return df