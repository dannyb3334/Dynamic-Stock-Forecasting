import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import numpy as np
import wandb
import pandas as pd
import torch

def evaluate(df, wandb=False):
    test_true = df['True Values'].values
    test_pred = df['Predictions'].values

    # Display prediction statistics
    mae = mean_absolute_error(test_true, test_pred)
    mse = mean_squared_error(test_true, test_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(test_true, test_pred)
    nll = -np.mean(np.log(np.maximum(1e-9, np.abs(test_pred - test_true))))

    # Log metrics to wandb
    if wandb:
        wandb.log({
            "Validation NLL": nll,
            "Prediction MAE": mae,
            "Prediction MSE": mse,
            "Prediction RMSE": rmse,
            "Prediction R2 Score": r2
        })
    else:
        print()

    ### Backtest of SMA crossover strategy
    active_position = False
    df['Strategy'] = np.nan
    equity =  df.iloc[0]['Close']
    df.iloc[0]['Strategy'] = equity


    # Iterate row by row of our historical data
    for index, row in df.iterrows():
        if index == 0:
            continue
        
        # change state of position
        if row['Predictions'] > 0:
            active_position = True
        else:
            active_position = False
        
        # update strategy equity
        if active_position:
            equity *= (row['True Values'] / 100) + 1
        df.loc[index, 'Strategy'] = equity

    # Print the last value of the 'Close' column
    print("Hold value:", df.iloc[-1]['Close'])
    print("Strategy value:", df.iloc[-1]['Strategy'])

    df = df[~df.index.duplicated(keep='first')].copy()
    df.reset_index(inplace=True)

    # Plot the strategy vs. close price
    plt.figure(figsize=(14, 7))
    plt.plot( df['Close'], label='Close Price')
    plt.plot( df['Strategy'], label='Strategy')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Strategy vs. Close Price')
    plt.legend()
    plt.show()

    # Plot the predictions vs. true values
    plt.figure(figsize=(14, 7))
    plt.plot(df['True Values'], label='True Values')
    plt.plot(df['Predictions'], label='Predictions')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Predictions vs. True Values')
    plt.legend()
    plt.show()


def test(model, data_loader, device):
    model.eval()
    y_pred = []
    y_true = [[],[],[],[]]
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device).view(-1, model.seq_len, model.features)
            targets = targets.numpy()
            outputs = model(inputs).detach().cpu().numpy()
            unscaled_outputs = outputs * targets[:, 4] + targets[:, 3]
            unscaled_targets = targets[:, 0] * targets[:, 4] + targets[:, 3]

            y_pred.extend(unscaled_outputs)
            y_true[0].extend(unscaled_targets)
            y_true[1].extend(targets[:, 1])
            y_true[2].extend(targets[:, 2])
            y_true[3].extend(targets[:, 5])

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    df = pd.DataFrame({
        "Predictions": y_pred,
        "True Values": y_true[0],
        "Datetime": y_true[1],
        "Close": y_true[2],
        "Ticker": y_true[3]
    })
    print(df.columns)
    #df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ns').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    #df.set_index('Datetime', inplace=True)
    return df