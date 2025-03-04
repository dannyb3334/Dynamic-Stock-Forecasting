import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import numpy as np
import wandb

def evaluate(df):
    test_true = df['True Values'].values
    test_pred = df['Predictions'].values

    # Display prediction statistics
    mae = mean_absolute_error(test_true, test_pred)
    mse = mean_squared_error(test_true, test_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(test_true, test_pred)
    nll = -np.mean(np.log(np.maximum(1e-9, np.abs(test_pred - test_true))))

    # Log metrics to wandb
    wandb.log({
        "Validation NLL": nll,
        "Prediction MAE": mae,
        "Prediction MSE": mse,
        "Prediction RMSE": rmse,
        "Prediction R2 Score": r2
    })

    # Plot results
    print(f"Start: {df['Datetime'].iloc[0]}")
    print(f"End: {df['Datetime'].iloc[-1]}")
    plt.figure(figsize=(10, 6))
    plt.plot(df['True Values'], color='blue', label='True Values')
    plt.plot(df['Predictions'], color='orange', label='Predictions')
    plt.xlabel("Datetime")
    plt.ylabel("Values")
    plt.title("True Values vs. Predictions")
    plt.legend()
    wandb.log({"True Values vs Predictions": plt})

    # Initialize investments
    initial = 100
    strategy_investment = hold_investment = only_positive_investment = initial
    correct_positive = tot_positive = 0

    # Count total positives
    tot_positive = (df['True Values'] > 0).sum()

    # Compute holding investment with compounding returns
    hold_investment *= (1 + (df['True Values'] / 100)).prod()

    # Compute strategy performance
    positive_predictions = df[df['Predictions'] > 0]
    if not positive_predictions.empty:
        correct_positive = (positive_predictions['True Values'] > 0).sum()
        strategy_investment *= (1 + (positive_predictions['True Values'] / 100)).prod()

    # Compute only positive values compounding
    only_positive_values = df[df['True Values'] > 0]
    if not only_positive_values.empty:
        only_positive_investment *= (1 + (only_positive_values['True Values'] / 100)).prod()

    # Print statistics
    print(f"Strategy correct positive: {correct_positive}")
    print(f"Total positive: {tot_positive}")
    print(f"Accuracy: {correct_positive / tot_positive:.2%}")

    print(f"Final return on strategy investment: ${strategy_investment:.2f}")
    print(f"Final return on holding investment: ${hold_investment:.2f}")
    print(f"Greatest investment possible: ${only_positive_investment:.2f}")
