import math
import numpy as np
import torch
from preprocess import DataProcessor
from input_pipe import ModelMode, load_feature_dataframes
from model import TransformerModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Using CPU")
    exit()  # Only use GPU (preference)

def predict(model, data_loader, scalerY):
    model.eval()
    predictions, true_values = [], []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().detach().numpy()
            temp = np.column_stack((outputs, np.zeros_like(outputs)))  # Add a second column of zeros for transformation
            unscaled_outputs = scalerY[0].inverse_transform(temp)[:, 0]
            unscaled_targets = scalerY[0].inverse_transform(targets)[:, 0]
            predictions.extend(unscaled_outputs)
            true_values.extend(unscaled_targets)

    return np.array(true_values), np.array(predictions)

if __name__ == "__main__":
    # Load model parameters
    model_dir  = 'best_transformer_model_1M.pth'
    checkpoint = torch.load(model_dir)
    lag = checkpoint['lag']
    lead = checkpoint['lead']
    input_dim = checkpoint['input_dim']
    features = checkpoint['features']
    embed_dim = checkpoint['embed_dim']
    num_heads = checkpoint['num_heads']
    ff_dim = checkpoint['ff_dim']
    num_layers = checkpoint['num_layers']
    dropout = checkpoint['dropout']

    print("Processing data...")
    #tickers = ['SPY']
    #processor = DataProcessor(tickers, lag=lag, lead=lead,inference=True)
    #columns = processor.process_all_tickers()

    test_loader, test_scalerY = load_feature_dataframes(ModelMode.INFERENCE, batch_size=128, shuffle=False)

    # Instantiate model
    model = TransformerModel(input_dim, lag+lead, features, embed_dim, num_heads, ff_dim, num_layers, dropout).to(device)
    model.load_state_dict(torch.load(model_dir)['model_state_dict'])

    print("Making predictions...")
    test_true, test_pred = predict(model, test_loader, test_scalerY)

    # Display prediction statistics
    mae = mean_absolute_error(test_true, test_pred)
    mse = mean_squared_error(test_true, test_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(test_true, test_pred)

    print(f"Prediction MAE: {mae:.4f}")
    print(f"Prediction MSE: {mse:.4f}")
    print(f"Prediction RMSE: {rmse:.4f}")
    print(f"Prediction R2 Score: {r2:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(test_true, color='blue', label='True Values')
    plt.plot(test_pred, color='orange', label='Predictions')
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.title("True Values vs. Predictions")
    plt.legend()
    plt.show()
