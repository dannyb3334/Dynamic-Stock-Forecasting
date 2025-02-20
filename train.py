import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import DataProcessor
from input_pipe import ModelMode, load_feature_dataframes
from model import TransformerModel

print("Num GPUs Available: ", torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Using CPU")
    exit() # Only use GPU (preference)

def train_model(model, train_loader, val_loader, optimizer, epochs):
    loss_fn = nn.L1Loss()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets[:, 0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets[:])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets[:, 0].to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'lag': lag,
                'lead': lead,
                'input_dim': input_dim,
                'features': features,
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'ff_dim': ff_dim,
                'num_layers': num_layers,
                'dropout': dropout
            }, "best_transformer_model.pth")
    
    print("Training complete. Best Validation Loss: ", best_val_loss)

def test_model(model, test_loader):
    model.load_state_dict(torch.load("best_transformer_model.pth")['model_state_dict'])
    model.eval()
    predictions, true_values = [], []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().detach().numpy()
            temp = np.column_stack((outputs, np.zeros_like(outputs)))  # Add a second column of zeros fro transformation
            unscaled_outputs = test_scalerY[0].inverse_transform(temp)[:, 0]
            unscaled_targets = test_scalerY[0].inverse_transform(targets)[:, 0]
            predictions.extend(unscaled_outputs)
            true_values.extend(unscaled_targets)

    return np.array(true_values), np.array(predictions)


if __name__ == "__main__":
    lag = 60
    lead = 10

    print("Processing data...")
    tickers = ['SPY']
    processor = DataProcessor(tickers, lag=lag, lead=lead, train_split_amount=0.7, val_split_amount=0.15)
    columns = processor.process_all_tickers()

    train_loader, train_scalerY = load_feature_dataframes(ModelMode.TRAIN, batch_size=128, shuffle=True)
    val_loader, val_scalerY = load_feature_dataframes(ModelMode.EVAL, batch_size=128, shuffle=False)
    
    # Hyperparameters
    input_dim = columns
    features = columns//lag
    embed_dim = 32
    num_heads = 4
    ff_dim = 512
    num_layers = 12
    dropout = 0.1

    # Instantiate model, optimizer, and train
    model = TransformerModel(input_dim, lag, features, embed_dim, num_heads, ff_dim, num_layers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 50
    print("Training model...")
    train_model(model, train_loader, val_loader, optimizer, epochs)

    test_loader, test_scalerY = load_feature_dataframes(ModelMode.INFERENCE, batch_size=128, shuffle=False)
    test_true, test_pred = test_model(model, test_loader)
    # Display validation statistics

    mae = mean_absolute_error(test_true, test_pred)
    mse = mean_squared_error(test_true, test_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(test_true, test_pred)

    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R2 Score: {r2:.4f}")
    # Plot results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(test_true, color='blue', label='True Values')
    plt.plot(test_pred, color='orange', label='Predictions')
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.title("True Values vs. Predictions")
    plt.legend()
    plt.show()
