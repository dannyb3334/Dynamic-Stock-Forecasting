import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
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

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs):
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets[:, 0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets[:])
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets[:, 0].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the model with the best validation loss
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

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

def test_model(model, test_loader):
    """Test the model"""
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

def evaluate_model(test_true, test_pred):
    """Evaluate the model"""
    mae = mean_absolute_error(test_true, test_pred)
    mse = mean_squared_error(test_true, test_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(test_true, test_pred)
    nll = -np.mean(np.log(np.maximum(1e-9, np.abs(test_pred - test_true))))

    print(f"Validation NLL: {nll:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R2 Score: {r2:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(test_true, test_pred, color='orange', label='Predictions vs True Values')
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True Values vs. Predictions")
    plt.legend()
    plt.show()

def get_scheduler(optimizer, warmup_steps=500, total_steps=5000):
    """ Warm-up and cosine decay scheduler """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warm-up
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item()))
    return LambdaLR(optimizer, lr_lambda)

if __name__ == "__main__":
    lag = 30
    lead = 5

    print("Processing data...")
    tickers = ['SPY'] # Example
    processor = DataProcessor(tickers, lag=lag, lead=lead, train_split_amount=0.7, val_split_amount=0.15, col_to_predict='percent_change')
    columns = processor.process_all_tickers()

    train_loader, train_scalerY = load_feature_dataframes(ModelMode.TRAIN, batch_size=128, shuffle=True)
    val_loader, val_scalerY = load_feature_dataframes(ModelMode.EVAL, batch_size=128, shuffle=False)
    
    # Hyperparameters
    embed_dim = 128  # 16 dims per head 
    num_heads = 8
    ff_dim = 512
    num_layers = 6
    dropout = 0.3

    input_dim = columns
    features = columns//(lag+lead)

    # Instantiate model
    model = TransformerModel(input_dim, lag+lead, features, embed_dim, num_heads, ff_dim, num_layers, dropout).to(device)

    # Optimizer: AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.02)

    # Scheduler
    warmup_steps = 500
    total_steps = 5000
    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)

    # Loss function
    criterion = nn.L1Loss()

    epochs = 32

    print("Training model...")
    train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs)

    test_loader, test_scalerY = load_feature_dataframes(ModelMode.INFERENCE, batch_size=128, shuffle=False)
    test_true, test_pred = test_model(model, test_loader)

    # Evaluate the model
    evaluate_model(test_true, test_pred)
