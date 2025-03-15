import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

from preprocess import DataProcessor
from input_pipe import ModelMode, load_feature_dataframes
from model import TransformerModel
import wandb
import time
from evaluate import evaluate, test
import glob
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

print("Num GPUs Available: ", torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Using CPU")
    exit() # Only use GPU (preference)

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, save_dict):
    best_val_loss = float('inf')
    model_name = "models/best_model.pth"
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    patience = 5
    patience_counter = 0
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss, total_train_samples = 0.0, 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device).view(-1, model.seq_len, model.features)
            targets = targets[:, 0].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            batch_size = inputs.shape[0]
            train_loss += loss.item() * batch_size
            total_train_samples += batch_size
        
        train_loss /= total_train_samples
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss, total_val_samples = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device).view(-1, model.seq_len, model.features)
                targets = targets[:, 0].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                batch_size = inputs.shape[0]
                val_loss += loss.item() * batch_size
                total_val_samples += batch_size
        
        val_loss /= total_val_samples
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict['model_state_dict'] = model.state_dict()
            torch.save(save_dict, model_name)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break


    
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

    return train_losses, val_losses


if __name__ == "__main__":
    # Initialize Weights and Biases
    #wandb.init(project="dynamic-stock-forecasting")

    lag = 120
    lead = 5
    print("Processing data...")
    tickers = ['SPY'] # Example
    #provider = 'Databento'
    #processor = DataProcessor(provider, tickers, lag=lag, lead=lead, train_split_amount=0.80, val_split_amount=0.15, col_to_predict='percent_change', tail=10000)
    #columns = processor.process_all_tickers()
    print("Data processing complete.")

    train_loader, columns = load_feature_dataframes(tickers, ModelMode.TRAIN, batch_size=256, shuffle=True)
    val_loader, _ = load_feature_dataframes(tickers, ModelMode.EVAL, batch_size=256, shuffle=False)

    # Calculate features correctly
    n_features = columns // (lag + lead)  # number of features per timestep
    seq_len = lag + lead

    # Hyperparameters
    embed_dim = 128
    num_heads =16
    ff_dim = 1024
    num_layers = 4
    dropout = 0.1

    model = TransformerModel(
        seq_len=seq_len,
        features=n_features,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.001)  # Reduced weight decay
    criterion = nn.MSELoss()
    epochs = 32

    # Log hyperparameters
    #wandb.config.update({
    #    "lag": lag,
    #    "lead": lead,
    #    "embed_dim": embed_dim,
    #    "num_heads": num_heads,
    #    "ff_dim": ff_dim,
    #    "num_layers": num_layers,
    #    "dropout": dropout,
    #    "learning_rate": 0.00005,
    #    "weight_decay": 0.02,
    #    "warmup_steps": warmup_steps,
    #    "total_steps": total_steps,
    #    "epochs": epochs
    #})

    save_dict = {
        'model_state_dict': None,
        'lag': lag,
        'lead': lead,
        'features': n_features,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'ff_dim': ff_dim,
        'num_layers': num_layers,
        'dropout': dropout
    }

    print("Training model...")
    train_model(model, train_loader, val_loader, optimizer, criterion, epochs, save_dict)

    #test_loader = load_feature_dataframes(tickers, ModelMode.INFERENCE, batch_size=32, shuffle=False)

    #print("Testing Model...")
    #results_df = test(model, test_loader, device)
#
    ## Evaluate model
    #evaluate(results_df, wandb=False)

    # Finish the wandb run
    #wandb.finish()
