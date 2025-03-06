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
from predict import predict
from evaluate import evaluate
import glob

print("Num GPUs Available: ", torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Using CPU")
    exit() # Only use GPU (preference)

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, save_dict):
    best_val_loss = float('inf')
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    model_name = f"models/model_{timestamp}.pth"
    
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
            save_dict['model_state_dict'] = model.state_dict()
            torch.save(save_dict, model_name)
    
    print("Training complete. Best Validation Loss: ", best_val_loss)

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    wandb.log({"Training and Validation Loss Over Epochs": plt})

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
    # Initialize Weights and Biases
    wandb.init(project="dynamic-stock-forecasting")

    lag = 60
    lead = 5
    print("Processing data...")
    tickers = ['SPY'] # Example
    provider = 'Databento'
    processor = DataProcessor(provider, tickers, lag=lag, lead=lead, train_split_amount=0.80, val_split_amount=0.15, col_to_predict='percent_change')
    columns = processor.process_all_tickers()
    print("Data processing complete.")

    train_loader, train_scalerY = load_feature_dataframes(tickers, ModelMode.TRAIN, batch_size=128, shuffle=True)
    val_loader, val_scalerY = load_feature_dataframes(tickers, ModelMode.EVAL, batch_size=128, shuffle=False)
    
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

    epochs = 2

    # Log hyperparameters
    wandb.config.update({
        "lag": lag,
        "lead": lead,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "ff_dim": ff_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": 0.00005,
        "weight_decay": 0.02,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "epochs": epochs
    })

    save_dict = {
        'model_state_dict': None,
        'lag': lag,
        'lead': lead,
        'input_dim': input_dim,
        'features': features,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'ff_dim': ff_dim,
        'num_layers': num_layers,
        'dropout': dropout
    }

    print("Training model...")
    train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, save_dict)

    test_loader, test_scalerY = load_feature_dataframes(tickers, ModelMode.INFERENCE, batch_size=256, shuffle=False)

    print("Testing Model...")
    results_df = predict(model, test_loader, test_scalerY)

    # Evaluate model
    evaluate(results_df, test_loader[0][1])

    # Finish the wandb run
    wandb.finish()
