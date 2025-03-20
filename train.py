import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from preprocess import DataProcessor
from input_pipe import ModelMode, load_feature_dataframes
from model import TransformerModel
import wandb

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

print("Num GPUs Available: ", torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Using CPU")
    exit() # Only use GPU (preference)


def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, save_dict, wandb=False):
    best_val_loss = float('inf')
    model_name = "models/best_model.pth"
    
    patience = 5
    patience_counter = 0
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss, total_train_samples = 0.0, 0
        
        for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):  # Ignore m since global scaling removes need for per-batch mean/std
            inputs = inputs.to(device).view(-1, model.seq_len, model.features)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # MSE loss directly on scaled values
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(loss)
            
            batch_size = inputs.shape[0]
            train_loss += loss.item() * batch_size
            total_train_samples += batch_size
        
        train_loss /= total_train_samples
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss, total_val_samples = 0.0, 0
        
        with torch.no_grad():
            for inputs, targets, _ in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device).view(-1, model.seq_len, model.features)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                batch_size = inputs.shape[0]
                val_loss += loss.item() * batch_size
                total_val_samples += batch_size
        
        val_loss /= total_val_samples
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if wandb:
            # Log metrics to WandB
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss
            })
        
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

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig('plots/training_validation_loss.png')
    plt.close()

    return train_losses, val_losses

class CombinedLoss(nn.Module):
    def __init__(self, primary_criterion='nn.MSELoss()', primary_criterion_weight=0.3, direction_weight=0.7):
        super().__init__()
        self.criterion = eval(primary_criterion)  # Dynamically evaluate the string to create the loss function
        self.criterion_weight = primary_criterion_weight
        self.direction_weight = direction_weight

    def forward(self, outputs, targets):
        # criterion loss for magnitude
        criterion_loss = self.criterion(outputs, targets)
        # Directional loss: penalize wrong sign predictions
        sign_correct = (outputs * targets >= 0).float()
        direction_loss = (1 - sign_correct).mean()  # Proportion of incorrect signs
        return self.criterion_weight * criterion_loss + self.direction_weight * direction_loss


if __name__ == "__main__":

    #wandb.init(project="Stock_Transformer", entity="your_entity_name")  # Replace with your WandB entity name
    wanb = False
    lag = 60
    lead = 5
    print("Processing data...")
    tickers = ['SPY']
    provider = 'Databento'
    column_to_predict = 'percent_change'
    
    processor = DataProcessor(provider, tickers, lag=lag, lead=lead, train_split_amount=0.90, 
                              val_split_amount=0.05, col_to_predict=column_to_predict, tail=10000)
    columns = processor.process_all_tickers()
    print("Data processing complete.")

    train_loader, columns = load_feature_dataframes(tickers, ModelMode.TRAIN, batch_size=128, shuffle=True)
    val_loader, _ = load_feature_dataframes(tickers, ModelMode.EVAL, batch_size=128, shuffle=False)

    n_features = columns // (lag + lead)
    seq_len = lag + lead

    ## Hyperparameters
    embed_dim = 128
    num_heads = 16
    ff_dim = 512
    num_layers = 4
    dropout = 0.2

    model = TransformerModel(
        seq_len=seq_len,
        features=n_features,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
    
    #criterion = CombinedLoss(primary_criterion='nn.L1Loss()', primary_criterion_weight=0.3, direction_weight=0.7) # Directional loss
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    epochs = 32

    save_dict = {
        'model_state_dict': None,
        'column_to_predict': column_to_predict,
        'tickers': tickers,
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
    train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, save_dict, wanb)
    print("Training complete. Model saved.")
    
    if wandb:
        wandb.finish()