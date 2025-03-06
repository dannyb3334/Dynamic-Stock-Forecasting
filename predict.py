import numpy as np
import torch
from preprocess import DataProcessor
from input_pipe import ModelMode, load_feature_dataframes
from model import TransformerModel
import pandas as pd
from evaluate import evaluate
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Using CPU")
    exit()  # Only use GPU (preference)

def predict(model, data_loader, scalerY):
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().detach().numpy()
            temp = np.column_stack((outputs, np.zeros_like(outputs), np.zeros_like(outputs)))  # Add a second column of zeros for transformation
            unscaled_outputs = scalerY[0].inverse_transform(temp)[:, 0]
            unscaled_targets = scalerY[0].inverse_transform(targets)
            y_pred.extend(unscaled_outputs)
            y_true.extend(unscaled_targets)

    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true = np.array(y_true)

    merged = np.hstack((y_pred, y_true))
    df = pd.DataFrame(merged, columns=["Predictions", "True Values", "Datetime", "Ticker"])
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ns').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    return df



if __name__ == "__main__":
    # Load model parameters
    model_dir  = 'models/best_transformer_model_1M_30lag.pth'
    print("model_dir", model_dir)
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
    wandb.init(project="dynamic-stock-forecasting")


    tickers = ['NVDA']
    provider = 'YahooFinance'
    processor = DataProcessor(provider, tickers, lag=lag, lead=lead, inference=True, col_to_predict='percent_change')
    columns = processor.process_all_tickers()

    test_loader, test_scalerY = load_feature_dataframes(tickers, ModelMode.INFERENCE, batch_size=256, shuffle=False)

    # Instantiate model
    model = TransformerModel(input_dim, lag+lead, features, embed_dim, num_heads, ff_dim, num_layers, dropout).to(device)
    model.load_state_dict(torch.load(model_dir, weights_only=True)['model_state_dict'])

    print("Making predictions...")
    results_df = predict(model, test_loader, test_scalerY)

    # Evaluate model
    evaluate(results_df, 114.59)

    # Finish the wandb run
    wandb.finish()
