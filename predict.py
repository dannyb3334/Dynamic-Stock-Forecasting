import numpy as np
import torch
from preprocess import DataProcessor
from input_pipe import ModelMode, load_feature_dataframes
from model import TransformerModel
import pandas as pd
from evaluate import evaluate, test
import os
#import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Using CPU")
    exit()  # Only use GPU (preference)


if __name__ == "__main__":
    # Load model parameters
    model_dir = max([os.path.join('models', f) for f in os.listdir('models') if f.endswith('.pth')], key=os.path.getctime)
    print("model_dir", model_dir)
    checkpoint = torch.load(model_dir, weights_only=True)
    lag = checkpoint['lag']
    lead = checkpoint['lead']
    n_features = checkpoint['features']
    embed_dim = checkpoint['embed_dim']
    num_heads = checkpoint['num_heads']
    ff_dim = checkpoint['ff_dim']
    num_layers = checkpoint['num_layers']
    dropout = checkpoint['dropout']
    seq_len = lag + lead
    #wandb.init(project="dynamic-stock-forecasting")

    tickers = ['SPY']
    #provider = 'Databento'
    #processor = DataProcessor(provider, tickers, lag=lag, lead=lead, inference=True, col_to_predict='percent_change', tail = 1000)
    #columns = processor.process_all_tickers()

    test_loader, _ = load_feature_dataframes(tickers, ModelMode.INFERENCE, batch_size=128, shuffle=False)

    # Instantiate model
    model = TransformerModel(
        seq_len=seq_len,
        features=n_features,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)


    model.load_state_dict(torch.load(model_dir, weights_only=True)['model_state_dict'])

    print("Making predictions...")
    results_df = test(model, test_loader, device)
    results_df.set_index('Datetime', inplace=True)

    # Evaluate model
    evaluate(results_df, wandb=False)

    # Finish the wandb run
    #wandb.finish()
