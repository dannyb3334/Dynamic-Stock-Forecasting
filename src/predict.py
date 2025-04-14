import torch
from preprocess import DataProcessor
from input_pipe import ModelMode, format_feature_dataframes
from model import TransformerModel
from evaluate import predict
import matplotlib.pyplot as plt
import datetime
import pandas as pd
# Set the device to GPU if available, otherwise exit (only GPU is supported)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Using CPU")
    exit()  # Exit if no GPU is available

if __name__ == "__main__":
    # Load model parameters from the saved checkpoint
    model_dir = "models/best_model.pth"
    checkpoint = torch.load(model_dir, weights_only=True)
    # Extract model configuration
    lag = checkpoint['lag']
    lead = checkpoint['lead']
    n_features = checkpoint['features']
    embed_dim = checkpoint['embed_dim']
    num_heads = checkpoint['num_heads']
    ff_dim = checkpoint['ff_dim']
    num_layers = checkpoint['num_layers']
    dropout = checkpoint['dropout']
    seq_len = lag + lead
    tickers = checkpoint['tickers']
    col_to_predict = checkpoint['column_to_predict']

    # Instantiate the Transformer model with the loaded parameters
    model = TransformerModel(
        seq_len=seq_len,
        features=n_features,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    # Load the model's state dictionary from the checkpoint
    model.load_state_dict(torch.load(model_dir, weights_only=True)['model_state_dict'])


    provider = 'YahooFinance'
    total_predicitons = pd.DataFrame(columns=['Datetime', 'Predictions', 'Close', 'Ticker'])
    
    # Modify for prefered timedelta
    minute = None
    while datetime.datetime.now().hour < 23: # Loop until market close
        if datetime.datetime.now().minute != minute:
            minute = datetime.datetime.now().minute
            processor = DataProcessor(provider, tickers, lag=lag, lead=lead, inference=True, unknown_y=True, col_to_predict=col_to_predict)
            columns = processor.process_all_tickers()
            test_loader, _ = format_feature_dataframes(tickers, ModelMode.INFERENCE, batch_size=1, shuffle=False)

            results_df = predict(model, test_loader, device)
            total_predicitons = pd.concat([total_predicitons, results_df], ignore_index=True)
            # Plotting the predictions
            plt.clf()  # Clear the previous figure
            plt.figure(figsize=(12, 6))
            plt.plot(total_predicitons['Datetime'], total_predicitons['Predictions'], marker='o', linestyle='-', color='b')
            plt.title('Predictions over time')
            plt.xlabel('Date')
            plt.ylabel('Prediction')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(1)  # Pause to allow the plot to update
