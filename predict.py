import torch
from preprocess import DataProcessor
from input_pipe import ModelMode, load_feature_dataframes
from model import TransformerModel
from evaluate import evaluate, test

# Set the device to GPU if available, otherwise exit (only GPU is supported)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Using CPU")
    exit()  # Exit if no GPU is available

if __name__ == "__main__":
    # Load model parameters from the saved checkpoint
    model_dir = "models/best_model.pth"
    checkpoint = torch.load(model_dir, weights_only=True)

    # Extract model configuration from the checkpoint
    lag = checkpoint['lag']  # Number of lagging time steps
    lead = checkpoint['lead']  # Number of leading time steps
    n_features = checkpoint['features']  # Number of input features
    embed_dim = checkpoint['embed_dim']  # Embedding dimension
    num_heads = checkpoint['num_heads']  # Number of attention heads
    ff_dim = checkpoint['ff_dim']  # Feed-forward network dimension
    num_layers = checkpoint['num_layers']  # Number of transformer layers
    dropout = checkpoint['dropout']  # Dropout rate
    seq_len = lag + lead  # Total sequence length
    tickers = checkpoint['tickers']  # List of stock tickers
    col_to_predict = checkpoint['column_to_predict']  # Column to predict

    # Print the column to predict and tickers for debugging
    print(col_to_predict)
    print(tickers)

    # Expected that the inference set is already prepared
    # Uncomment the following lines to prepare the data if needed
    # Define the data provider (e.g., Databento)
    # provider = 'Databento'
    # processor = DataProcessor(provider, tickers, lag=lag, lead=lead, inference=True, col_to_predict='col_to_predict', tail = 1000)
    # columns = processor.process_all_tickers()

    # Load test data using the feature dataframes loader
    # ModelMode.INFERENCE ensures the data is prepared for evaluation
    test_loader, _ = load_feature_dataframes(tickers, ModelMode.INFERENCE, batch_size=128, shuffle=False)

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

    # Make predictions using the test data
    print("Making predictions...")
    results_df = test(model, test_loader, device)

    # Set the index of the results dataframe to 'Datetime' for better organization
    results_df.set_index('Datetime', inplace=True)

    # Evaluate the model's performance using the predictions
    evaluate(results_df, wandb=False)
