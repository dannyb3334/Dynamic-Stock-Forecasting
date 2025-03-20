import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Enum-like class to represent different modes of the model
class ModelMode:
    TRAIN = 0
    EVAL = 1
    INFERENCE = 2

mode_to_name = {
    ModelMode.TRAIN: 'train',
    ModelMode.EVAL: 'val',
    ModelMode.INFERENCE: 'test'
}

# Custom PyTorch Dataset for stock data
class StockDataset(Dataset):
    def __init__(self, sequences, labels):
        """
        Initialize the dataset with sequences and labels.

        Args:
            sequences (numpy.ndarray): Input features.
            labels (numpy.ndarray): Labels with the first column as targets and the rest as metadata.
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)  # Convert sequences to PyTorch tensors
        self.labels = torch.tensor(labels[:, 0], dtype=torch.float32)  # Extract the first column as labels
        self.metadata = labels[:, 1:]  # Extract remaining columns as metadata
        self.columns = sequences.shape[1]  # Number of feature columns
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (sequence, label, metadata) for the given index.
        """
        return self.sequences[idx], self.labels[idx], self.metadata[idx]

# Function to load feature dataframes and create a DataLoader
def load_feature_dataframes(tickers, mode, batch_size=32, shuffle=False):
    """
    Load feature dataframes for the specified tickers and mode, and return a DataLoader.

    Args:
        tickers (list): List of stock tickers to load data for.
        mode (int): Mode of the model (TRAIN, EVAL, or INFERENCE).
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        tuple: (DataLoader, int) where DataLoader is the PyTorch DataLoader and int is the number of feature columns.
    """
    if len(tickers) > 1:
        raise ValueError("Only one ticker is currently supported.")

    # Directory containing the feature dataframes
    feature_dataframes_dir = 'feature_dataframes'

    # List all files in the directory
    feature_files = os.listdir(feature_dataframes_dir)

    # Initialize containers for input features (X) and labels (y)
    dataframes = {'X': [], 'y': []}

    # Iterate over all files in the directory and process matching tickers
    for file in feature_files:
        if any(ticker in file for ticker in tickers):
            file_path = os.path.join(feature_dataframes_dir, file)
            if not os.path.isfile(file_path):
                continue

            # Load the file using pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            name = mode_to_name.get(mode)
            if name is None:
                raise ValueError("Invalid mode. Use ModelMode.TRAIN, ModelMode.EVAL, or ModelMode.INFERENCE.")

            # Append the data to the respective lists
            dataframes['X'].append(data[name]['X'])
            dataframes['y'].append(data[name]['y'])
            break # Only process the first matching file, only one ticker premitted

    # Create a StockDataset instance
    dataset = StockDataset(np.concatenate(dataframes['X']), np.concatenate(dataframes['y']))
    
    # Create a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, dataset.columns
