import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ModelMode():
    TRAIN = 0
    EVAL = 1
    INFERENCE = 2

class StockDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def load_feature_dataframes(tickers, mode, batch_size=32, shuffle=False):
    # Directory containing the feature dataframes
    feature_dataframes_dir = 'feature_dataframes'

    # List all files in the directory
    feature_files = os.listdir(feature_dataframes_dir)

    # Load each file into a dataframe
    dataframes = {'X': [], 'y': [], 'scalerY': {}}

    for file in feature_files:
        if any(ticker in file for ticker in tickers):
            file_path = os.path.join(feature_dataframes_dir, file)
            if os.path.isfile(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    if mode == ModelMode.TRAIN:
                        dataframes['X'].append(data['train']['X'])
                        dataframes['y'].append(data['train']['y'])
                        dataframes['scalerY'][data['train']['y'][-1][-1]] = data['train']['scalerY']
                    elif mode == ModelMode.EVAL:
                        dataframes['X'].append(data['val']['X'])
                        dataframes['y'].append(data['val']['y'])
                        dataframes['scalerY'][data['val']['y'][-1][-1]] = data['val']['scalerY']
                    elif mode == ModelMode.INFERENCE:
                        dataframes['X'].append(data['test']['X'])
                        dataframes['y'].append(data['test']['y'])
                        dataframes['scalerY'][data['test']['y'][-1][-1]] = data['test']['scalerY']

    dataset = StockDataset(np.concatenate(dataframes['X']), np.concatenate(dataframes['y']))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, dataframes['scalerY']
