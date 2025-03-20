# Stock Prediction Transformer Model

## Project Overview

This project implements a stock prediction model using a Transformer-based architecture. The model is designed to predict a stock's movements based on historical data and various technical indicators. It includes data preprocessing, feature engineering, model training, evaluation, and prediction.

## File Descriptions

- **`model.py`**: Contains the implementation of the Transformer-based model, including the encoder and positional encoding logic.
- **`features.py`**: Defines various technical indicators and statistical features used for feature engineering.
- **`get_source_data.py`**: Provides classes for fetching stock data from different sources like Yahoo Finance or local CSV files.
- **`preprocess.py`**: Handles data preprocessing, including feature application, sequence creation, and data splitting.
- **`input_pipe.py`**: Implements a PyTorch Dataset and DataLoader for feeding data into the model during training and inference.
- **`train.py`**: Contains the training loop for the Transformer model, including loss calculation, optimization, and validation.
- **`evaluate.py`**: Provides functions for evaluating the model's predictions, calculating metrics, and backtesting strategies.
- **`predict.py`**: Loads a trained model and performs predictions on new data, followed by evaluation.

## License & Acknowledgments

This project is licensed under the MIT License.
