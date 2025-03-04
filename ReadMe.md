# Stock Prediction Transformer Model

## Project Title & Description

This project aims to develop a Transformer-based model for predicting stock prices using historical stock data. The model leverages advanced deep learning techniques to forecast future stock prices based on past trends and patterns. Key functionalities include training the model, making predictions (inference), and evaluating the model's performance using various financial metrics.

## Results

The following images illustrate the model's performance on the SPY stock data provided by Databento, covering the period from May 2018 to January 2025:

- **Test Data Metrics**:
    ![Test Data Metrics](Resources/test_data_metrics.png)

- **Test Data Scatter Plot (Zoomed)**:
    ![Test Data Scatter Plot (Zoomed)](Resources/test_data_scatter_zoomed.png)

- **Test Data Scatter Plot**:
    ![Test Data Scatter Plot](Resources/test_data_scatter.png)

- **Test Data Time Series**:
    ![Test Data Time Series](Resources/test_data_time_series.png)


## Features

- **Data Preprocessing**: Fetches and preprocesses historical stock data, applies various statistical indicators, and handles stock splits.
- **Transformer Model Training**: Utilizes a Transformer architecture for time-series forecasting, with customizable hyperparameters.
- **Inference**: Makes predictions on new data using the trained model.
- **Evaluation**: Assesses model performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2) score.
- **Visualization**: Generates plots to compare predicted vs actual stock prices.

## Installation

### Dependencies

The project requires the following Python packages:
- `torch`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

### Setup Instructions

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/stock-prediction-transformer.git
    cd stock-prediction-transformer
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the model, run the `train.py` script:
```sh
python train.py
```
Configuration options:
- `lag`: Number of past time steps to consider.
- `lead`: Number of future time steps to predict.
- `embed_dim`: Embedding dimension for the Transformer.
- `num_heads`: Number of attention heads.
- `ff_dim`: Feedforward network dimension.
- `num_layers`: Number of Transformer layers.
- `dropout`: Dropout rate.
- `epochs`: Number of training epochs.

### Prediction

To make predictions using the trained model, run the `predict.py` script:
```sh
python predict.py
```

## Evaluation Metrics

The model is evaluated using the following metrics:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predictions.
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **R-squared (R2) Score**: Indicates the proportion of variance in the dependent variable predictable from the independent variables.

These metrics are computed in the `evaluate_model` function in `train.py`.

## Dataset Format

The expected input format is a CSV file with columns: `time`, `open`, `high`, `low`, `close`, `volume`. The data should be preprocessed to remove holes and adjust for stock splits. The `preprocess.py` script handles these preprocessing steps.

## Results & Visualization

To visualize the results, the `train.py` and `predict.py` scripts generate plots comparing predicted vs actual stock prices. Example output:
```sh
python train.py
```
This will display a plot of training and validation loss over epochs.

## Future Improvements

- Fine-tuning hyperparameters for better performance.
- Adding alternative financial indicators.
- Expanding the model to predict multiple stocks simultaneously.
- Incorporating additional data sources for more robust predictions.

## License & Acknowledgments

This project is licensed under the MIT License.
