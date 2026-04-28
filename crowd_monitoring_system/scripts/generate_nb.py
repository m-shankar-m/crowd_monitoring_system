import json
import os

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Crowd Predictor Framework - Model Training\n",
            "Run this notebook to orchestrate training for both the LSTM neural network and Prophet forecasting models based on the raw telemetry dataset."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import os\n",
            "import sys\n",
            "import torch\n",
            "import pickle\n",
            "from prophet.serialize import model_to_json\n",
            "from src.ml.lstm_model import CrowdLSTMModel\n",
            "from src.ml.prophet_model import ForecastModel\n",
            "\n",
            "# Ensure directories\n",
            "os.makedirs('models', exist_ok=True)\n",
            "\n",
            "print('Loading dataset...')\n",
            "df = pd.read_csv('data/crowd_data.csv')\n",
            "df['ds'] = pd.to_datetime(df['timestamp'])\n",
            "df['y'] = df['count'].rolling(window=3, min_periods=1).mean()\n",
            "\n",
            "# Truncate to most recent data for training optimization\n",
            "df_train = df.tail(10000).copy()\n",
            "print(f'Training on {len(df_train)} datapoints.')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Train & Export Prophet Model\n",
            "Prophet utilizes robust statistical mechanics to establish seasonal trajectories based on the time series."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "prophet = ForecastModel()\n",
            "print('Training Prophet...')\n",
            "prophet.train(df_train[['ds', 'y']])\n",
            "\n",
            "# Save Prophet Model\n",
            "with open('models/prophet_model.json', 'w') as f:\n",
            "    f.write(model_to_json(prophet.model))\n",
            "print('Prophet model saved successfully to models/prophet_model.json')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Train & Export LSTM Model\n",
            "The LSTM module catches high-frequency fluctuations utilizing a neural gradient network architecture."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "lstm = CrowdLSTMModel(sequence_length=10, epochs=20)\n",
            "print('Training LSTM...')\n",
            "lstm.train(df_train[['ds', 'y']])\n",
            "\n",
            "# Save LSTM Network Weights\n",
            "torch.save(lstm.model.state_dict(), 'models/lstm_weights.pth')\n",
            "\n",
            "# Save Scaler (Needed to un-normalize predictions during inference)\n",
            "with open('models/scaler.pkl', 'wb') as f:\n",
            "    pickle.dump(lstm.scaler, f)\n",
            "    \n",
            "print('LSTM weights and scaling architecture saved successfully to models/')\n"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/train_models.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated!")
