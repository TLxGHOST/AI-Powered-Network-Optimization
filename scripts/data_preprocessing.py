# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path, seq_length=10):
    
    try:
        # Load data
        data = pd.read_csv(file_path)
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data.set_index("timestamp", inplace=True)

        # Normalize data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Create sequences for LSTM
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i : i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)

        X, y = create_sequences(data_scaled, seq_length)
        return X, y, scaler

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise