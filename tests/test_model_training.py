# tests/test_model_training.py
import unittest
import numpy as np
from scripts.model_training import train_lstm_model
from scripts.data_preprocessing import preprocess_data

class TestModelTraining(unittest.TestCase):
    def test_train_lstm_model(self):
        # Preprocess data
        X, y, _ = preprocess_data("data/raw/network_traffic.csv", seq_length=10)

        # Train the model
        model, history = train_lstm_model(X, y, epochs=1, batch_size=16)  # Use 1 epoch for testing

        # Check if the model is trained
        self.assertIsNotNone(model)  # Model should not be None
        self.assertIn("loss", history.history)  # Loss should be in training history

if __name__ == "__main__":
    unittest.main()