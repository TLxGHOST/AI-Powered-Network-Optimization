# tests/test_data_preprocessing.py
import unittest
import numpy as np
from scripts.data_preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):
    def test_preprocess_data(self):
        # Test if the function preprocesses data correctly
        X, y, scaler = preprocess_data("data/raw/network_traffic.csv", seq_length=10)

        # Check if X and y have the correct shapes
        self.assertEqual(X.shape[0], y.shape[0])  # Number of samples should match
        self.assertEqual(X.shape[1], 10)          # Sequence length should be 10
        self.assertEqual(X.shape[2], 2)           # Number of features (traffic, latency)

        # Check if y has the correct shape
        self.assertEqual(y.shape[1], 2)           # Number of features in y

if __name__ == "__main__":
    unittest.main()