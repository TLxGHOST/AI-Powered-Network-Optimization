# tests/test_api.py
import unittest
from fastapi.testclient import TestClient
from api.app import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        # Create a test client for the FastAPI app
        self.client = TestClient(app)

    def test_predict_endpoint(self):
        # Test the /predict endpoint
        response = self.client.post(
            "/predict",
            json={"traffic": 200, "latency": 50}
        )

        # Check if the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())

    def test_predict_endpoint_missing_data(self):
        # Test the /predict endpoint with missing data
        response = self.client.post(
            "/predict",
            json={"traffic": 200}  # Missing latency
        )

        # Check if the response is a 400 error
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())

if __name__ == "__main__":
    unittest.main()