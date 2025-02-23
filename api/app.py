# api/app.py
from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
import logging

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model once at startup
try:
    model = tf.keras.models.load_model("A:\\DISH5G\\DEEPSEEK\\models\\network_traffic_model.keras")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Prevents using an invalid model

@app.post("/predict")
async def predict(data: dict):
   
    try:
        logging.info(f"Received input data: {data}")

        # Validate input
        traffic = data.get("traffic")
        latency = data.get("latency")

        if traffic is None or latency is None:
            raise HTTPException(status_code=400, detail="Missing input data (traffic or latency)")

        # Convert input to numpy array with the correct shape
        input_data = np.array([[traffic, latency]])  # Shape: (1, 2)
        input_data = np.expand_dims(input_data, axis=0)  # Shape: (1, 1, 2)

        logging.info(f"Reshaped input data: {input_data}")

        # Check if the model is loaded
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded correctly")

        # Make a prediction
        prediction = model.predict(input_data)
        # prediction= scaler.inverse_transform(prediction)
        logging.info(f"Prediction: {prediction}")

        return {"prediction": float(prediction[0][0])}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))