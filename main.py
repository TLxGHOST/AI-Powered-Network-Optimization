from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np

app = FastAPI()

# Load the model once at startup
try:
    model = tf.keras.models.load_model("network_traffic_model.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Prevents using an invalid model

@app.post("/predict")
async def predict(data: dict):
    try:
        # Validate input
        traffic = data.get("traffic")
        latency = data.get("latency")

        if traffic is None or latency is None:
            raise HTTPException(status_code=400, detail="Missing input data (traffic or latency)")

        # Convert input to numpy array
        input_data = np.array([[traffic, latency]])

        # Check if the model is loaded
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded correctly")

        # Make a prediction
        prediction = model.predict(input_data)
        return {"prediction": float(prediction[0][0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
