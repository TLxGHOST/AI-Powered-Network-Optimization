# model_training.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

def train_lstm_model(X, y, epochs=20, batch_size=16):
    """
    Train an LSTM model on preprocessed data.
    """
    try:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

        # Define LSTM model
        model = Sequential([
            LSTM(50, activation="relu", return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            LSTM(50, activation="relu"),
            Dense(1)
        ])

        # Compile and train the model
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

        # Save the model
        model.save("models/network_traffic_model.keras")
        print("✅ Model training complete and saved.")

        return model, history

    except Exception as e:
        print(f"Error during model training: {e}")
        raise