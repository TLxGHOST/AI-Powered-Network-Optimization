# tests/test_reinforcement_learning.py
import unittest
import numpy as np
from scripts.reinforcement_learning import NetworkEnv, train_rl_model

class TestReinforcementLearning(unittest.TestCase):
    def test_network_env(self):
        # Test the environment
        env = NetworkEnv()
        state = env.reset()

        # Check if the state is within bounds
        self.assertTrue(50 <= state[0] <= 500)

        # Take a step and check the output
        new_state, reward, done, _ = env.step(0)  # Action: Decrease traffic
        self.assertTrue(50 <= new_state[0] <= 500)
        self.assertIsInstance(reward, float)
        self.assertFalse(done)

    def test_train_rl_model(self):
        # Test RL model training
        try:
            train_rl_model()  # Train the model
            self.assertTrue(True)  # If no error, test passes
        except Exception as e:
            self.fail(f"RL training failed: {e}")

if __name__ == "__main__":
    unittest.main()