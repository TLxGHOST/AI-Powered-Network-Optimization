# reinforcement_learning.py
import gym
import numpy as np
from stable_baselines3 import DQN

class NetworkEnv(gym.Env):
    def __init__(self):
        self.state = np.random.randint(50, 500)  # Initial traffic load
        self.action_space = gym.spaces.Discrete(3)  # Increase, Decrease, Maintain
        self.observation_space = gym.spaces.Box(low=50, high=500, shape=(1,), dtype=np.int32)

    def step(self, action):
        if action == 0:
            self.state = max(50, self.state - 10)  # Reduce traffic load
        elif action == 1:
            self.state = min(500, self.state + 10)  # Increase traffic load

        reward = -abs(self.state - 250)  # Ideal load is 250
        done = False
        return np.array([self.state]), reward, done, {}

    def reset(self):
        self.state = np.random.randint(50, 500)
        return np.array([self.state])

def train_rl_model():
    """
    Train a DQN model for bandwidth optimization.
    """
    try:
        env = NetworkEnv()
        model = DQN("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)
        model.save("models/bandwidth_optimizer")
        print("✅ Reinforcement Learning Model Saved.")
    except Exception as e:
        print(f"Error during RL training: {e}")
        raise