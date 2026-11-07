from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import highway_env
from .hyper import PPO_HYPERPARAMS

class ACCAgent:
    """
    ACC Agent for Adaptive Cruise Control using Reinforcement Learning.
    """

class ACCAgent:
    def __init__(self, env):
        self.env = DummyVecEnv([lambda: env])
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            **PPO_HYPERPARAMS
        )
    def train(self, total_timesteps):
        """
        Train the agent.

        Args:
            total_timesteps (int): The number of timesteps to train the agent.
        """
        print("Starting training...")
        self.model.learn(total_timesteps=total_timesteps)
        print("Training complete.")

    def save(self, path):
        """
        Save the trained agent model to the specified path.

        Args:
            path (str): Path where the model should be saved.
        """
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path):
        """
        Load a pre-trained agent model.

        Args:
            path (str): Path from where the model should be loaded.
        """
        self.model = PPO.load(path)
        print(f"Model loaded from {path}")

    def predict(self, state):
        """
        Predict an action based on the current state.

        Args:
            state (numpy array): The current state of the environment.

        Returns:
            action (int): The action to take.
        """
        action, _ = self.model.predict(state)
        return action
