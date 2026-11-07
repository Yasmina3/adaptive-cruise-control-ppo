import gym
import highway_env
import yaml
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import PPO  # or import DQN, TD3, etc. depending on your agent choice
from Models.acc_agent import ACCAgent
from Models.utils import make_highway_env, reward_shaping, wrap_highway_env
import os
import numpy as np
from Models.utils import RewardWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

def load_config(config_path="training/config.yaml"):
    """
    Load the configuration settings for the training.

    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration settings
    config = load_config()

    # Create the environment based on the configuration
    env_config = config.get('env', {})
    env = make_highway_env(config)
    env = wrap_highway_env(env)  
    env = RewardWrapper(env)     
    env = DummyVecEnv([lambda: env])  
    env = VecMonitor(env)             

    # Initialize the RL agent
    agent = PPO('MlpPolicy', env, verbose=1, **config['agent'])

    # Training settings
    total_timesteps = config['training']['total_timesteps']
    save_path = config['training']['save_model_path']

    # Create the directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Train the agent
    print(f"Starting training for {total_timesteps} timesteps...")
    # Create checkpoint callback to save model every X steps
    checkpoint_callback = CheckpointCallback(save_freq=25_000, save_path=save_path,
                                         name_prefix='PPO_actor_model')

# Create eval environment
    eval_env = make_highway_env(config=env_config)
    eval_env = wrap_highway_env(eval_env)

# Create evaluation callback to save best model
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_path,
                             log_path=save_path, eval_freq=25_000,
                             deterministic=True, render=False)

# Train with callbacks
    agent.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])


    # Save the trained model
    agent.save(os.path.join(save_path, 'PPO_actor_model'))

    print(f"Training completed. Model saved at {os.path.join(save_path, 'PPO_actor_model_')}")

    # Evaluate the trained model (optional)
    evaluate_model(agent, env)

def evaluate_model(agent, env, episodes=20):
    """
    Evaluate the trained agent.

    Args:
        agent (PPO): The trained PPO agent.
        env (gym.Env): The environment.
        episodes (int): Number of episodes to evaluate.
    """
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = agent.predict(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward: {episode_reward}")
    
    average_reward = np.mean(total_rewards)
    print(f"Average reward over {episodes} episodes: {average_reward}")


if __name__ == "__main__":
    main()
