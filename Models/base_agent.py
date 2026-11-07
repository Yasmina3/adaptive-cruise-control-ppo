from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

def create_agent(env, policy='MlpPolicy', learning_rate=1e-3, save_path='./models/PPO'):
    """
    Create and return a PPO agent with basic configuration.
    
    Args:
        env: The Gym environment.
        policy: The policy type, e.g., 'MlpPolicy' or 'CnnPolicy'.
        learning_rate: Learning rate for the optimizer.
        save_path: Path to save model checkpoints.
        
    Returns:
        model: PPO agent ready for training.
        checkpoint_callback: Callback to save checkpoints during training.
    """
    # Initialize PPO agent
    model = PPO(policy, env, verbose=1, learning_rate=learning_rate)

    # Create checkpoint callback to save every 25,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=save_path,
        name_prefix='PPO_actor_model_'
    )
    
    return model, checkpoint_callback
