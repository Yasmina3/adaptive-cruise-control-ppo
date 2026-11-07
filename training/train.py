# train.py

from models.utils import make_highway_env
from models.acc_agent import train_acc_agent

def main():
    # Define a more challenging environment configuration
    config = {
        "lanes_count": 4,
        "vehicles_count": 50,
        "duration": 60,
        "policy_frequency": 15,
        "controlled_vehicles": 1,
        "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",
        "screen_width": 600,
        "screen_height": 150,
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "collision_reward": -1.0,
        "right_lane_reward": 0.0,
        "high_speed_reward": 0.4,
        "lane_change_reward": 1,
        "reward_speed_range": [20, 30],
        "simulation_frequency": 15,
    }

    # Create the environment
    env = make_highway_env(config)
    
    # Train the agent with longer timesteps for deeper learning
    train_acc_agent(env, total_timesteps=1_000_000)

if __name__ == "__main__":
    main()
