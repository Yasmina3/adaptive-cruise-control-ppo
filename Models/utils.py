import gymnasium as gym
import highway_env
import numpy as np

def make_highway_env(config=None):
    """
    Create and configure a HighwayEnv environment.

    Args:
        config (dict, optional): A dictionary with environment parameters 
                                 like lanes_count, vehicles_count, etc.

    Returns:
        env: Configured Gym environment.
    """
    env = gym.make('highway-v0', render_mode='human')
    
    if config:
        env.unwrapped.configure(config)
    
    return env


def calculate_distance(vehicle_1, vehicle_2):
    """
    Calculate the Euclidean distance between two vehicles.
    
    Args:
        vehicle_1 (dict): Vehicle 1's state (e.g., position).
        vehicle_2 (dict): Vehicle 2's state (e.g., position).
        
    Returns:
        float: The Euclidean distance between the two vehicles.
    """
    pos_1 = vehicle_1['position']
    pos_2 = vehicle_2['position']
    return np.linalg.norm(np.array(pos_1) - np.array(pos_2))


def reward_shaping(env, done, current_vehicle, vehicles, target_distance=10):
    """
    Custom reward function encouraging safe following distance.
    
    Args:
        env (gym.Env): The environment object.
        done (bool): Whether the episode has ended.
        current_vehicle (dict): The ego vehicle's state.
        vehicles (list): List of other vehicles in the environment.
        target_distance (float): Desired safe distance.
        
    Returns:
        float: Shaped reward.
    """
    reward = 0
    closest_vehicle = None
    min_distance = float('inf')
    
    for vehicle in vehicles:
        if vehicle != current_vehicle:
            distance = calculate_distance(current_vehicle, vehicle)
            if distance < min_distance:
                min_distance = distance
                closest_vehicle = vehicle

    if closest_vehicle:
        distance_to_front = min_distance
        if distance_to_front > target_distance:
            reward += 1
        elif distance_to_front < target_distance:
            reward -= 1

    if done:
        reward -= 5

    return reward


def custom_reward(env, reward):
    base_env = env.unwrapped  

    if base_env.vehicle.crashed:
        return -10 

    # Penalize getting too close to other vehicles
    for vehicle in base_env.road.vehicles:
        if vehicle != base_env.vehicle:
            distance = np.linalg.norm(base_env.vehicle.position - vehicle.position)
            if distance < 5:
                reward -= 2

    # Penalize lane deviation
    lane_center_offset = abs(base_env.vehicle.lane_offset)
    reward -= lane_center_offset

    return reward



class RewardWrapper(gym.Wrapper):
    """
    Wraps the environment to apply the custom reward function.
    """
    def step(self, action):
     obs, reward, terminated, truncated, info = self.env.step(action)
     shaped_reward = custom_reward(self.env, reward)
     shaped_reward = float(np.mean(shaped_reward))
     return obs, shaped_reward, terminated, truncated, info
    
def wrap_highway_env(env):
    """
    Normalize observations from the environment.

    Args:
        env (gym.Env): The original environment.

    Returns:
        gym.Env: Wrapped environment.
    """
    class NormalizeState(gym.Wrapper):
        def reset(self, **kwargs):
            state, info = env.reset(**kwargs)
            return np.array(state) / 255.0, info
        
        def step(self, action):
            state, reward, terminated, truncated, info = env.step(action)
            return np.array(state) / 255.0, reward, terminated, truncated, info

    return NormalizeState(env)
