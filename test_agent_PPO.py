import gymnasium
import highway_env
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from training.agent_PPO import Agent_PPO


def evaluate_model(episode_load, printing=False):
    '''Evaluate the PPO agent after the specified number of training episodes'''
    env = gymnasium.make(
        'highway-v0',
        config={'action': {'type': 'DiscreteMetaAction'}, 'lanes_count': 3, 'ego_spacing': 1.5},
        render_mode='human' if printing else None
    )

    agent = Agent_PPO(env=env)
    agent.load_models(
        f'Models/PPO/PPO_actor_model_{episode_load}.pt',
        f'Models/PPO/PPO_critic_model_{episode_load}.pt'
    )

    state, _ = env.reset()
    state = state.reshape(-1)

    episode = 1
    returns = []
    average_reward = []
    rewards_episode = []

    while episode <= 20:
        episode_return = 0
        done, truncated = False, False
        episode_steps = 0

        while not done and not truncated:
            action = agent.act(torch.tensor(state))
            state, reward, done, truncated, _ = env.step(action)

            if printing:
                env.render()

            state = state.reshape(-1)
            episode_return += reward
            rewards_episode.append(reward)
            episode_steps += 1

        print(f'Episode {episode} | Steps: {episode_steps} | Return: {episode_return:.2f}')

        returns.append(episode_return)
        average_reward.append(np.mean(rewards_episode))
        rewards_episode = []
        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1

    env.close()
    return [returns, average_reward]


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    # === PPO Evaluation ===
    episode = 100000     
    printing = True  

    print(f'Evaluating PPO after {episode} episodes...')
    res = evaluate_model(episode, printing)

    # === Plotting ===
    f = plt.figure()
    plt.plot(res[0])
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'Return over episodes using PPO ({episode} episodes)')
    f.savefig(f'Figures/Return_over_episode_PPO_{episode}.pdf', bbox_inches='tight')

    f = plt.figure()
    plt.plot(res[1])
    plt.xlabel('Episodes')
    plt.ylabel('Average reward')
    plt.title(f'Average reward over episodes using PPO ({episode} episodes)')
    f.savefig(f'Figures/Average_reward_over_episode_PPO_{episode}.pdf', bbox_inches='tight')

    plt.show()
