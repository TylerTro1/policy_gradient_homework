import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import matplotlib.pyplot as plt

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def reward_to_go(rews):
    n = len(rews)
    rtg = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtg[i] = rews[i] + (rtg[i+1] if i+1 < n else 0)
    return rtg

def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, use_reward_to_go=True):

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    def get_policy(obs):
        return Categorical(logits=logits_net(obs))

    def get_action(obs):
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    optimizer = Adam(logits_net.parameters(), lr=lr)
    
    # Store returns for plotting
    epoch_returns = []

    def train_one_epoch():
        batch_obs, batch_acts, batch_weights = [], [], []
        batch_rets, batch_lens = [], []
        obs, info = env.reset()
        done, ep_rews = False, []

        while True:
            batch_obs.append(obs.copy())
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                if use_reward_to_go:
                    batch_weights += list(reward_to_go(ep_rews))
                else:
                    batch_weights += [ep_ret] * ep_len

                obs, info = env.reset()
                done, ep_rews = False, []
                if len(batch_obs) > batch_size:
                    break

        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()
        return np.mean(batch_rets)

    for i in range(epochs):
        avg_ret = train_one_epoch()
        epoch_returns.append(avg_ret)
    
    env.close()
    return epoch_returns

def run_experiment(num_runs=5):
    all_simple_results = []
    all_rtg_results = []

    print(f"Starting experiment: {num_runs} runs per method...")
    
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs} for Simple PG...")
        all_simple_results.append(train(use_reward_to_go=False))
        
        print(f"Run {i+1}/{num_runs} for Reward-to-Go...")
        all_rtg_results.append(train(use_reward_to_go=True))

    # Calculate averages
    avg_simple = np.mean(all_simple_results, axis=0)
    avg_rtg = np.mean(all_rtg_results, axis=0)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(avg_simple, label='Simple Policy Gradient')
    plt.plot(avg_rtg, label='Reward-to-Go Policy Gradient')
    plt.xlabel('Epochs')
    plt.ylabel('Average Return')
    plt.title('Learning Curve Comparison: Simple vs Reward-to-Go')
    plt.legend()
    plt.grid(True)
    plt.savefig('part2_comparison.png')
    plt.show()

if __name__ == '__main__':
    run_experiment(num_runs=5)