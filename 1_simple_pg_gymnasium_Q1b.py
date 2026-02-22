import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000):

    # 1. Main training env (no rendering)
    env = gym.make(env_name)
    
    # 2. SEPARATE rendering env for Part 1b
    # This environment is only used for the visual rollout
    render_env = gym.make(env_name, render_mode="human")

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    optimizer = Adam(logits_net.parameters(), lr=lr)

    # NEW: Visual Evaluation Function for Part 1b
    def watch_agent():
        print("--- Visualizing current policy ---")
        obs, info = render_env.reset()
        done = False
        ep_ret = 0
        while not done:
            # Use torch.no_grad() as requested
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                act = get_action(obs_tensor)
            
            obs, rew, terminated, truncated, info = render_env.step(act)
            done = terminated or truncated
            ep_ret += rew
        print(f"Visual Episode Return: {ep_ret}")

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
        return batch_loss, batch_rets, batch_lens

    # Main Training Loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
        
        # After each epoch, watch the agent perform!
        watch_agent()

    render_env.close()
    env.close()

if __name__ == '__main__':
    train()
