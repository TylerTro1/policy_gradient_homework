import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import gymnasium as gym
import core  # Ensure core.py is in the same folder

def watch_agent(ac, env_name):
    """Visual rollout for Part 1b."""
    print(f"\n--- Visualizing Policy on {env_name} ---")
    render_env = gym.make(env_name, render_mode="human")
    obs, info = render_env.reset()
    done = False
    ep_ret = 0
    while not done:
        # ac.act uses torch.no_grad() internally as required by 1b
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        action = ac.act(obs_t)
        obs, rew, terminated, truncated, info = render_env.step(action)
        done = terminated or truncated
        ep_ret += rew
    print(f"Visual Episode Return: {ep_ret}\n")
    render_env.close()

def train(env_name='InvertedPendulum-v4', hidden_sizes=(64,64), lr=1e-3, 
          epochs=50, batch_size=5000, gamma=0.99, lam=0.97):

    # 1. Setup Environment
    env = gym.make(env_name)
    
    # 2. Initialize Actor-Critic (Supports Part 3 Continuous/Discrete)
    ac = core.MLPActorCritic(env.observation_space, env.action_space, hidden_sizes)

    # 3. Two Optimizers (Extra Credit A)
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    v_optimizer = Adam(ac.v.parameters(), lr=lr)

    def compute_loss_pi(obs, act, adv):
        # Policy Loss using Advantage
        _, logp = ac.pi(obs, act)
        return -(logp * adv).mean()

    def compute_loss_v(obs, ret):
        # Value Loss (MSE)
        return ((ac.v(obs) - ret)**2).mean()

    def train_one_epoch():
        batch_obs, batch_acts, batch_adv, batch_ret = [], [], [], []
        batch_rets, batch_lens = [], []

        obs, info = env.reset()
        ep_rews, ep_vals = [], []
        
        while True:
            # Get action and current value estimate
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            act, v, logp = ac.step(obs_t)

            next_obs, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            batch_obs.append(obs)
            batch_acts.append(act)
            ep_rews.append(rew)
            ep_vals.append(v)

            obs = next_obs

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # GAE-Lambda calculation
                # Bootstrap if truncated; 0 if terminated
                last_val = 0 if terminated else ac.v(torch.as_tensor(obs, dtype=torch.float32)).item()
                
                rews = np.append(ep_rews, last_val)
                vals = np.append(ep_vals, last_val)
                
                # delta = r + gamma*V(s') - V(s)
                deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
                adv_to_go = core.discount_cumsum(deltas, gamma * lam)
                ret_to_go = core.discount_cumsum(rews, gamma)[:-1]

                batch_adv.extend(adv_to_go)
                batch_ret.extend(ret_to_go)

                obs, info = env.reset()
                ep_rews, ep_vals = [], []

                if len(batch_obs) > batch_size:
                    break

        # Convert batch to tensors
        obs_t = torch.as_tensor(np.array(batch_obs), dtype=torch.float32)
        act_t = torch.as_tensor(np.array(batch_acts))
        adv_t = torch.as_tensor(np.array(batch_adv), dtype=torch.float32)
        ret_t = torch.as_tensor(np.array(batch_ret), dtype=torch.float32)

        # Advantage Normalization (Stability for EC-A)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Update Policy (Actor)
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(obs_t, act_t, adv_t)
        loss_pi.backward()
        pi_optimizer.step()

        # Update Value Function (Critic) - Multiple steps for stability
        for _ in range(10):
            v_optimizer.zero_grad()
            loss_v = compute_loss_v(obs_t, ret_t)
            loss_v.backward()
            v_optimizer.step()

        return loss_pi.item(), np.mean(batch_rets), np.mean(batch_lens)

    # 4. Training Loop
    for i in range(epochs):
        loss, avg_ret, avg_len = train_one_epoch()
        print(f"Epoch: {i+1:3d} | Pi-Loss: {loss:.4f} | AvgReturn: {avg_ret:.2f} | AvgLen: {avg_len:.1f}")
        
        # Render a visual rollout after each epoch (Part 1b)
        watch_agent(ac, env_name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Switch between CartPole-v1 or InvertedPendulum-v4 here
    parser.add_argument('--env', type=str, default='InvertedPendulum-v4')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    train(env_name=args.env, epochs=args.epochs)