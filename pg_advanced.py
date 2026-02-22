import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import core  # Ensure core.py is in the same directory

def train(env_name='InvertedPendulum-v4', hidden_sizes=(64,64), lr=1e-2, 
          epochs=50, batch_size=5000, gamma=0.99, lam=0.97):

    # 1. Setup Environments
    # We use a non-render env for speed. 
    env = gym.make(env_name)
    
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # 2. Initialize Actor-Critic
    ac = core.MLPActorCritic(env.observation_space, env.action_space, hidden_sizes)

    # 3. Optimizers
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    v_optimizer = Adam(ac.v.parameters(), lr=lr)

    def compute_loss_pi(obs, act, adv):
        _, logp = ac.pi(obs, act)
        return -(logp * adv).mean()

    def compute_loss_v(obs, ret):
        return ((ac.v(obs) - ret)**2).mean()

    # Trackers for plotting
    history_ret = []
    history_loss = []

    def train_one_epoch():
        batch_obs, batch_acts, batch_adv, batch_ret = [], [], [], []
        batch_rets, batch_lens = [], []

        obs, info = env.reset()
        ep_rews, ep_vals = [], []
        
        while True:
            obs_torch = torch.as_tensor(obs, dtype=torch.float32)
            act, v, logp = ac.step(obs_torch)

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

                last_val = 0 if terminated else ac.v(torch.as_tensor(obs, dtype=torch.float32)).item()
                rews = np.append(ep_rews, last_val)
                vals = np.append(ep_vals, last_val)
                
                # GAE-Lambda
                deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
                adv_to_go = core.discount_cumsum(deltas, gamma * lam)
                ret_to_go = core.discount_cumsum(rews, gamma)[:-1]

                batch_adv.extend(adv_to_go)
                batch_ret.extend(ret_to_go)

                obs, info = env.reset()
                ep_rews, ep_vals = [], []

                if len(batch_obs) > batch_size:
                    break

        # Convert to tensors
        obs_t = torch.as_tensor(np.array(batch_obs), dtype=torch.float32)
        act_t = torch.as_tensor(np.array(batch_acts))
        adv_t = torch.as_tensor(np.array(batch_adv), dtype=torch.float32)
        ret_t = torch.as_tensor(np.array(batch_ret), dtype=torch.float32)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Policy Update
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(obs_t, act_t, adv_t)
        loss_pi.backward()
        pi_optimizer.step()

        # Value Update
        for _ in range(10):
            v_optimizer.zero_grad()
            loss_v = compute_loss_v(obs_t, ret_t)
            loss_v.backward()
            v_optimizer.step()

        return loss_pi.item(), np.mean(batch_rets), np.mean(batch_lens)

    # 4. Training Loop
    for i in range(epochs):
        loss, avg_ret, avg_len = train_one_epoch()
        history_ret.append(avg_ret)
        history_loss.append(loss)
        print(f"Epoch: {i+1:3d} | Loss: {loss:.4f} | Ret: {avg_ret:.2f} | Len: {avg_len:.2f}")

    # Proper Cleanup
    env.close()
    return history_ret, history_loss

if __name__ == '__main__':
    # 1. Run Training
    rets, losses = train(env_name='InvertedPendulum-v4', epochs=50)

    # 2. Create Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Return Plot
    ax1.plot(rets, color='blue', label='Avg Return')
    ax1.set_title('InvertedPendulum Performance')
    ax1.set_ylabel('Average Return')
    ax1.grid(True)
    ax1.legend()

    # Loss Plot
    ax2.plot(losses, color='red', label='Policy Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    
    # 3. Save and Show
    plt.savefig('training_results.png')
    print("\nResults saved to training_results.png")
    plt.show()
