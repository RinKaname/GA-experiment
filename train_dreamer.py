import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

from hammurabi_env import DemocraticHammurabi
from micro_dreamer import WorldModel, Actor, ReplayBuffer, normalize_state

def train_micro_dreamer(num_epochs=300, collect_steps=100, batch_size=64, imagination_horizon=3):
    print("Initializing Micro-Dreamer Environment...")
    env = DemocraticHammurabi()

    world_model = WorldModel()
    actor = Actor()

    wm_optimizer = optim.Adam(world_model.parameters(), lr=1e-3)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)

    buffer = ReplayBuffer()

    print("Pre-filling buffer...")
    state = env.reset()
    for _ in range(500):
        raw_action = [np.random.uniform(-1, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)]

        action_land = raw_action[0]
        action_feed = raw_action[1] * (state[1] * 20 * 1.5) / max(1.0, state[2])
        action_plant = raw_action[2] * min(state[3], state[1] * 10) / max(1.0, state[2])
        env_action = [action_land, action_feed, action_plant]

        next_state, reward, done, info = env.step(env_action)
        buffer.push(state, raw_action, reward, next_state)

        state = next_state
        if done:
            state = env.reset()

    print(f"Starting Micro-Dreamer Training for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(num_epochs):
        state = env.reset()
        epoch_reward = 0

        with torch.no_grad():
            for _ in range(collect_steps):
                norm_s = normalize_state(state).unsqueeze(0)
                action_tensor = actor(norm_s) + torch.randn(1, 3) * 0.1
                action_tensor[:, 0] = torch.clamp(action_tensor[:, 0], -1, 1)
                action_tensor[:, 1:] = torch.clamp(action_tensor[:, 1:], 0, 1)

                raw_action = action_tensor.numpy()[0]

                action_land = raw_action[0]
                action_feed = raw_action[1] * (state[1] * 20 * 1.5) / max(1.0, state[2])
                action_plant = raw_action[2] * min(state[3], state[1] * 10) / max(1.0, state[2])
                env_action = [action_land, action_feed, action_plant]

                next_state, reward, done, info = env.step(env_action)
                buffer.push(state, raw_action, reward, next_state)
                epoch_reward += reward

                state = next_state
                if done:
                    state = env.reset()

        wm_loss_total = 0
        for _ in range(10):
            s, a, r, ns = buffer.sample(batch_size)
            norm_s = normalize_state(s)
            norm_ns = normalize_state(ns)

            pred_ns, pred_r = world_model(norm_s, a)

            loss_state = nn.MSELoss()(pred_ns, norm_ns)
            loss_reward = nn.MSELoss()(pred_r, r)
            wm_loss = loss_state + loss_reward

            wm_optimizer.zero_grad()
            wm_loss.backward()
            wm_optimizer.step()
            wm_loss_total += wm_loss.item()

        actor_loss_total = 0
        for _ in range(10):
            s, _, _, _ = buffer.sample(batch_size)
            imagined_state = normalize_state(s)

            total_imagined_reward = 0

            for t in range(imagination_horizon):
                imagined_action = actor(imagined_state)
                next_imagined_state, imagined_reward = world_model(imagined_state, imagined_action)

                total_imagined_reward += imagined_reward
                imagined_state = next_imagined_state

            actor_loss = -total_imagined_reward.mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            actor_loss_total += actor_loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | WM Loss: {wm_loss_total/10:.2f} | Actor Loss: {actor_loss_total/10:.2f} | Avg Env Reward: {epoch_reward/collect_steps:.2f}")

    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    os.makedirs('models', exist_ok=True)
    torch.save(actor.state_dict(), 'models/dreamer_actor.pth')
    torch.save(world_model.state_dict(), 'models/dreamer_wm.pth')
    print("Saved Micro-Dreamer models to models/")

if __name__ == "__main__":
    train_micro_dreamer()
