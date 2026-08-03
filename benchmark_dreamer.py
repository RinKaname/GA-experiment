import torch
import numpy as np
from hammurabi_env import DemocraticHammurabi
from micro_dreamer import Actor, normalize_state

def benchmark_dreamer(model_path='models/dreamer_actor.pth', num_episodes=100):
    env = DemocraticHammurabi()

    actor = Actor()
    actor.load_state_dict(torch.load(model_path))
    actor.eval() # Set to evaluation mode

    wins = 0
    total_starved = 0
    total_population = 0
    final_years = []

    print(f"Benchmarking Micro-Dreamer policy over {num_episodes} episodes...")

    for i in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            norm_s = normalize_state(state).unsqueeze(0)
            with torch.no_grad():
                action_tensor = actor(norm_s)
                # Ensure actions are bounded
                action_tensor[:, 0] = torch.clamp(action_tensor[:, 0], -1, 1)
                action_tensor[:, 1:] = torch.clamp(action_tensor[:, 1:], 0, 1)
                raw_action = action_tensor.numpy()[0]

            # Translate to env
            action_land = raw_action[0]
            action_feed = raw_action[1] * (state[1] * 20 * 1.5) / max(1.0, state[2])
            action_plant = raw_action[2] * min(state[3], state[1] * 10) / max(1.0, state[2])
            env_action = [action_land, action_feed, action_plant]

            state, reward, done, info = env.step(env_action)

        final_years.append(env.year)
        total_starved += env.starved_total
        total_population += env.population

        if env.year > env.max_years:
            wins += 1

    win_rate = (wins / num_episodes) * 100
    avg_starved = total_starved / num_episodes
    avg_pop = total_population / num_episodes
    avg_year = sum(final_years) / num_episodes

    print("-" * 30)
    print("MICRO-DREAMER BENCHMARK RESULTS")
    print("-" * 30)
    print(f"Win Rate (Survive 12 yrs): {win_rate:.1f}%")
    print(f"Average Final Year: {avg_year:.1f}")
    print(f"Average Total Starved: {avg_starved:.1f} people per game")
    print(f"Average Final Population: {avg_pop:.1f} people")
    print("-" * 30)

if __name__ == "__main__":
    benchmark_dreamer(num_episodes=100)
