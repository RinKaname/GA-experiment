import numpy as np
from stable_baselines3 import PPO
from gym_env import GymDemocraticHammurabi

def benchmark_ppo(model_path="models/ppo_hammurabi", num_episodes=100):
    env = GymDemocraticHammurabi()

    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Could not load PPO model: {e}. Please run train_ppo.py first.")
        return

    wins = 0
    total_starved = 0
    total_population = 0
    final_years = []

    print(f"Benchmarking PPO policy over {num_episodes} episodes...")

    for i in range(num_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        final_years.append(env.env.year)
        total_starved += env.env.starved_total
        total_population += env.env.population

        if env.env.year > env.env.max_years:
            wins += 1

    win_rate = (wins / num_episodes) * 100
    avg_starved = total_starved / num_episodes
    avg_pop = total_population / num_episodes
    avg_year = sum(final_years) / num_episodes

    print("-" * 30)
    print("PPO BENCHMARK RESULTS")
    print("-" * 30)
    print(f"Win Rate (Survive 12 yrs): {win_rate:.1f}%")
    print(f"Average Final Year: {avg_year:.1f}")
    print(f"Average Total Starved: {avg_starved:.1f} people per game")
    print(f"Average Final Population: {avg_pop:.1f} people")
    print("-" * 30)

if __name__ == "__main__":
    benchmark_ppo(num_episodes=100)
