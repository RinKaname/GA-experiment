import numpy as np
from hammurabi_env import DemocraticHammurabi
from ga_agent import GAPolicy

def load_policy(model_dir='models'):
    policy = GAPolicy()
    policy.W1 = np.load(f"{model_dir}/W1.npy")
    policy.b1 = np.load(f"{model_dir}/b1.npy")
    policy.W2 = np.load(f"{model_dir}/W2.npy")
    policy.b2 = np.load(f"{model_dir}/b2.npy")
    return policy

def benchmark(policy, num_episodes=100):
    env = DemocraticHammurabi()

    wins = 0
    total_starved = 0
    total_population = 0
    final_years = []

    print(f"Benchmarking best policy over {num_episodes} episodes...")

    for i in range(num_episodes):
        state = env.reset()
        done = False

        if hasattr(policy, 'reset'):
            policy.reset()

        while not done:
            action = policy.act(state)
            state, reward, done, info = env.step(action)

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
    print("BENCHMARK RESULTS")
    print("-" * 30)
    print(f"Win Rate (Survive 12 yrs): {win_rate:.1f}%")
    print(f"Average Final Year: {avg_year:.1f}")
    print(f"Average Total Starved: {avg_starved:.1f} people per game")
    print(f"Average Final Population: {avg_pop:.1f} people")
    print("-" * 30)

if __name__ == "__main__":
    best_policy = load_policy()
    benchmark(best_policy, num_episodes=100)
