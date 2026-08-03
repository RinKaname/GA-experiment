import numpy as np
from hammurabi_env import DemocraticHammurabi
from ga_agent import GAHebbianRNNPolicy

def load_policy(model_dir='models'):
    policy = GAHebbianRNNPolicy()
    policy.W_ih_init = np.load(f"{model_dir}/W_ih_init.npy")
    policy.W_hh_init = np.load(f"{model_dir}/W_hh_init.npy")
    policy.b_h_init = np.load(f"{model_dir}/b_h_init.npy")
    policy.W_ho_init = np.load(f"{model_dir}/W_ho_init.npy")
    policy.b_o_init = np.load(f"{model_dir}/b_o_init.npy")

    policy.alpha_ih = np.load(f"{model_dir}/alpha_ih.npy")
    policy.alpha_hh = np.load(f"{model_dir}/alpha_hh.npy")
    policy.alpha_ho = np.load(f"{model_dir}/alpha_ho.npy")

    # Must call reset to load the init weights into active weights
    policy.reset()
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
