import numpy as np
import random
from hammurabi_env import DemocraticHammurabi

class GAPolicy:
    def __init__(self, state_size, action_size):
        # We will use a simple linear policy: Action = State * Weights
        # State: 8 features. Action: 3 continuous values
        self.weights = np.random.randn(action_size, state_size) * 0.1
        self.bias = np.random.randn(action_size) * 0.1

    def act(self, state):
        # state is a list/array of 8 features
        state_np = np.array(state)
        # Normalize state features to prevent exploding values in dot product
        # Rough normalization based on expected ranges:
        # year (1-12), pop (~100), grain (~2800), land (~1000), price (~20), approvals (0-100)
        norm_state = np.array([
            state_np[0] / 12.0,
            state_np[1] / 1000.0,
            state_np[2] / 10000.0,
            state_np[3] / 5000.0,
            state_np[4] / 30.0,
            state_np[5] / 100.0,
            state_np[6] / 100.0,
            state_np[7] / 100.0
        ])

        raw_action = np.dot(self.weights, norm_state) + self.bias

        # Actions are:
        # 0: land (-1 to 1) -> use tanh
        # 1: feed (0 to 1) -> use sigmoid
        # 2: plant (0 to 1) -> use sigmoid

        action_land = np.tanh(raw_action[0])
        action_feed = 1.0 / (1.0 + np.exp(-raw_action[1]))
        action_plant = 1.0 / (1.0 + np.exp(-raw_action[2]))

        return [action_land, action_feed, action_plant]

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        # Mutate weights
        weight_mutation_mask = np.random.rand(*self.weights.shape) < mutation_rate
        self.weights += weight_mutation_mask * np.random.randn(*self.weights.shape) * mutation_scale

        # Mutate bias
        bias_mutation_mask = np.random.rand(*self.bias.shape) < mutation_rate
        self.bias += bias_mutation_mask * np.random.randn(*self.bias.shape) * mutation_scale

def crossover(parent1, parent2):
    child = GAPolicy(8, 3)
    # Uniform crossover
    weight_mask = np.random.rand(*child.weights.shape) < 0.5
    child.weights = np.where(weight_mask, parent1.weights, parent2.weights)

    bias_mask = np.random.rand(*child.bias.shape) < 0.5
    child.bias = np.where(bias_mask, parent1.bias, parent2.bias)
    return child

def evaluate_policy(policy, num_episodes=3):
    env = DemocraticHammurabi()
    total_reward = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = policy.act(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        # We care about the final state/reward heavily.
        # The env calculates reward progressively, so we can just use the final reward call as the fitness
        total_reward += reward

    return total_reward / num_episodes

if __name__ == "__main__":
    # Test random policy
    policy = GAPolicy(8, 3)
    fitness = evaluate_policy(policy, 1)
    print(f"Random policy fitness: {fitness}")
