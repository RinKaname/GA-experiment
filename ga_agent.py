import numpy as np
import random
from hammurabi_env import DemocraticHammurabi

class GAPolicy:
    def __init__(self, state_size=8, hidden_size=16, action_size=3):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        # MLP: 8 -> 16 -> 3
        self.W1 = np.random.randn(hidden_size, state_size) * np.sqrt(2.0 / state_size) # He initialization
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.randn(action_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(action_size)

    def _relu(self, x):
        return np.maximum(0, x)

    def act(self, state):
        state_np = np.array(state)
        # Normalize state features
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

        # Layer 1
        z1 = np.dot(self.W1, norm_state) + self.b1
        a1 = self._relu(z1)

        # Layer 2
        raw_action = np.dot(self.W2, a1) + self.b2

        # Actions
        # Make the output easier to map to the environment
        # land: [-1, 1]
        action_land = np.tanh(raw_action[0])
        # feed: [0, 1] - representing fraction of optimal food (20 * pop)
        action_feed = 1.0 / (1.0 + np.exp(-raw_action[1]))
        # plant: [0, 1] - representing fraction of max plantable (land or pop*10)
        action_plant = 1.0 / (1.0 + np.exp(-raw_action[2]))

        # Translate the "easier" actions into what the env expects (fractions of grain)
        optimal_feed_grain = state_np[1] * 20
        actual_feed_grain = action_feed * optimal_feed_grain * 1.5 # Allow overfeeding up to 1.5x
        env_action_feed = actual_feed_grain / max(1.0, state_np[2])

        max_plantable_grain = min(state_np[3], state_np[1] * 10)
        actual_plant_grain = action_plant * max_plantable_grain
        env_action_plant = actual_plant_grain / max(1.0, state_np[2])

        return [action_land, env_action_feed, env_action_plant]

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        for param in [self.W1, self.b1, self.W2, self.b2]:
            mask = np.random.rand(*param.shape) < mutation_rate
            param += mask * np.random.randn(*param.shape) * mutation_scale

def crossover(parent1, parent2):
    child = GAPolicy(parent1.state_size, parent1.hidden_size, parent1.action_size)

    # Uniform crossover for all parameters
    child.W1 = np.where(np.random.rand(*child.W1.shape) < 0.5, parent1.W1, parent2.W1)
    child.b1 = np.where(np.random.rand(*child.b1.shape) < 0.5, parent1.b1, parent2.b1)
    child.W2 = np.where(np.random.rand(*child.W2.shape) < 0.5, parent1.W2, parent2.W2)
    child.b2 = np.where(np.random.rand(*child.b2.shape) < 0.5, parent1.b2, parent2.b2)

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

        total_reward += reward

    return total_reward / num_episodes

if __name__ == "__main__":
    policy = GAPolicy()
    fitness = evaluate_policy(policy, 1)
    print(f"Random MLP policy fitness: {fitness}")
