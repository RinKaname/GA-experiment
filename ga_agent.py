import numpy as np
import random
from hammurabi_env import DemocraticHammurabi

class GAPolicy:
    def __init__(self, state_size=9, hidden_size=16, action_size=3):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        # MLP: 9 -> 16 -> 3
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
            state_np[7] / 100.0,
            state_np[8] / 4.0
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

class GARNNPolicy:
    def __init__(self, state_size=9, hidden_size=16, action_size=3):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        # RNN parameters
        self.W_ih = np.random.randn(hidden_size, state_size) * np.sqrt(2.0 / state_size) # Input to Hidden
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size) # Hidden to Hidden
        self.b_h = np.zeros(hidden_size) # Hidden bias

        self.W_ho = np.random.randn(action_size, hidden_size) * np.sqrt(2.0 / hidden_size) # Hidden to Output
        self.b_o = np.zeros(action_size) # Output bias

        self.hidden_state = np.zeros(hidden_size)

    def reset(self):
        self.hidden_state = np.zeros(self.hidden_size)

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
            state_np[7] / 100.0,
            state_np[8] / 4.0
        ])

        # RNN Step: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b_h)
        self.hidden_state = np.tanh(np.dot(self.W_ih, norm_state) + np.dot(self.W_hh, self.hidden_state) + self.b_h)

        # Output Step: y_t = W_ho * h_t + b_o
        raw_action = np.dot(self.W_ho, self.hidden_state) + self.b_o

        # Actions
        action_land = np.tanh(raw_action[0])
        action_feed = 1.0 / (1.0 + np.exp(-raw_action[1]))
        action_plant = 1.0 / (1.0 + np.exp(-raw_action[2]))

        optimal_feed_grain = state_np[1] * 20
        actual_feed_grain = action_feed * optimal_feed_grain * 1.5
        env_action_feed = actual_feed_grain / max(1.0, state_np[2])

        max_plantable_grain = min(state_np[3], state_np[1] * 10)
        actual_plant_grain = action_plant * max_plantable_grain
        env_action_plant = actual_plant_grain / max(1.0, state_np[2])

        return [action_land, env_action_feed, env_action_plant]

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        for param in [self.W_ih, self.W_hh, self.b_h, self.W_ho, self.b_o]:
            mask = np.random.rand(*param.shape) < mutation_rate
            param += mask * np.random.randn(*param.shape) * mutation_scale


class GAHebbianRNNPolicy:
    def __init__(self, state_size=9, hidden_size=16, action_size=3):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        # Meta-parameters (Evolved by GA)
        self.W_ih_init = np.random.randn(hidden_size, state_size) * np.sqrt(2.0 / state_size)
        self.W_hh_init = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_h_init = np.zeros(hidden_size)
        self.W_ho_init = np.random.randn(action_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_o_init = np.zeros(action_size)

        # Plasticity rates (Evolved by GA)
        self.alpha_ih = np.random.randn(hidden_size, state_size) * 0.01
        self.alpha_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.alpha_ho = np.random.randn(action_size, hidden_size) * 0.01

        # Runtime variables (Reset every episode)
        self.hidden_state = np.zeros(hidden_size)
        self.W_ih = None
        self.W_hh = None
        self.b_h = None
        self.W_ho = None
        self.b_o = None
        self.reset()

    def reset(self):
        self.hidden_state = np.zeros(self.hidden_size)
        # Copy initial evolved weights to active runtime weights
        self.W_ih = np.copy(self.W_ih_init)
        self.W_hh = np.copy(self.W_hh_init)
        self.b_h = np.copy(self.b_h_init)
        self.W_ho = np.copy(self.W_ho_init)
        self.b_o = np.copy(self.b_o_init)

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
            state_np[7] / 100.0,
            state_np[8] / 4.0
        ])

        prev_hidden_state = np.copy(self.hidden_state)

        # Forward Pass
        # RNN Step: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b_h)
        self.hidden_state = np.tanh(np.dot(self.W_ih, norm_state) + np.dot(self.W_hh, prev_hidden_state) + self.b_h)

        # Output Step: y_t = W_ho * h_t + b_o
        raw_action = np.dot(self.W_ho, self.hidden_state) + self.b_o

        # Hebbian Learning Updates (Oja's Rule to prevent explosion: dW = alpha * (y*x - y^2 * W))
        # Update Input->Hidden
        dW_ih = self.alpha_ih * (self.hidden_state[:, None] * norm_state[None, :] - (self.hidden_state[:, None]**2) * self.W_ih)
        self.W_ih = np.clip(self.W_ih + dW_ih, -10.0, 10.0)

        # Update Hidden->Hidden
        dW_hh = self.alpha_hh * (self.hidden_state[:, None] * prev_hidden_state[None, :] - (self.hidden_state[:, None]**2) * self.W_hh)
        self.W_hh = np.clip(self.W_hh + dW_hh, -10.0, 10.0)

        # Update Hidden->Output
        # Since raw_action is pre-activation, we use it directly as the "output" for Hebbian learning
        dW_ho = self.alpha_ho * (raw_action[:, None] * self.hidden_state[None, :] - (raw_action[:, None]**2) * self.W_ho)
        self.W_ho = np.clip(self.W_ho + dW_ho, -10.0, 10.0)

        # Actions translation
        action_land = np.tanh(raw_action[0])
        action_feed = 1.0 / (1.0 + np.exp(-raw_action[1]))
        action_plant = 1.0 / (1.0 + np.exp(-raw_action[2]))

        optimal_feed_grain = state_np[1] * 20
        actual_feed_grain = action_feed * optimal_feed_grain * 1.5
        env_action_feed = actual_feed_grain / max(1.0, state_np[2])

        max_plantable_grain = min(state_np[3], state_np[1] * 10)
        actual_plant_grain = action_plant * max_plantable_grain
        env_action_plant = actual_plant_grain / max(1.0, state_np[2])

        return [action_land, env_action_feed, env_action_plant]

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        params = [
            self.W_ih_init, self.W_hh_init, self.b_h_init, self.W_ho_init, self.b_o_init,
            self.alpha_ih, self.alpha_hh, self.alpha_ho
        ]
        for param in params:
            mask = np.random.rand(*param.shape) < mutation_rate
            param += mask * np.random.randn(*param.shape) * mutation_scale


def crossover(parent1, parent2):
    if isinstance(parent1, GAPolicy):
        child = GAPolicy(parent1.state_size, parent1.hidden_size, parent1.action_size)
        child.W1 = np.where(np.random.rand(*child.W1.shape) < 0.5, parent1.W1, parent2.W1)
        child.b1 = np.where(np.random.rand(*child.b1.shape) < 0.5, parent1.b1, parent2.b1)
        child.W2 = np.where(np.random.rand(*child.W2.shape) < 0.5, parent1.W2, parent2.W2)
        child.b2 = np.where(np.random.rand(*child.b2.shape) < 0.5, parent1.b2, parent2.b2)
        return child

    elif isinstance(parent1, GARNNPolicy):
        child = GARNNPolicy(parent1.state_size, parent1.hidden_size, parent1.action_size)
        child.W_ih = np.where(np.random.rand(*child.W_ih.shape) < 0.5, parent1.W_ih, parent2.W_ih)
        child.W_hh = np.where(np.random.rand(*child.W_hh.shape) < 0.5, parent1.W_hh, parent2.W_hh)
        child.b_h = np.where(np.random.rand(*child.b_h.shape) < 0.5, parent1.b_h, parent2.b_h)
        child.W_ho = np.where(np.random.rand(*child.W_ho.shape) < 0.5, parent1.W_ho, parent2.W_ho)
        child.b_o = np.where(np.random.rand(*child.b_o.shape) < 0.5, parent1.b_o, parent2.b_o)
        return child

    elif isinstance(parent1, GAHebbianRNNPolicy):
        child = GAHebbianRNNPolicy(parent1.state_size, parent1.hidden_size, parent1.action_size)
        child.W_ih_init = np.where(np.random.rand(*child.W_ih_init.shape) < 0.5, parent1.W_ih_init, parent2.W_ih_init)
        child.W_hh_init = np.where(np.random.rand(*child.W_hh_init.shape) < 0.5, parent1.W_hh_init, parent2.W_hh_init)
        child.b_h_init = np.where(np.random.rand(*child.b_h_init.shape) < 0.5, parent1.b_h_init, parent2.b_h_init)
        child.W_ho_init = np.where(np.random.rand(*child.W_ho_init.shape) < 0.5, parent1.W_ho_init, parent2.W_ho_init)
        child.b_o_init = np.where(np.random.rand(*child.b_o_init.shape) < 0.5, parent1.b_o_init, parent2.b_o_init)

        child.alpha_ih = np.where(np.random.rand(*child.alpha_ih.shape) < 0.5, parent1.alpha_ih, parent2.alpha_ih)
        child.alpha_hh = np.where(np.random.rand(*child.alpha_hh.shape) < 0.5, parent1.alpha_hh, parent2.alpha_hh)
        child.alpha_ho = np.where(np.random.rand(*child.alpha_ho.shape) < 0.5, parent1.alpha_ho, parent2.alpha_ho)
        return child

def evaluate_policy(policy, num_episodes=3):
    env = DemocraticHammurabi()
    total_reward = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        if hasattr(policy, 'reset'):
            policy.reset()

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
