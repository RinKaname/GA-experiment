import gymnasium as gym
from gymnasium import spaces
import numpy as np
from hammurabi_env import DemocraticHammurabi

class GymDemocraticHammurabi(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    """
    metadata = {'render_modes': ['console']}

    def __init__(self, max_years=12):
        super(GymDemocraticHammurabi, self).__init__()
        self.env = DemocraticHammurabi(max_years=max_years)

        # Action space: 3 continuous values between -1 and 1
        # [land_action, feed_action, plant_action]
        # We will scale these inside the step function to what the game expects
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space: 9 features
        # year, pop, grain, land, price, f_app, w_app, e_app, yrs_to_elec
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(9,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # We need to manually seed our underlying environment's random calls if needed,
        # but for simplicity we rely on the global random state as it's already set up.
        obs = self.env.reset()

        # Gymnasium expects (obs, info)
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        # Action from PPO comes in as [-1, 1]
        action_land = action[0]

        # feed and plant from PPO are [-1, 1], convert to [0, 1] for easy math
        action_feed_frac = (action[1] + 1.0) / 2.0
        action_plant_frac = (action[2] + 1.0) / 2.0

        state_np = self.env._get_state()

        # Same translation logic as our GA agent so it's a fair comparison
        optimal_feed_grain = state_np[1] * 20
        actual_feed_grain = action_feed_frac * optimal_feed_grain * 1.5
        env_action_feed = actual_feed_grain / max(1.0, state_np[2])

        max_plantable_grain = min(state_np[3], state_np[1] * 10)
        actual_plant_grain = action_plant_frac * max_plantable_grain
        env_action_plant = actual_plant_grain / max(1.0, state_np[2])

        env_actions = [action_land, env_action_feed, env_action_plant]

        # Step the actual environment
        obs, reward, done, info = self.env.step(env_actions)

        # In modern gymnasium, done is split into terminated and truncated
        terminated = done
        truncated = False # We don't artificially truncate unless max_years is hit via terminated

        return np.array(obs, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        print(self.env._get_state())
