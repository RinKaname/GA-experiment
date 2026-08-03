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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space: 9 features
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(9,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        action_land = action[0]
        action_feed_frac = (action[1] + 1.0) / 2.0
        action_plant_frac = (action[2] + 1.0) / 2.0

        state_np = self.env._get_state()

        optimal_feed_grain = state_np[1] * 20
        actual_feed_grain = action_feed_frac * optimal_feed_grain * 1.5
        env_action_feed = actual_feed_grain / max(1.0, state_np[2])

        max_plantable_grain = min(state_np[3], state_np[1] * 10)
        actual_plant_grain = action_plant_frac * max_plantable_grain
        env_action_plant = actual_plant_grain / max(1.0, state_np[2])

        env_actions = [action_land, env_action_feed, env_action_plant]

        obs, reward, done, info = self.env.step(env_actions)
        terminated = done
        truncated = False

        return np.array(obs, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        print(self.env._get_state())
