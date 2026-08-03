import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym_env import GymDemocraticHammurabi
import time

def train_ppo(timesteps=200000):
    print("Initializing PPO Environment...")
    env = GymDemocraticHammurabi()

    # Check that the environment follows gymnasium API perfectly
    check_env(env, warn=True)

    # Initialize PPO model
    # We use a small MLP policy similar to our GA agent for a fair comparison:
    # pi (actor network) and vf (critic network) with 2 hidden layers of 32 nodes
    policy_kwargs = dict(net_arch=[32, 32])

    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./ppo_tensorboard/")

    print(f"Starting PPO Training for {timesteps} timesteps...")
    start_time = time.time()
    model.learn(total_timesteps=timesteps)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    os.makedirs('models', exist_ok=True)
    model.save("models/ppo_hammurabi")
    print("Saved PPO model to models/ppo_hammurabi.zip")

if __name__ == "__main__":
    # Train for 1,000,000 timesteps to allow PPO enough exploration
    # to escape the early starvation local minimums.
    train_ppo(timesteps=1000000)
