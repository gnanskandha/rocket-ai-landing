# Fixed version: no hover bonus, small time penalty, much longer training
import os
HERE = os.path.dirname(os.path.abspath(__file__))  # models are saved next to this script, so git can see them

import gymnasium as gym
from stable_baselines3 import PPO
from reward_experiments import RewardExperiment

MODEL_PATH = os.path.join(HERE, "ppo_lander_corrected")

env = RewardExperiment(gym.make("LunarLander-v3"), mode="corrected")
model = PPO("MlpPolicy", env, verbose=1, seed=42)

print("Training with the corrected reward...")
model.learn(total_timesteps=3_000_000)

model.save(MODEL_PATH)
env.close()
print("Saved to", MODEL_PATH + ".zip")