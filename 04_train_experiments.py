# Experiment: add a hover bonus to the reward and see what the agent does with it
import os
HERE = os.path.dirname(os.path.abspath(__file__))  # models are saved next to this script, so git can see them

import gymnasium as gym
from stable_baselines3 import PPO
from reward_experiments import RewardExperiment

MODEL_PATH = os.path.join(HERE, "ppo_lander_hover")

env = RewardExperiment(gym.make("LunarLander-v3"), mode="hover_bonus")
model = PPO("MlpPolicy", env, verbose=1, seed=42)

print("Training with the hover bonus reward...")
model.learn(total_timesteps=300_000)

model.save(MODEL_PATH)
env.close()
print("Saved to", MODEL_PATH + ".zip")