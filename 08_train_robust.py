# Retrain the corrected agent on random gravity so it also works when gravity is weak.
# Starts from my corrected model instead of from zero, so it's faster than 3M from scratch.
# (If the results are bad, train from scratch: replace PPO.load(...) with PPO("MlpPolicy", env, ...))
import os
HERE = os.path.dirname(os.path.abspath(__file__))

import gymnasium as gym
from stable_baselines3 import PPO
from reward_experiments import RewardExperiment
from gravity_wrapper import RandomGravity

START_FROM = os.path.join(HERE, "ppo_lander_corrected")
MODEL_PATH = os.path.join(HERE, "ppo_lander_robust")

env = RandomGravity(RewardExperiment(gym.make("LunarLander-v3"), mode="corrected"))
model = PPO.load(START_FROM, env=env)

print("Training with random gravity between -11.5 and -2...")
model.learn(total_timesteps=2_000_000)

model.save(MODEL_PATH)
env.close()
print("Saved to", MODEL_PATH + ".zip")