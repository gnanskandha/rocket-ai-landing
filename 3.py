# Watch the baseline agent fly 3 episodes
import os
HERE = os.path.dirname(os.path.abspath(__file__))  # models are saved next to this script, so git can see them

import time
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="human")
model = PPO.load(os.path.join(HERE, "ppo_lander_baseline"))

for episode in range(3):
    obs, info = env.reset()
    total, steps = 0, 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.02)
        total += reward
        steps += 1
        if terminated or truncated:
            print(f"episode {episode + 1}: {steps} steps, reward {total:.1f}")
            time.sleep(1)
            break

env.close()