import gymnasium as gym
from stable_baselines3 import PPO
import time

MODEL_PATH = r"C:\Users\Admin\PycharmProjects\PythonProject2\.venv\ppo_lander_hover"

print("Loading the simulation...")
env = gym.make("LunarLander-v3", render_mode="human")
print("Loading the CHEATING AI brain...")
model = PPO.load(MODEL_PATH)
print("Watch closely! The AI will try to hover instead of landing.")

observation, info = env.reset()
total_reward = 0
steps = 0

while True:
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.02)
    total_reward += reward
    steps += 1
    if terminated or truncated:
        print("-" * 50)
        print(f"Total steps survived: {steps}")
        print(f"Total reward: {total_reward:.2f}")
        print("-" * 50)
        break

env.close()