import gymnasium as gym
from stable_baselines3 import PPO
import time

MODEL_PATH = r"C:\Users\Admin\PycharmProjects\PythonProject2\.venv\ppo_lander_baseline"

print("Loading the simulation...")
env = gym.make("LunarLander-v3", render_mode="human")
print("Loading the trained AI brain...")
model = PPO.load(MODEL_PATH)
print("Watching the trained AI fly!")

for episode in range(3):
    observation, info = env.reset()
    total_reward = 0
    steps = 0
    print(f"Episode {episode + 1} started")

    while True:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.02)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            print("-" * 30)
            print(f"Episode {episode + 1} finished!")
            print(f"Total steps: {steps}")
            print(f"Total reward: {total_reward:.2f}")
            print("-" * 30)
            time.sleep(1)
            break

env.close()