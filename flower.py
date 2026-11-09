import gymnasium as gym
import time

print("Loading the physics engine...")
env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset(seed=42)
print("Watch the rocket! It is choosing actions completely randomly.")

for step in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Crash or landing at step {step}! Restarting...")
        observation, info = env.reset()
        time.sleep(1)

env.close()
print("Simulation finished.")