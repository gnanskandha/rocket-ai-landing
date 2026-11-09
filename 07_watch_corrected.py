import gymnasium as gym
from stable_baselines3 import PPO
import time
import os

MODEL_PATH = r"C:\Users\Admin\PycharmProjects\PythonProject2\.venv\ppo_lander_corrected"
VIDEO_FOLDER = r"C:\Users\Admin\PycharmProjects\PythonProject2\videos"

# Create the videos folder if it doesn't exist
os.makedirs(VIDEO_FOLDER, exist_ok=True)

print("Loading the simulation WITH VIDEO RECORDING...")
# rgb_array mode is REQUIRED for video recording
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Wrap the environment to automatically record videos
env = gym.wrappers.RecordVideo(
    env,
    video_folder=VIDEO_FOLDER,
    name_prefix="landing",
    episode_trigger=lambda x: True  # Record every episode
)

print("Loading the CORRECTED AI brain...")
model = PPO.load(MODEL_PATH)

print("🎥 Recording video of the AI landing...")
print(f"Video will be saved in: {VIDEO_FOLDER}")
print("-" * 50)

for episode in range(3):
    observation, info = env.reset()
    total_reward = 0
    steps = 0
    print(f"Episode {episode + 1} started")

    while True:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        if terminated or truncated:
            print("-" * 50)
            print(f"Episode {episode + 1} finished!")
            print(f"Total steps: {steps}")
            print(f"Total reward: {total_reward:.2f}")
            if total_reward >= 200:
                print("🎉 SUCCESS: The AI landed safely!")
            print("-" * 50)
            break

# Close the environment to finalize the video files
env.close()
print("-" * 50)
print("🎥 VIDEO SAVED!")
print(f"Check your 'videos' folder for the .mp4 files!")
print(f"Location: {VIDEO_FOLDER}")