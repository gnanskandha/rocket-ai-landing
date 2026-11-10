# Record the corrected agent landing to mp4
import os
HERE = os.path.dirname(os.path.abspath(__file__))  # models are saved next to this script, so git can see them

import gymnasium as gym
from stable_baselines3 import PPO

VIDEO_FOLDER = os.path.join(HERE, "videos")
os.makedirs(VIDEO_FOLDER, exist_ok=True)

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, video_folder=VIDEO_FOLDER, name_prefix="landing",
                               episode_trigger=lambda x: True)
model = PPO.load(os.path.join(HERE, "ppo_lander_corrected"))

for episode in range(3):
    obs, info = env.reset()
    total, steps = 0, 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if terminated or truncated:
            print(f"episode {episode + 1}: {steps} steps, reward {total:.1f}")
            break

env.close()
print("videos saved in", VIDEO_FOLDER)