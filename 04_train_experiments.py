import gymnasium as gym
from stable_baselines3 import PPO
from reward_experiments import RewardExperiment

MODEL_SAVE_PATH = r"C:\Users\Admin\PycharmProjects\PythonProject2\.venv\ppo_lander_hover"

print("Creating environment with HOVER BONUS reward...")
base_env = gym.make("LunarLander-v3")
env = RewardExperiment(base_env, mode="hover_bonus")
print("Initializing the AI brain (PPO)...")
model = PPO("MlpPolicy", env, verbose=1, seed=42)

print("Starting training with the flawed reward!")
model.learn(total_timesteps=300_000)

print("-" * 50)
print("Saving model to:")
print(MODEL_SAVE_PATH + ".zip")
model.save(MODEL_SAVE_PATH)
env.close()
print("Training complete!")