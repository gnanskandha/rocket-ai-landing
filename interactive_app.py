# Interactive Rocket AI Landing App
# This script runs a Streamlit web app to test my trained PPO model in real-time.
# I can change the terrain, move the landing pad, and watch the AI adapt.

import streamlit as st
import time
import os
from stable_baselines3 import PPO
from custom_env import CustomLunarLander

# 1. Page setup
st.set_page_config(page_title="Rocket AI Lab", page_icon="🚀", layout="wide")

# 2. Sidebar controls
st.sidebar.title("🎮 Mission Control")
st.sidebar.markdown("---")

# Sliders to change the environment
terrain_seed = st.sidebar.slider("🌍 Planet Surface (Seed)", 0, 100, 0, 1)
target_x = st.sidebar.slider("🎯 Pad Position (X)", -1.5, 1.5, 0.0, 0.1)
speed_delay = st.sidebar.slider("⏳ Time Dilation (Slow Motion)", 0, 100, 20, 5)

st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
with col1:
    run_button = st.button("🚀 Launch", use_container_width=True, type="primary")
with col2:
    stop_button = st.button("🛑 Stop", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Built with PPO, Gymnasium & Streamlit")

# 3. Main title
st.title("🚀 Interactive Rocket AI Lab")
st.markdown(
    "Adjust the controls on the left to change the environment, then hit **Launch** to watch the AI fly in real-time.")


# 4. Load the trained AI model
# Using cache so it doesn't reload the heavy model every time I change a slider
@st.cache_resource
def load_model():
    # This finds the model in the same folder as the script (works on both Windows and Streamlit Cloud)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "ppo_lander_corrected")
    return PPO.load(model_path)


model = load_model()

# 5. Handle button clicks and session state
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

if stop_button:
    st.session_state.is_running = False
    st.rerun()

if run_button:
    st.session_state.is_running = True

# 6. Run the simulation
if st.session_state.is_running:
    # Create the custom environment with my chosen settings
    env = CustomLunarLander(target_x=target_x, terrain_seed=terrain_seed, render_mode="rgb_array")
    obs, info = env.reset()

    total_reward = 0
    steps = 0
    terminated = False
    truncated = False

    # Split the main screen: 2/3 for video, 1/3 for stats
    feed_col, stats_col = st.columns([2, 1])

    with feed_col:
        st.subheader("📡 Live Telemetry Feed")
        image_placeholder = st.empty()

    with stats_col:
        st.subheader("📊 Flight Metrics")
        reward_metric = st.empty()
        steps_metric = st.empty()
        status_box = st.empty()

    status_box.info("🚀 Rocket is in flight...")

    # Step through the environment until it lands or crashes
    while not (terminated or truncated) and st.session_state.is_running:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        # Get the current frame and update the UI
        frame = env.render()
        image_placeholder.image(frame, caption=f"Step: {steps} | Action: {action}", use_container_width=True)

        # Update the metrics on the side
        reward_metric.metric("Total Reward", f"{total_reward:.2f}")
        steps_metric.metric("Steps Survived", steps)

        # Slow down the loop so I can actually see what's happening
        time.sleep(speed_delay / 1000.0)

    # Clean up when the episode ends
    env.close()
    st.session_state.is_running = False

    if total_reward >= 200:
        status_box.success(f"🎉 SUCCESS! Landed safely! Final Reward: {total_reward:.2f}")
    else:
        status_box.error(f"💥 CRASHED or failed. Final Reward: {total_reward:.2f}")

else:
    # Show a preview when the app is idle
    st.subheader("📡 Live Telemetry Feed")
    st.info("Adjust the controls on the left and click **🚀 Launch** to start the simulation.")

    env = CustomLunarLander(target_x=target_x, terrain_seed=terrain_seed, render_mode="rgb_array")
    env.reset()
    frame = env.render()
    st.image(frame, caption="Environment Preview", use_container_width=True)
    env.close()