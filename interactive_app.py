import streamlit as st
import time
import gymnasium as gym
from stable_baselines3 import PPO
from custom_env import CustomLunarLander

# --- 1. Page Setup & Styling ---
st.set_page_config(page_title="Rocket AI Lab", page_icon="🚀", layout="wide")

# --- 2. Sidebar Controls (Mission Control) ---
st.sidebar.title("🎮 Mission Control")
st.sidebar.markdown("---")

# NEW NAMES HERE:
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

# --- 3. Main UI Header ---
st.title("🚀 Interactive Rocket AI Lab")
st.markdown(
    "Train, tweak, and watch Reinforcement Learning in real-time. Adjust the controls on the left to change the environment, then hit **Launch**.")


# --- 4. Load the AI Model ---
@st.cache_resource
def load_model():
    MODEL_PATH = r"C:\Users\Admin\PycharmProjects\PythonProject2\.venv\ppo_lander_corrected"
    return PPO.load(MODEL_PATH)


model = load_model()

# --- 5. Session State & Buttons ---
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

if stop_button:
    st.session_state.is_running = False
    st.rerun()

if run_button:
    st.session_state.is_running = True

# --- 6. The Simulation Dashboard ---
if st.session_state.is_running:
    env = CustomLunarLander(target_x=target_x, terrain_seed=terrain_seed, render_mode="rgb_array")
    obs, info = env.reset()

    total_reward = 0
    steps = 0
    terminated = False
    truncated = False

    # Create a 2-column dashboard layout
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

    # Real-time loop
    while not (terminated or truncated) and st.session_state.is_running:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        frame = env.render()

        # Update the dashboard UI live
        image_placeholder.image(frame, caption=f"Step: {steps} | Action: {action}", use_container_width=True)
        reward_metric.metric("Total Reward", f"{total_reward:.2f}")
        steps_metric.metric("Steps Survived", steps)

        time.sleep(speed_delay / 1000.0)

    # Episode ended
    env.close()
    st.session_state.is_running = False

    if total_reward >= 200:
        status_box.success(f"🎉 SUCCESS! Landed safely! Final Reward: {total_reward:.2f}")
    else:
        status_box.error(f"💥 CRASHED or failed. Final Reward: {total_reward:.2f}")

else:
    # Default view when not running
    st.subheader("📡 Live Telemetry Feed")
    st.info("Adjust the controls on the left and click **🚀 Launch** to start the simulation.")

    # Show a static preview of the current landscape
    env = CustomLunarLander(target_x=target_x, terrain_seed=terrain_seed, render_mode="rgb_array")
    env.reset()
    frame = env.render()
    st.image(frame, caption="Environment Preview", use_container_width=True)
    env.close()