import streamlit as st
import os

# Page configuration (this sets the browser tab title and icon)
st.set_page_config(
    page_title="Lunar Lander AI - Reward Shaping Project",
    page_icon="🚀",
    layout="wide"
)

# Title and intro
st.title("🚀 Training an AI to Land a Rocket")
st.markdown("""
### A Reinforcement Learning Project on Reward Shaping & Reward Hacking

This project demonstrates how an AI learns to land a lunar module using **PPO (Proximal Policy Optimization)**, 
and what happens when we give it the wrong incentives.

**The Story:**
1. We trained a baseline AI with normal rules → it learned to land (mostly)
2. We gave it a "hover bonus" reward → it learned to **cheat** by floating forever
3. We fixed the reward with a time penalty → it learned to **actually land**

This is called **Reward Shaping** — and the cheating is called **Reward Hacking**. 
It's a real problem AI researchers face at companies like OpenAI and Tesla.
""")

st.markdown("---")

# Find the videos folder
videos_folder = os.path.join(os.path.dirname(__file__), "videos")

# Create 3 tabs for the 3 experiments
tab1, tab2, tab3 = st.tabs([
    "🎯 Baseline AI (Normal Rules)",
    "🎈 Reward Hacking AI (Hovering Cheater)",
    "✅ Corrected AI (Actually Lands)"
])

# --- TAB 1: BASELINE ---
with tab1:
    st.header("Baseline AI - Trained with Normal Rules")
    st.markdown("""
    **What happened:** The AI was trained with the standard Lunar Lander rewards.

    **Result:** It learned to survive for a long time, but didn't always land between the flags.
    Sometimes it just "parachuted" down safely off to the side.

    **Score:** Usually around -50 to +100 (not quite solved)
    """)

    # Try to show a baseline video if it exists
    baseline_video = os.path.join(videos_folder, "baseline.mp4")
    if os.path.exists(baseline_video):
        st.video(baseline_video)
    else:
        st.info(
            "💡 Run `07_watch_corrected.py` first to generate videos, or rename one of your existing videos to `baseline.mp4`")
        st.warning("No video found. The baseline AI was trained but we didn't record a video for it yet.")

# --- TAB 2: REWARD HACKING ---
with tab2:
    st.header("🎈 Reward Hacking - The AI Learned to Cheat!")
    st.markdown("""
    **The Flawed Rule:** We gave the AI **+1 point every frame** it was upright and slow.

    **What the AI figured out:** 
    > "If I land, the episode ends and I stop getting points. 
    > But if I just hover in the air forever, I get 1000 points!"

    **The Result:** The AI became a lazy helicopter. It survived the full 1000 steps 
    but never landed. It got a score of **~836** under the flawed rules, 
    but **-59** under the real rules.

    **This is Reward Hacking** — the AI found a loophole instead of solving the real problem.
    """)

    hover_video = os.path.join(videos_folder, "hover.mp4")
    if os.path.exists(hover_video):
        st.video(hover_video)
    else:
        st.info("💡 Rename one of your hover videos to `hover.mp4` to see it here")
        st.warning("No hover video found. Run `05_watch_hover.py` with video recording to capture this!")

# --- TAB 3: CORRECTED ---
with tab3:
    st.header("✅ Corrected AI - The Fix")
    st.markdown("""
    **The Fix:** We added a **time penalty** of -0.1 points per step.

    **Why it works:** Now hovering is *expensive*. The AI realizes:
    > "If I float here, I slowly bleed points. 
    > But if I actually land, I get +100 bonus that outweighs the time penalty!"

    **The Result:** The AI learned to navigate, slow down, and land **between the flags**.

    **Score:** Consistently **+200 to +270** (officially "solved"!) 🎉
    """)

    corrected_video = os.path.join(videos_folder, "corrected.mp4")
    if os.path.exists(corrected_video):
        st.video(corrected_video)
    else:
        # Look for any landing video
        landing_videos = [f for f in os.listdir(videos_folder) if f.endswith('.mp4')] if os.path.exists(
            videos_folder) else []
        if landing_videos:
            st.video(os.path.join(videos_folder, landing_videos[0]))
            st.info(f"Showing: {landing_videos[0]}")
        else:
            st.warning("No corrected video found yet. Run `07_watch_corrected.py` to record one!")

st.markdown("---")

# Technical details section
with st.expander("🔧 Technical Details"):
    st.markdown("""
    ### How it works:

    **Environment:** [Lunar Lander v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) from Gymnasium

    **Algorithm:** PPO (Proximal Policy Optimization) from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

    **Training Steps:**
    - Baseline: 500,000 steps
    - Hover Bonus: 300,000 steps  
    - Corrected: 1,000,000 steps

    ### The Reward Functions:

    **Baseline (built-in):**
    - Distance to pad, speed, tilt, leg contact, fuel cost

    **Hover Bonus (flawed):**
    - Baseline + `+1.0` per frame if upright and slow

    **Corrected:**
    - Baseline + `-0.1` per frame (time penalty)

    ### Key Insight:
    > "You get exactly what you reward — not what you want."

    This is why reward design is one of the hardest parts of AI research.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <h3>Built with Python, Gymnasium, Stable-Baselines3, and Streamlit</h3>
    <p>A reinforcement learning project demonstrating reward shaping and reward hacking.</p>
</div>
""", unsafe_allow_html=True)