# Lunar lander lab
# Loads the PPO models I trained and lets you change the environment
# (terrain, gravity, wind, a sideways shove at the start) to see how they cope.

import glob
import io
import os

import gymnasium as gym
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from stable_baselines3 import PPO

HERE = os.path.dirname(os.path.abspath(__file__))
ACTIONS = ["nothing", "left engine", "main engine", "right engine"]
NAMES = {
    "ppo_lander_baseline": "baseline",
    "ppo_lander_hover": "hover bonus",
    "ppo_lander_corrected": "corrected reward",
    "ppo_lander_robust": "robust (random gravity)",
}

st.set_page_config(page_title="Lunar lander lab", layout="wide")
st.markdown(
    "<style>footer {visibility: hidden;} .block-container {padding-top: 2rem;}</style>",
    unsafe_allow_html=True,
)


def find_models():
    """every ppo_lander_*.zip sitting next to this file"""
    found = {}
    for path in sorted(glob.glob(os.path.join(HERE, "ppo_lander_*.zip"))):
        key = os.path.basename(path)[:-4]
        found[NAMES.get(key, key)] = path[:-4]
    return found


@st.cache_resource
def load_model(path):
    # custom_objects stops the load from breaking if the server runs a different
    # python version than my laptop (the saved lr schedule is a pickled function)
    custom = {"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0}
    return PPO.load(path, device="cpu", custom_objects=custom)


def make_env(gravity, wind, render):
    return gym.make(
        "LunarLander-v3",
        render_mode="rgb_array" if render else None,
        gravity=gravity,
        enable_wind=wind > 0,
        wind_power=wind,
        turbulence_power=wind / 10,
    )


def run_episode(model, seed, gravity, wind, kick, record=False):
    env = make_env(gravity, wind, record)
    frames = []
    try:
        obs, _ = env.reset(seed=seed)
        if kick:
            lander = env.unwrapped.lander
            lander.ApplyLinearImpulse((kick, 0), lander.worldCenter, True)
        total, steps, last, done, trace = 0.0, 0, 0.0, False, []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            total += r
            steps += 1
            last = r
            done = term or trunc
            trace.append(total)
            if record and steps % 2 == 0:  # every 2nd frame is enough for a smooth gif
                frames.append(env.render())
    finally:
        env.close()
    # the env gives exactly +100 for a landing and -100 for a crash / leaving the screen
    result = "landed" if last >= 100 else "crashed" if last <= -100 else "timeout"
    return total, steps, trace, result, frames


def make_gif(frames, ms):
    """streaming frames live to the browser kept dropping them, so I play back a gif instead"""
    imgs = [
        Image.fromarray(f).resize((480, 320)).convert("P", palette=Image.ADAPTIVE, colors=64)
        for f in frames
    ]
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:], duration=ms, loop=0)
    return buf.getvalue()


# ---------- sidebar ----------
models = find_models()
if not models:
    st.error("No ppo_lander_*.zip found next to interactive_app.py. Copy the trained .zip files into the repo root and push them to GitHub.")
    st.stop()

names = list(models)
st.sidebar.header("Settings")
choice = st.sidebar.selectbox("model", names, index=names.index("corrected reward") if "corrected reward" in names else 0)
seed = st.sidebar.number_input("terrain seed", 0, 9999, 0)
gravity = st.sidebar.slider("gravity", -11.0, -5.0, -10.0, 0.5)
wind = st.sidebar.slider("wind", 0.0, 20.0, 0.0, 1.0)
kick = st.sidebar.slider("sideways shove at start", -20, 20, 0, 1)
frame_ms = st.sidebar.slider("playback (ms per frame, higher = slower)", 20, 100, 40, 10)

# ---------- main page ----------
st.title("Lunar lander lab")
st.write(
    "Three PPO agents, same game, different reward functions. One of them is rewarded "
    "for hovering, one for landing fast. Change the conditions and see which one holds up."
)

tab_fly, tab_test = st.tabs(["Fly one episode", "Stress test"])

with tab_fly:
    go = st.button("Launch", type="primary")
    view = st.empty()

    if go:
        with st.spinner("simulating the flight..."):
            total, steps, trace, result, frames = run_episode(
                load_model(models[choice]), seed, gravity, wind, kick, record=True
            )
            if frames:
                view.image(make_gif(frames, frame_ms))
        msg = f"{result}: {steps} steps, total reward {total:.1f}"
        {"landed": st.success, "crashed": st.error}.get(result, st.warning)(msg)
        c1, c2, c3 = st.columns(3)
        c1.metric("steps", steps)
        c2.metric("total reward", f"{total:.1f}")
        c3.metric("result", result)
        st.line_chart(pd.DataFrame({"total reward": trace}))
    else:
        env = make_env(gravity, wind, True)
        env.reset(seed=seed)
        view.image(env.render())
        env.close()
        st.caption("Set things up in the sidebar, then hit Launch.")

with tab_test:
    st.write("Runs each model on seeds 0..N with the current gravity / wind / shove, no video, and counts what happens.")
    picks = st.multiselect("models", names, default=names)
    n = st.slider("episodes per model", 5, 30, 10)
    if st.button("Run stress test") and picks:
        rows, bar = [], st.progress(0.0)
        for i, name in enumerate(picks):
            model = load_model(models[name])
            res = [run_episode(model, s, gravity, wind, kick) for s in range(n)]
            rows.append({
                "model": name,
                "landed %": 100 * np.mean([r[3] == "landed" for r in res]),
                "crashed %": 100 * np.mean([r[3] == "crashed" for r in res]),
                "timeout %": 100 * np.mean([r[3] == "timeout" for r in res]),
                "avg reward": np.mean([r[0] for r in res]),
            })
            bar.progress((i + 1) / len(picks))
        df = pd.DataFrame(rows).set_index("model").round(1)
        st.dataframe(df)
        st.bar_chart(df[["landed %", "crashed %", "timeout %"]])
