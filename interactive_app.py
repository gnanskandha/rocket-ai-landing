# Lunar lander lab
# Loads the PPO models I trained and lets you change the environment
# (terrain, gravity, wind, a sideways shove at the start) to see how they cope.

import glob
import os
import time

import gymnasium as gym
import numpy as np
import pandas as pd
import streamlit as st
from stable_baselines3 import PPO

HERE = os.path.dirname(os.path.abspath(__file__))
ACTIONS = ["nothing", "left engine", "main engine", "right engine"]
NAMES = {
    "ppo_lander_baseline": "baseline",
    "ppo_lander_hover": "hover bonus",
    "ppo_lander_corrected": "corrected reward",
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


def run_episode(model, seed, gravity, wind, kick, render=False, on_frame=None, delay=0.0):
    env = make_env(gravity, wind, render)
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
            if on_frame and steps % 2 == 0:  # drawing every frame is too slow online
                on_frame(env.render(), obs, int(action), total, steps)
                time.sleep(delay)
    finally:
        env.close()
    # the env gives exactly +100 for a landing and -100 for a crash / leaving the screen
    result = "landed" if last >= 100 else "crashed" if last <= -100 else "timeout"
    return total, steps, trace, result


# ---------- sidebar ----------
models = find_models()
if not models:
    st.error("No ppo_lander_*.zip found next to interactive_app.py. Copy the trained .zip files into the repo root and push them to GitHub.")
    st.stop()

names = list(models)
st.sidebar.header("Settings")
choice = st.sidebar.selectbox("model", names, index=names.index("corrected reward") if "corrected reward" in names else 0)
seed = st.sidebar.number_input("terrain seed", 0, 9999, 0)
gravity = st.sidebar.slider("gravity", -11.5, -2.0, -10.0, 0.5)
wind = st.sidebar.slider("wind", 0.0, 20.0, 0.0, 1.0)
kick = st.sidebar.slider("sideways shove at start", -20, 20, 0, 1)
delay_ms = st.sidebar.slider("delay per frame (ms)", 0, 60, 15, 5)

# ---------- main page ----------
st.title("Lunar lander lab")
st.write(
    "Three PPO agents, same game, different reward functions. One of them is rewarded "
    "for hovering, one for landing fast. Change the conditions and see which one holds up."
)

tab_fly, tab_test = st.tabs(["Fly one episode", "Stress test"])

with tab_fly:
    go = st.button("Launch", type="primary")
    left, right = st.columns([3, 1])
    view = left.empty()
    readout = right.empty()

    if go:
        def draw(frame, obs, action, total, steps):
            x, y, vx, vy, ang, _, l1, l2 = obs
            view.image(frame)
            readout.code(
                f"step    {steps:>5}\n"
                f"engine  {ACTIONS[action]}\n"
                f"x       {x:>7.2f}\n"
                f"y       {y:>7.2f}\n"
                f"vx      {vx:>7.2f}\n"
                f"vy      {vy:>7.2f}\n"
                f"angle   {ang:>7.2f}\n"
                f"legs    {int(l1)} {int(l2)}\n"
                f"reward  {total:>7.1f}",
                language=None,
            )

        total, steps, trace, result = run_episode(
            load_model(models[choice]), seed, gravity, wind, kick, True, draw, delay_ms / 1000
        )
        msg = f"{result}: {steps} steps, total reward {total:.1f}"
        {"landed": st.success, "crashed": st.error}.get(result, st.warning)(msg)
        st.line_chart(pd.DataFrame({"total reward": trace}))
    else:
        env = make_env(gravity, wind, True)
        env.reset(seed=seed)
        view.image(env.render())
        env.close()
        readout.caption("Set things up in the sidebar, then hit Launch.")

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