# Lunar Lander: reward shaping and reward hacking

I trained PPO agents to land the LunarLander from Gymnasium, then changed the reward function to see what the agent does with it. The main thing I wanted to show is that an agent optimises the reward you write, not the goal you had in mind.

**Live app:** https://rocket-ai-landing-u52e6ggcf6u3g5iuwvausq.streamlit.app  
**Project page:** https://github.com/gnanskandha/rocket-ai-landing

![results]<img width="1050" height="570" alt="results" src="https://github.com/user-attachments/assets/5652a9f7-4e7a-4f24-82a0-99e43c8967d6" />


## Results

Stress test tab of the app: 30 episodes per model, seeds 0 to 29, gravity -10, no wind, no sideways shove.

| Model | Landed % | Crashed % | Timeout % | Avg reward |
|---|---|---|---|---|
| baseline | 40.0 | 60.0 | 0.0 | 31.9 |
| hover bonus | 0.0 | 16.7 | 83.3 | -78.5 |
| corrected reward | 100.0 | 0.0 | 0.0 | 284.8 |
| robust (random gravity) | 96.7 | 3.3 | 0.0 | 267.3 |

What the numbers show:

- **Hover bonus never landed.** 83% of its episodes ran into the 1000-step time limit without landing or crashing, which fits an agent that learned to stay in the air to collect the +1 per step.
- **Removing the hover bonus and adding a small time penalty fixed it.** The corrected agent landed in all 30 episodes.
- **The baseline is weaker, but not a clean comparison.** It landed 40% of the time, but it also trained for 500k steps against 3M for the corrected one, so part of the gap is just training time and not only the reward.
- **The robust model gives up very little at normal gravity** (96.7% vs 100%), even though it was trained on random gravity.

## The experiments

All agents use PPO (Stable-Baselines3, MlpPolicy, seed 42) on `LunarLander-v3`. Only the reward changes, using the wrapper in `reward_experiments.py`.

| Model | Reward | Steps | File |
|---|---|---|---|
| baseline | the normal environment reward | 500k | `ppo_lander_baseline.zip` |
| hover bonus | +1 every step the lander is upright and almost still, even in mid-air | 300k | `ppo_lander_hover.zip` |
| corrected reward | no hover bonus, small time penalty of 0.1 per step | 3M | `ppo_lander_corrected.zip` |
| robust | corrected reward, gravity random in [-11.5, -2] each episode, continued from the corrected model | +2M | `ppo_lander_robust.zip` |

## The interactive app

`interactive_app.py` is a Streamlit app that loads the trained models and lets you test them:

- choose which model flies
- terrain seed, wind, and a sideways shove at the start
- the flight is simulated first and then played back as a gif, with a reward-over-time chart
- a stress test tab that runs each model on many seeds and counts landings, crashes and timeouts
- gravity is fixed at -10 by default. It can be unlocked under "advanced", where the models other than the robust one are expected to crash more

## Limitations

- One training seed per model, so these are not statistically solid comparisons.
- The baseline and corrected models differ in training length as well as reward (see above).
- Models other than the robust one only trained at gravity -10. Gravity is not part of the observation, so the agent cannot tell when the physics changes.
- The robust model was only tested at gravity -10 here, so these results say nothing yet about how it does at low gravity.
- Success is judged by the environment's own signal: +100 on landing, -100 on a crash or leaving the screen, otherwise a timeout.

## Next steps

- Run the same stress test at low gravity (for example -3) and compare corrected vs robust.
- Train a few seeds per model and report the spread.
- Train the baseline for 3M steps so the comparison with the corrected reward is fair.

## Run it locally

```
pip install -r requirements.txt
python 06_train_corrected.py      # trains the corrected model (slow, 3M steps)
streamlit run interactive_app.py
```

The app looks for `ppo_lander_*.zip` in the same folder as `interactive_app.py`, and the training scripts save the models there too.

On Windows, `gymnasium[box2d]` sometimes fails to build. Installing `swig` first (`pip install swig`) usually fixes it.

## Files

```
interactive_app.py        streamlit app
reward_experiments.py     reward wrapper (baseline / time_penalty / hover_bonus / corrected)
gravity_wrapper.py        random gravity wrapper for the robust model
results.png               chart of the table above
flower.py                 random agent, just to see the environment
banana.py                 train baseline
04_train_experiment.py    train with the hover bonus
06_train_corrected.py     train with the corrected reward
08_train_robust.py        train with random gravity
3.py                      watch the baseline agent
05_watch_hover.py         watch the hover agent
07_watch_corrected.py     record the corrected agent to mp4
requirements.txt          python packages
packages.txt              system packages for streamlit cloud
.streamlit/config.toml    app theme
```

## Deploying on Streamlit Community Cloud

1. Push the code and the `.zip` models to the repo root. Do not save models inside `.venv/`, because it is gitignored.
2. Main file: `interactive_app.py`.
3. `packages.txt` installs `swig`, which the Box2D build needs.
