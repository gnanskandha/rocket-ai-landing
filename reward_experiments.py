import gymnasium as gym

class RewardExperiment(gym.Wrapper):
    def __init__(self, env, mode="baseline"):
        super().__init__(env)
        self.mode = mode

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # EXPERIMENT B: Time Penalty
        if self.mode == "time_penalty":
            reward -= 0.1

        # EXPERIMENT C: The Flawed Hover Bonus (Reward Hacking)
        elif self.mode == "hover_bonus":
            vx = float(obs[2])
            vy = float(obs[3])
            angle = float(obs[4])
            upright = abs(angle) < 0.15
            slow = abs(vx) < 0.1 and abs(vy) < 0.1
            if upright and slow and not (terminated or truncated):
                reward += 1.0

        # EXPERIMENT D: The Corrected Reward
        elif self.mode == "corrected":
            reward -= 0.1

        return obs, reward, terminated, truncated, info