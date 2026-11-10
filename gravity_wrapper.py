# Picks a new random gravity at the start of every episode so the agent
# doesn't only learn one set of physics.
import numpy as np
import gymnasium as gym


class RandomGravity(gym.Wrapper):
    def __init__(self, env, low=-11.5, high=-2.0):
        super().__init__(env)
        self.low = low
        self.high = high
        self.rng = np.random.default_rng()

    def reset(self, **kwargs):
        g = self.rng.uniform(self.low, self.high)
        # the box2d world is created once in __init__, so I just change its gravity here
        self.env.unwrapped.world.gravity = (0, g)
        return self.env.reset(**kwargs)