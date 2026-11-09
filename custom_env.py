import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander


class CustomLunarLander(LunarLander):
    def __init__(self, target_x=0.0, terrain_seed=0, **kwargs):
        super().__init__(**kwargs)
        self.custom_target_x = target_x
        self.terrain_seed = terrain_seed

    def reset(self, **kwargs):
        # 1. Reset the game with a NEW seed to generate a different landscape!
        obs, info = super().reset(seed=self.terrain_seed, **kwargs)

        # 2. Move the landing pad to our custom X position
        shift = self.custom_target_x

        # The pad is 1.0 unit wide. We shift the left and right edges.
        self.helipad_x1 += shift
        self.helipad_x2 += shift

        # 3. Move the physical Box2D bodies (the pad and the flags)
        for body in self.world.bodies:
            if body.userData == "helipad" or body.userData == "flag":
                pos = body.position
                # Create a new position shifted by our amount
                new_pos = (pos[0] + shift, pos[1])
                body.position = new_pos
                body.linearVelocity = (0, 0)
                body.angularVelocity = 0

        # Take a dummy step to update the observation with the new pad location
        obs, _, _, _, info = self.step(0)
        return obs, info