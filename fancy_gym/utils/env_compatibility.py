import gymnasium as gym
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0


class EnvCompatibility(GymV21CompatibilityV0):
    """Gymnasium-1 replacement for the removed compatibility wrapper."""

    def __init__(self, old_env, render_mode=None):
        super().__init__(env=old_env, render_mode=render_mode)

    def __getattr__(self, item):
        """Propagate only non-existent properties to wrapped env."""
        if item.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(item))
        if item in self.__dict__:
            return getattr(self, item)
        return getattr(self.env, item)
