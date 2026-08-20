"""DIMEX's raw Fancy Gym task registrations.

The optional DMC, MetaWorld, and OpenAI-Gym namespaces target incompatible
dependency stacks.  DIMEX uses only the custom raw tasks registered in envs.
"""

from . import envs as fancy
from .envs.registry import register, upgrade


def make(*args, **kwargs):
    """
    As part of the refactor of Fancy Gym and upgrade to gymnasium the use of fancy_gym.make has been discontinued. Regular gym.make should be used instead. For more details check out the github README. If your codebase was build for older versions of Fancy Gym and relies on the old behavior and dependency versions, please check out the legacy branch.
    """
    raise Exception('As part of the refactor of Fancy Gym and upgrade to gymnasium the use of fancy_gym.make has been discontinued. Regular gym.make should be used instead. For more details check out the github README. If your codebase was build for older versions of Fancy Gym and relies on the old behavior and dependency versions, please check out the legacy branch.')
