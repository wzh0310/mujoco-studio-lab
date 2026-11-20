# -*- coding: utf-8 -*-
"""Gymnasium registration for the PushCube environment.

Importing this package registers ``PushCube-v0`` in Gymnasium's global registry,
so ``gym.make("PushCube-v0")`` works after ``pip install -e`` of this repo.

The legacy ``envs.lerobot_push_cube_env`` module additionally registers the
namespaced ``gym_push_cube/PushCube-v0`` and ``gym_push_cube_raw/PushCube-v0``
ids used by the LeRobot integration; those are intentionally left untouched.
"""
import gymnasium as gym

from .push_cube_env import PushCubeEnv  # noqa: F401

GYM_ID = "PushCube-v0"


def _register():
    if GYM_ID not in gym.registry:
        gym.register(
            id=GYM_ID,
            entry_point="pushcube_envs:PushCubeEnv",
            max_episode_steps=50,
            disable_env_checker=True,
        )


_register()

__all__ = ['PushCubeEnv', 'GYM_ID']
