# -*- coding: utf-8 -*-
"""Minimal smoke test: random actions + scripted push, print reward / obs shapes."""
import numpy as np
from push_cube_env import PushCubeEnv

env = PushCubeEnv()
obs, info = env.reset(seed=0)
print('image:', obs['image'].shape, obs['image'].dtype, '| state:', obs['state'].shape, obs['state'].dtype)
print('--- 50 random actions ---')
for i in range(50):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if i % 25 == 0:
        print(f'step {i:3d}  reward {r:+.3f}  dist {info["cube_target_dist"]:.3f}  term={term} trunc={trunc}')
print('--- scripted push toward the cube (reward must respond) ---')
env.reset(seed=0)
seq = [([0.02, -0.005, 0.0, 0.0], 25), ([0.0, 0.0, -0.012, 0.0], 8), ([0.02, 0.0, 0.0, 0.0], 15)]
for a, n in seq:
    for _ in range(n):
        obs, r, term, trunc, info = env.step(a)
    print(f'action {a[:3]} x{n}: reward {r:+.3f}  dist {info["cube_target_dist"]:.3f}  cube {np.round(info["cube_pos"],3)}')
env.close()
print('OK')
