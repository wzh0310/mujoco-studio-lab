# -*- coding: utf-8 -*-
"""Single-step verification for the LeRobot-compatible push_cube env.

Run in the `lerobot` conda environment.  Verifies:
  1. Registration: gym.make(GYM_ID) and make_env_config("push_cube").
  2. reset() returns canonical keys identical to the dataset.
  3. 10 random steps: obs shape/dtype, reward finite, done flags, no errors.
  4. render() returns an RGB frame; close() releases resources.
  5. LeRobot native vector env (make_env) produces canonical observations.
"""
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "envs"))

import lerobot_push_cube_env as lc  # noqa: E402

OBS_IMG = "observation.images.main"
OBS_STATE = "observation.state"

print("=" * 60)
print("[1] gym registry")
import gymnasium as gym  # noqa: E402
assert lc.GYM_ID in gym.registry, "canonical env not registered"
assert lc.RAW_GYM_ID in gym.registry, "raw env not registered"
print("    registered:", lc.GYM_ID, "|", lc.RAW_GYM_ID)

env = gym.make(lc.GYM_ID, disable_env_checker=True)
obs, info = env.reset(seed=0)
assert OBS_IMG in obs and OBS_STATE in obs, f"keys: {list(obs)}"
assert obs[OBS_IMG].shape == (224, 224, 3) and obs[OBS_IMG].dtype == np.uint8
assert obs[OBS_STATE].shape == (13,) and obs[OBS_STATE].dtype == np.float32
print("    reset keys:", sorted(obs.keys()))
print("    image", obs[OBS_IMG].shape, obs[OBS_IMG].dtype,
      "state", obs[OBS_STATE].shape, obs[OBS_STATE].dtype)

print("=" * 60)
print("[2] 10 random steps (normalized actions)")
for t in range(10):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    assert obs[OBS_IMG].shape == (224, 224, 3)
    assert obs[OBS_STATE].shape == (13,)
    assert np.isfinite(reward), f"non-finite reward at step {t}"
    assert "raw_action" in info, "denorm info missing"
    print(f"    step {t:2d}: reward={reward:+.3f} term={int(term)} "
          f"trunc={int(trunc)} dist={info.get('cube_target_dist', float('nan')):.3f} "
          f"raw_a={np.round(info['raw_action'], 4)}")
    if term or trunc:
        obs, info = env.reset(seed=t + 1)

frame = env.render()
assert frame.shape == (224, 224, 3) and frame.dtype == np.uint8
print("    render OK:", frame.shape, frame.dtype, "mean", round(float(frame.mean()), 1))
env.close()
print("    close OK")

print("=" * 60)
print("[3] LeRobot EnvConfig registry + native make_env")
from lerobot.envs import make_env, make_env_config  # noqa: E402

cfg = make_env_config("push_cube")
print("    cfg.type =", cfg.type, "| task =", cfg.task)
envs = make_env(cfg, n_envs=1)
first = next(v for g in envs.values() for v in g.values())
vec_obs, vec_info = first.reset(seed=0)
print("    vector obs keys:", sorted(vec_obs.keys()))
assert OBS_IMG in vec_obs and OBS_STATE in vec_obs
assert vec_obs[OBS_IMG].shape == (1, 224, 224, 3)
assert vec_obs[OBS_STATE].shape == (1, 13)
print("    image", vec_obs[OBS_IMG].shape, vec_obs[OBS_IMG].dtype,
      "state", vec_obs[OBS_STATE].shape, vec_obs[OBS_STATE].dtype)

act = np.zeros((1, 4), dtype=np.float32)
vec_obs, reward, term, trunc, info = first.step(act)
print("    vector step: reward", np.round(reward, 3),
      "term", term, "trunc", trunc)
assert reward.shape == (1,)
first.close()
print("=" * 60)
print("TEST LEROBOT PUSH CUBE: PASS")
