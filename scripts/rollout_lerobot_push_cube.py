# -*- coding: utf-8 -*-
"""Rollout validation: run a full episode with a random policy through
LeRobot's native vector env + official ``lerobot_eval.rollout`` machinery, and
save an evaluation video.

Run in the `lerobot` conda environment.  Produces (in the container repo):
  assets/eval_random_rollout.mp4   - evaluation video of one full episode
  data/eval_rollout_metrics.json   - reward/success metrics from LeRobot rollout
"""
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "envs"))

import lerobot_push_cube_env as lc  # noqa: E402

from lerobot.envs import make_env, make_env_config  # noqa: E402
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402
from lerobot.processor.converters import (  # noqa: E402
    policy_action_to_transition, transition_to_policy_action)
from lerobot.utils.io_utils import write_video  # noqa: E402
from lerobot.scripts.lerobot_eval import rollout  # noqa: E402


class RandomPolicy(torch.nn.Module):
    """Minimal policy conforming to LeRobot's policy call interface.

    ``select_action`` receives a batch of canonical observation tensors and
    returns ``{"action": tensor(b, 4)}`` in normalized space; the environment
    de-normalizes internally.  A mild forward bias makes the random rollout
    actually approach the target so the video is meaningful.
    """

    def __init__(self, action_dim=4):
        super().__init__()
        self.action_dim = action_dim
        self.config = None  # no checkpoint features to validate

    def reset(self):
        pass

    def select_action(self, observation, **kwargs):
        b = 1
        for v in observation.values():
            if torch.is_tensor(v) and v.ndim >= 1:
                b = v.shape[0]
                break
        action = torch.zeros((b, self.action_dim), dtype=torch.float32)
        action[:, 0] = 0.6 + 0.2 * torch.randn(b)   # push +x toward target
        action[:, 1] = 0.1 * torch.randn(b)
        action[:, 2] = 0.1 * torch.randn(b)
        return {"action": action}

    def forward(self, observation):
        return self.select_action(observation)


def main():
    print("=" * 60)
    print("[rollout] building LeRobot native env (push_cube)")
    cfg = make_env_config("push_cube")
    envs = make_env(cfg, n_envs=1)
    first = next(v for g in envs.values() for v in g.values())
    descriptor = getattr(first, "descriptor", None)
    print("    descriptor fps:", descriptor.fps if descriptor else "n/a",
          "| action:", tuple(descriptor.action_feature.shape) if descriptor else "n/a",
          "| features:", [f.name for f in descriptor.features] if descriptor else "n/a")

    # Identity processors (random policy needs no normalization).  The policy
    # postprocessor must use the tensor converters that rollout expects.
    pre = PolicyProcessorPipeline(steps=[])
    post = PolicyProcessorPipeline(
        steps=[],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    env_pre = PolicyProcessorPipeline(steps=[])
    env_post = PolicyProcessorPipeline(steps=[])
    policy = RandomPolicy(action_dim=4)

    # Capture frames via the official render_callback hook. LocalVectorEnv
    # implements call("render"); fall back to env.envs[i] if present.
    frames = []

    def render_cb(vec_env):
        try:
            rendered = vec_env.call("render")
        except Exception:
            rendered = tuple(vec_env.envs[i].render()
                             for i in range(vec_env.num_envs))
        for frame in rendered:
            frames.append(np.ascontiguousarray(frame, dtype=np.uint8))

    print("[rollout] running 1 full episode via lerobot.scripts.lerobot_eval.rollout")
    data = rollout(
        env=first,
        policy=policy,
        env_preprocessor=env_pre,
        preprocessor=pre,
        postprocessor=post,
        env_postprocessor=env_post,
        seeds=[7],
        return_observations=False,
        render_callback=render_cb,
    )
    steps = int(data["reward"].shape[1])
    total_reward = float(data["reward"].sum().item())
    success = bool(data["success"].any().item())
    print(f"    steps={steps} total_reward={total_reward:.3f} success={success}")
    print(f"    rendered_frames={len(frames)}")

    # Save evaluation video in the container repo.
    os.makedirs(os.path.join(REPO, "assets"), exist_ok=True)
    video_path = os.path.join(REPO, "assets", "eval_random_rollout.mp4")
    if frames:
        stacked = np.stack(frames)  # (T, H, W, 3)
        write_video(video_path, stacked, fps=50)
        print("    wrote", video_path, stacked.shape)

    metrics = {
        "steps": steps,
        "total_reward": total_reward,
        "success": success,
        "rendered_frames": len(frames),
        "policy": "RandomPolicy (mild forward bias)",
        "env": "push_cube (LeRobot native vector env)",
        "rollout_api": "lerobot.scripts.lerobot_eval.rollout",
    }
    os.makedirs(os.path.join(REPO, "data"), exist_ok=True)
    with open(os.path.join(REPO, "data", "eval_rollout_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    first.close()
    print("=" * 60)
    print("ROLLOUT LEROBOT PUSH CUBE: PASS")


if __name__ == "__main__":
    main()
