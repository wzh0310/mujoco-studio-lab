# -*- coding: utf-8 -*-
"""Canonical preprocessing rules for the push_cube LeRobot dataset (Day-3).

This module is the SINGLE SOURCE OF TRUTH for observation/action normalization.
Both the ACT training pipeline (via LeRobot's stats-derived processor) and the
evaluation script import these constants so that preprocessing is identical.

Rules
-----
Images (``observation.images.main``):
    LeRobotDataset decodes a uint8 HWC frame and divides by 255 -> float [0, 1].
    We define the canonical model input as **[-1, 1]** (a standard convention):

        x_model = x_[0,1] * 2 - 1

    NOTE: The ACT config uses ``use_imagenet_stats: true``. That path applies
    ImageNet mean/std inside the policy processor instead. To keep training and
    eval identical, the evaluation script does NOT manually re-normalize images;
    it feeds uint8 HWC frames to the policy exactly like the dataset does, and
    the policy's own processor performs the normalization.  ``normalize_image``
    below is provided for external consumers / visualisation and documents the
    [-1, 1] convention explicitly.

Actions (``action`` = [dx, dy, dz, gripper]):
    Stored raw in the dataset.  The policy operates in a normalized space.
    LeRobot's ACT normalizer uses per-dim mean/std from meta/stats.json
    (z-score).  We ALSO expose an explicit min-max -> [-1, 1] transform driven
    by the raw action bounds of the simulator action_space, which is used by the
    evaluation script when a policy is loaded WITHOUT its processor.  When the
    processor is available (normal case) we defer to it, so train/eval match.

    Raw action bounds (from envs/push_cube_env.py action_space):
        dx, dy, dz in [-0.02, 0.02];  gripper in [0, 1].
"""
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATS_PATH = os.path.join(REPO, "data", "push_cube_demos", "meta", "stats.json")

# Raw action_space bounds (metres / unit gripper)
ACTION_MIN = np.array([-0.02, -0.02, -0.02, 0.0], dtype=np.float64)
ACTION_MAX = np.array([0.02, 0.02, 0.02, 1.0], dtype=np.float64)

IMAGE_MEAN = 0.5
IMAGE_SCALE = 0.5   # x_model = (x_[0,1] - 0.5) / 0.5 == x_[0,1]*2 - 1


def normalize_image(img):
    """uint8 HWC -> float32 CHW normalized to [-1, 1]."""
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 3 and x.shape[-1] in (1, 3, 4):
        x = np.transpose(x, (2, 0, 1))       # HWC -> CHW
    x = x / 255.0
    return ((x - IMAGE_MEAN) / IMAGE_SCALE).astype(np.float32)


def normalize_action_minmax(action):
    """Raw 4-D action -> [-1, 1] using the simulator action_space bounds."""
    a = np.asarray(action, dtype=np.float64)
    return (2.0 * (a - ACTION_MIN) / (ACTION_MAX - ACTION_MIN) - 1.0).astype(np.float32)


def denormalize_action_minmax(action_norm):
    """Inverse of :func:`normalize_action_minmax`."""
    a = np.asarray(action_norm, dtype=np.float64)
    return (ACTION_MIN + (a + 1.0) * (ACTION_MAX - ACTION_MIN) / 2.0).astype(np.float32)


def load_action_stats(path=STATS_PATH):
    """Return (mean, std) from the dataset stats for z-score (LeRobot ACT) path."""
    with open(path, "r", encoding="utf-8") as fh:
        stats = json.load(fh)
    act = stats["action"]
    mean = np.asarray(act["mean"], dtype=np.float64)
    std = np.asarray(act["std"], dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)     # gripper dim has std == 0
    return mean, std


def zscore_action(action, mean=None, std=None):
    """Raw action -> z-scored action (matches LeRobot ACT normalizer)."""
    if mean is None or std is None:
        mean, std = load_action_stats()
    a = np.asarray(action, dtype=np.float64)
    return ((a - mean) / std).astype(np.float32)


def unnormalize_zscore(action_norm, mean=None, std=None):
    """Inverse of :func:`zscore_action`."""
    if mean is None or std is None:
        mean, std = load_action_stats()
    a = np.asarray(action_norm, dtype=np.float64)
    return (a * std + mean).astype(np.float32)


if __name__ == "__main__":
    print("Preprocessing rules (push_cube):")
    print("  image  : uint8 [0,255] -> /255 -> x*2-1 -> [-1,1]")
    print("  action : raw -> min-max -> [-1,1]  bounds=", ACTION_MIN, ACTION_MAX)
    mean, std = load_action_stats()
    print("  ACT z-score mean=", mean, " std=", std)
    print("  roundtrip minmax:", denormalize_action_minmax(normalize_action_minmax([0.02, 0.0, -0.01, 0.5])))
