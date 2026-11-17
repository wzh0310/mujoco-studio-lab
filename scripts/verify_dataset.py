#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify native LeRobot loading of the custom push-cube dataset.

Run in the `lerobot` conda env:
    python scripts/verify_dataset.py
"""
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.join(REPO, "data", "push_cube_demos")
REPO_ID = "push_cube_expert"

# --- 1. native load: total size & episode count --------------------------------
ds = LeRobotDataset(REPO_ID, root=ROOT, video_backend="pyav")
print("total_frames =", ds.num_frames, " num_episodes =", ds.num_episodes, " fps =", ds.fps)

# --- 2. batch: check image / action shape & dtype ------------------------------
batch = next(iter(DataLoader(ds, batch_size=8, shuffle=True)))
img, act = batch["observation.images.main"], batch["action"]
assert tuple(img.shape[1:]) == (3, 224, 224) and img.dtype == torch.float32
assert tuple(act.shape[1:]) == (4,) and act.dtype == torch.float32
print("batch image", tuple(img.shape), img.dtype, "| action", tuple(act.shape), act.dtype)

# --- 3. 8:2 split via native `episodes` (mirrors info.json splits) --------------
info = json.load(open(os.path.join(ROOT, "meta", "info.json")))
a, b = info["splits"]["train"].split(":"), info["splits"]["val"].split(":")
tr = LeRobotDataset(REPO_ID, root=ROOT, video_backend="pyav", episodes=list(range(int(a[0]), int(a[1]))))
va = LeRobotDataset(REPO_ID, root=ROOT, video_backend="pyav", episodes=list(range(int(b[0]), int(b[1]))))
print("split train frames =", tr.num_frames, " val frames =", va.num_frames,
      " ratio = %.1f/%.1f" % (100 * tr.num_frames / ds.num_frames, 100 * va.num_frames / ds.num_frames))

# --- 4. random frame: visualize image + action, confirm alignment --------------
rng = np.random.default_rng(7)
i = int(rng.integers(ds.num_frames))
s = ds[i]
frame = (s["observation.images.main"].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).numpy()
im = Image.fromarray(frame).resize((448, 448))
canvas = Image.new("RGB", (448, 500), "white")
canvas.paste(im, (0, 0))
d = ImageDraw.Draw(canvas)
act_list = [round(float(x), 5) for x in s["action"]]
d.text((8, 454), "ep=%d frame=%d t=%.2f" % (s["episode_index"], s["frame_index"], s["timestamp"]), fill="black")
d.text((8, 472), "action[dx,dy,dz,gripper]=%s" % act_list, fill="black")
out = os.path.join(REPO, "assets", "dataset_sample.png")
canvas.save(out)
print("random frame i=%d -> %s" % (i, out))

# --- 5. normalization: action (x-mean)/std is well-scaled ----------------------
stats = json.load(open(os.path.join(ROOT, "meta", "stats.json")))["action"]
mean = np.array(stats["mean"])
std = np.array(stats["std"])
all_act = np.stack([ds[j]["action"].numpy() for j in range(ds.num_frames)])
norm = (all_act - mean) / np.where(std > 1e-6, std, 1.0)
print("action mean =", mean.round(5).tolist(), " std =", std.round(5).tolist())
print("normalized action range = [%.3f, %.3f] (gripper const-dim guarded)" % (norm.min(), norm.max()))
assert np.isfinite(norm).all() and np.abs(norm[:, :3]).max() < 8
print("VERIFY DATASET OK")
