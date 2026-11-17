#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Complete dataset metadata for native LeRobot 0.7.0 (v3) loading.

Run in the `lerobot` conda env:
    python scripts/prepare_dataset.py

What it does (keeps the v3 layout that LeRobotDataset can load natively):
  1. Writes meta/features.json  -- a standalone, human-readable definition of
     observation / action spaces (mirrors info.json["features"]).
  2. Recomputes action (and state) mean/std/min/max from the parquet rows and
     writes them to meta/stats.json (used by LeRobot for normalization).
  3. Sets info.json["splits"] = {"train": "0:4", "val": "4:5"} -> an 8:2
     episode split (4 train / 1 val = 146 / 183 frames ~ 80/20).

The v3 data/videos layout and episode numbering are left untouched, so
`LeRobotDataset(repo_id, root=...)` still loads the dataset natively.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "push_cube_demos"))
META = os.path.join(ROOT, "meta")


def main():
    info = json.load(open(os.path.join(META, "info.json")))
    stats = json.load(open(os.path.join(META, "stats.json")))

    # 1. standalone features definition (aligned with info.json features)
    features = info["features"]
    with open(os.path.join(META, "features.json"), "w") as f:
        json.dump({
            "observation": {
                "image": {"key": "observation.images.main", "dtype": features["observation.images.main"]["dtype"],
                          "shape": features["observation.images.main"]["shape"]},
                "state": {"key": "observation.state", "dtype": features["observation.state"]["dtype"],
                          "shape": features["observation.state"]["shape"],
                          "names": features["observation.state"]["names"]},
            },
            "action": {"key": "action", "dtype": features["action"]["dtype"],
                       "shape": features["action"]["shape"], "names": features["action"]["names"]},
            "episode_index": {"dtype": features["episode_index"]["dtype"]},
            "_source": "mirrors info.json['features'] (LeRobot v3 native schema)",
        }, f, indent=2)
    print("[prepare] wrote meta/features.json")

    # 2. recompute action/state statistics from the actual parquet rows
    pqs = sorted(glob.glob(os.path.join(ROOT, "data", "chunk-*", "*.parquet")))
    assert pqs, "no parquet files found"
    df = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
    for key, shape in [("action", (4,)), ("observation.state", (13,))]:
        arr = np.stack(df[key].to_numpy()).astype(np.float64)
        assert arr.shape[1] == shape[0], (key, arr.shape)
        stats[key] = {
            "mean": [round(float(x), 6) for x in arr.mean(0)],
            "std": [round(float(x), 6) for x in arr.std(0)],
            "min": [round(float(x), 6) for x in arr.min(0)],
            "max": [round(float(x), 6) for x in arr.max(0)],
        }
        print("[prepare] %s mean=%s std=%s" % (key, stats[key]["mean"], stats[key]["std"]))
    with open(os.path.join(META, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("[prepare] wrote meta/stats.json")

    # 3. 8:2 episode split (LeRobot native splits field)
    n_ep = int(info["total_episodes"])
    n_train = max(1, int(round(n_ep * 0.8)))
    info["splits"] = {"train": "0:%d" % n_train, "val": "%d:%d" % (n_train, n_ep)}
    with open(os.path.join(META, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("[prepare] info.json splits=%s (train %d ep / val %d ep)" % (
        info["splits"], n_train, n_ep - n_train))
    print("PREPARE DATASET DONE")


if __name__ == "__main__":
    main()
