# -*- coding: utf-8 -*-
"""Export & classify failure rollout trajectories of the baseline ACT policy.

Records per-step cube / EE trajectory for 3 seeds x 10 episodes using the exact
same seed scheme as eval_act.py (seed*1000+ep), so failures are reproducible.
Then classifies each FAILED episode into:
  T1_LOST       - EE never got near the cube (never approached) -> cube barely moved
  T2_MISPUSH    - EE reached the cube but pushed sideways (large lateral y error)
  T3_WEAK_PUSH  - EE reached the cube, direction OK, but not enough displacement
  T4_OTHER      - doesn't match the above

Run (lerobot env, GPU):
  python scripts/analyze_failures.py --checkpoint .../004000 \
      --out data/failure_trajectories.json --report data/failure_analysis.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import mujoco

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "envs"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_act import load_policy_from_checkpoint  # noqa: E402
from push_cube_env import PushCubeEnv, TARGET_XY  # noqa: E402


def run_traced_episode(env, policy, pre, post, device, seed, max_steps):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    dx = float(rng.uniform(-0.030, 0.030))
    dy = float(rng.uniform(-0.035, 0.035))
    env.data.qpos[9] += dx
    env.data.qpos[10] += dy
    env.data.qvel[9:15] = 0.0
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    env._prev_dist = env._cube_target_dist()

    cb = env._cube_body
    hb = env._hand_body
    cube_start = env.data.xpos[cb][:2].copy()
    init_dist = float(np.linalg.norm(cube_start - TARGET_XY))

    traj = []
    min_ee_cube = 1e9
    for t in range(max_steps):
        img_chw = np.transpose(obs["image"], (2, 0, 1)).astype(np.float32) / 255.0
        canon = {
            "observation.images.main": torch.from_numpy(
                np.ascontiguousarray(img_chw)[None]).to(device),
            "observation.state": torch.from_numpy(
                np.ascontiguousarray(obs["state"]).astype(np.float32)[None]).to(device),
        }
        with torch.inference_mode():
            processed = pre(canon)
            action = policy.select_action(processed)
            action = post(action)
        if isinstance(action, dict):
            action = action["action"]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        a = np.asarray(action).reshape(-1)[:4]
        raw = np.clip(a, env.action_space.low, env.action_space.high)
        obs, reward, term, trunc, info = env.step(raw)

        cube = env.data.xpos[cb][:3].copy()
        hand = env.data.xpos[hb][:3].copy()
        d_ee_cube = float(np.linalg.norm(hand[:2] - cube[:2]))
        min_ee_cube = min(min_ee_cube, d_ee_cube)
        traj.append({
            "t": int(t),
            "cube": [round(float(x), 5) for x in cube[:2]],
            "hand": [round(float(x), 5) for x in hand[:2]],
            "action": [round(float(x), 5) for x in raw],
            "dist": round(float(np.linalg.norm(cube[:2] - TARGET_XY)), 5),
            "ee_cube": round(d_ee_cube, 5),
        })
        if term or trunc:
            break

    cube_final = env.data.xpos[cb][:2].copy()
    final_dist = float(np.linalg.norm(cube_final - TARGET_XY))
    disp = float(np.linalg.norm(cube_final - cube_start))
    y_err = float(abs(cube_final[1] - TARGET_XY[1]))
    x_target = TARGET_XY[0]
    x_progress = float((cube_final[0] - cube_start[0]) /
                       (x_target - cube_start[0])) if (x_target - cube_start[0]) > 1e-6 else 0.0
    success = bool(final_dist < 0.03)
    return {
        "seed": seed, "steps": len(traj),
        "success": success,
        "init_dist": round(init_dist, 5),
        "final_dist": round(final_dist, 5),
        "cube_start": [round(float(x), 5) for x in cube_start],
        "cube_final": [round(float(x), 5) for x in cube_final],
        "displacement": round(disp, 5),
        "min_ee_cube": round(min_ee_cube, 5),
        "y_error": round(y_err, 5),
        "x_progress": round(x_progress, 4),
        "traj": traj,
    }


def classify(ep):
    """Classify a FAILED episode into failure type based on trajectory stats."""
    if ep["success"]:
        return "SUCCESS"
    # T1: EE never approached the cube -> cube essentially didn't move
    if ep["min_ee_cube"] > 0.10 or ep["displacement"] < 0.02:
        return "T1_LOST"
    # T2: reached cube but large lateral (y) error -> pushed sideways
    if ep["y_error"] > 0.05:
        return "T2_MISPUSH"
    # T3: reached cube, direction OK (small y error), but insufficient push
    if ep["x_progress"] < 0.85:
        return "T3_WEAK_PUSH"
    return "T4_OTHER"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300])
    ap.add_argument("--episodes-per-seed", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--out", default="data/failure_trajectories.json")
    ap.add_argument("--report", default="data/failure_analysis.json")
    args = ap.parse_args()

    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    rpt = args.report if os.path.isabs(args.report) else os.path.join(REPO, args.report)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print("=" * 64)
    print("[analyze] loading policy from", args.checkpoint)
    policy, pre, post, cfg = load_policy_from_checkpoint(args.checkpoint)
    device = next(policy.parameters()).device
    env = PushCubeEnv(image_obs=True, max_episode_steps=args.max_steps)

    episodes = []
    for seed in args.seeds:
        for ep in range(args.episodes_per_seed):
            e = run_traced_episode(env, policy, pre, post, device,
                                   seed=seed * 1000 + ep, max_steps=args.max_steps)
            e["group"] = seed
            e["ep"] = ep
            e["type"] = classify(e)
            episodes.append(e)
            print("[analyze] s%d e%02d %-11s final=%.3f disp=%.3f minEE=%.3f ye=%.3f xp=%.2f" % (
                seed, ep, e["type"], e["final_dist"], e["displacement"],
                e["min_ee_cube"], e["y_error"], e["x_progress"]))
    env.close()

    with open(out, "w") as fh:
        json.dump(episodes, fh, indent=1)

    # ---- aggregate ----
    from collections import Counter
    types = Counter(e["type"] for e in episodes)
    fails = [e for e in episodes if not e["success"]]
    by_type = {}
    for t in ["T1_LOST", "T2_MISPUSH", "T3_WEAK_PUSH", "T4_OTHER"]:
        group = [e for e in fails if e["type"] == t]
        if not group:
            by_type[t] = {"count": 0, "share_of_failures": 0.0}
            continue
        by_type[t] = {
            "count": len(group),
            "share_of_failures": round(len(group) / len(fails), 4),
            "avg_final_dist": round(float(np.mean([e["final_dist"] for e in group])), 4),
            "avg_displacement": round(float(np.mean([e["displacement"] for e in group])), 4),
            "avg_min_ee_cube": round(float(np.mean([e["min_ee_cube"] for e in group])), 4),
            "avg_y_error": round(float(np.mean([e["y_error"] for e in group])), 4),
            "avg_x_progress": round(float(np.mean([e["x_progress"] for e in group])), 4),
        }
    summary = {
        "checkpoint": args.checkpoint,
        "n_episodes": len(episodes),
        "n_failures": len(fails),
        "success_rate": round(types.get("SUCCESS", 0) / len(episodes), 4),
        "type_counts": dict(types),
        "by_type": by_type,
        "per_seed_success_rate": {
            str(s): round(sum(1 for e in episodes if e["group"] == s and e["success"]) /
                          args.episodes_per_seed, 4) for s in args.seeds
        },
    }
    with open(rpt, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("=" * 64)
    print("[analyze] summary:", json.dumps({
        "success_rate": summary["success_rate"],
        "type_counts": summary["type_counts"],
        "by_type_counts": {k: v["count"] for k, v in by_type.items()},
    }, ensure_ascii=False))
    print("[analyze] trajectories ->", out)
    print("[analyze] report ->", rpt)
    print("ANALYZE DONE")


if __name__ == "__main__":
    main()
