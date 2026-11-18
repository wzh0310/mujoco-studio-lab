#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Enhanced expert demo generation for push_cube (Day-3 full dataset).

Compared with collect_demos.py, this generator:
  * randomizes BOTH the cube initial position and the arm (mocap) start pose
    with small offsets, so the 40 episodes are genuinely diverse;
  * plans the approach trajectory from the *actual* (offset) arm start pose,
    keeping observation/action alignment intact;
  * keeps the best `select` episodes (success first, then smallest distance).

Stages (run in different conda envs, same as collect_demos.py):
  gen    (mujoco env):  simulate expert pushes -> data/raw_demos_v2.npz
  export (lerobot env): npz -> LeRobot v3 dataset at data/push_cube_demos
  verify (lerobot env): load back & sanity-check

Usage:
  python scripts/collect_demos_v2.py gen  --episodes 48 --select 40
  python scripts/collect_demos_v2.py export --raw data/raw_demos_v2.npz --out data/push_cube_demos
  python scripts/collect_demos_v2.py verify --root data/push_cube_demos
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_demos as cd  # noqa: E402  (reuse export/verify + constants)

# ---- diversity ranges (metres) ---------------------------------------------
CUBE_X_RANGE = (-0.030, 0.030)   # cube initial x offset
CUBE_Y_RANGE = (-0.035, 0.035)   # cube initial y offset
ARM_XYZ_RANGE = ((-0.008, 0.008), (-0.008, 0.008), (-0.006, 0.006))  # mocap start


def stage_gen(episodes, select, seed, out):
    import mujoco
    sys.path.insert(0, os.path.join(REPO, "envs"))
    from push_cube_env import PushCubeEnv, HOME_POS, TARGET_XY

    env = PushCubeEnv(image_obs=True, max_episode_steps=200)
    fps = int(round(1.0 / (env.model.opt.timestep * env.frame_skip)))
    print(f"[gen-v2] dt={env.model.opt.timestep} frame_skip={env.frame_skip} fps={fps}")
    print(f"[gen-v2] cube offset x{CUBE_X_RANGE} y{CUBE_Y_RANGE}; arm offset {ARM_XYZ_RANGE}")

    rng = np.random.default_rng(seed)
    records = []
    for ep in range(episodes):
        cube_off = np.array([rng.uniform(*CUBE_X_RANGE), rng.uniform(*CUBE_Y_RANGE)])
        arm_off = np.array([rng.uniform(*r) for r in ARM_XYZ_RANGE])

        obs, info = env.reset(seed=1000 + ep)
        # cube initial position (free joint qpos[9:12])
        cube_start = env.data.xpos[env._cube_body][:3].copy()
        cube_start[0] += cube_off[0]
        cube_start[1] += cube_off[1]
        env.data.qpos[9:12] = cube_start
        env.data.qvel[9:15] = 0.0
        env.data.qpos[7:9] = cd.FINGER_Q
        # arm (mocap) start pose with small offset
        start_pos = HOME_POS + arm_off
        env.data.mocap_pos[0] = start_pos
        mujoco.mj_forward(env.model, env.data)
        # refresh obs AFTER injecting all offsets (alignment fix)
        obs = env._get_obs()
        env._prev_dist = env._cube_target_dist()

        lane_y = float(cube_start[1])
        rear = np.array([cube_start[0] - 0.08, lane_y, cd.PUSH_Z])
        fracs = cd.ease_frac(cd.APPROACH_STEPS)

        ep_imgs, ep_acts, ep_sts = [], [], []
        tgt = start_pos.copy()
        hold = 0
        for step in range(cd.APPROACH_STEPS + cd.PUSH_STEPS_MAX + cd.HOLD_STEPS):
            obs_t = obs
            if step < cd.APPROACH_STEPS:
                goal = start_pos + (rear - start_pos) * fracs[step]
                a3 = np.clip(goal - tgt, -cd.STEP_MAX, cd.STEP_MAX)
            else:
                cube_xy = env.data.xpos[env._cube_body][:2]
                dist = float(np.linalg.norm(cube_xy - TARGET_XY))
                if dist < cd.SUCCESS_DIST:
                    hold += 1
                    a3 = np.zeros(3)
                    if hold >= cd.HOLD_STEPS:
                        break
                else:
                    dx = max(min(cd.STEP_MAX, cd.MOCAP_HIGH[0] - env.data.mocap_pos[0][0]), 0.0)
                    dy = float(np.clip(0.5 * (TARGET_XY[1] - env.data.mocap_pos[0][1]), -0.002, 0.002))
                    a3 = np.array([dx, dy, 0.0])
            action = np.concatenate([a3, [0.0]]).astype(np.float32)
            obs, r, term, trunc, info = env.step(action)
            ep_imgs.append(obs_t["image"].copy())
            ep_acts.append(action)
            ep_sts.append(obs_t["state"].copy())
            tgt = env.data.mocap_pos[0].copy()

        n = len(ep_acts)
        final = env.data.xpos[env._cube_body]
        dist = float(np.linalg.norm(final[:2] - TARGET_XY))
        success = dist < 0.03
        records.append({
            "imgs": ep_imgs, "acts": ep_acts, "sts": ep_sts, "steps": n,
            "dist": dist, "success": success,
            "score": (0 if success else 1, round(dist, 4)),
            "start": cube_start, "arm_start": start_pos,
        })
        print(f"[gen-v2] ep{ep:02d}: steps={n} cube=({final[0]:.3f},{final[1]:.3f}) "
              f"start=({cube_start[0]:.3f},{cube_start[1]:.3f}) dist={dist:.3f} success={success}")
    env.close()

    records.sort(key=lambda d: d["score"])
    selected = records[:select]
    n_succ = sum(1 for d in selected if d["success"])
    print(f"[gen-v2] selected {len(selected)}/{episodes} (success={n_succ})")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        images=np.stack([f for d in selected for f in d["imgs"]]).astype(np.uint8),
        actions=np.stack([f for d in selected for f in d["acts"]]).astype(np.float32),
        states=np.stack([f for d in selected for f in d["sts"]]).astype(np.float32),
        episode_lengths=np.array([d["steps"] for d in selected], dtype=np.int64),
        fps=fps,
        cube_starts=np.stack([d["start"] for d in selected]).astype(np.float32),
        arm_starts=np.stack([d["arm_start"] for d in selected]).astype(np.float32),
        task=cd.TASK,
    )
    print(f"[gen-v2] saved {sum(d['steps'] for d in selected)} frames / {len(selected)} episodes -> {out}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["gen", "export", "verify"])
    p.add_argument("--episodes", type=int, default=48)
    p.add_argument("--select", type=int, default=40)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--raw", default=os.path.join(REPO, "data", "raw_demos_v2.npz"))
    p.add_argument("--out", default=os.path.join(REPO, "data", "push_cube_demos"))
    p.add_argument("--root", default=os.path.join(REPO, "data", "push_cube_demos"))
    args = p.parse_args()
    if args.stage == "gen":
        stage_gen(args.episodes, args.select, args.seed, args.raw)
    elif args.stage == "export":
        cd.stage_export(args.raw, args.out)
    else:
        cd.stage_verify(args.root)


if __name__ == "__main__":
    main()
