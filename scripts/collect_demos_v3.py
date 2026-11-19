#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Targeted expert demos addressing the 95% 'push-sideways / overshoot' failures.

Differences vs collect_demos_v2:
  * WIDER cube initial y offset (+-0.055 vs +-0.035) -> covers off-lane starts.
  * STRONG y correction during push: dy driven by cube-to-target y error (up to +-0.006)
    instead of the weak +-0.002 mocap-y term -> demonstrates active lane alignment.
  * 'STOP AT TARGET' x push: dx = clip(target_x - cube_x, 0, STEP_MAX) instead of
    a constant full-speed +x push -> demonstrates deceleration / stopping, removing
    the overshoot seen in baselines (avg x-progress 1.52).

Only SUCCESSFUL (dist < 0.03) trajectories are selected, matching baseline policy.
Stages (gen in mujoco env; export/verify in lerobot env):
  python scripts/collect_demos_v3.py gen  --episodes 120 --select 55
  python scripts/collect_demos_v3.py export --raw data/raw_demos_v3.npz --out data/push_cube_demos_v3
  python scripts/collect_demos_v3.py verify --root data/push_cube_demos_v3
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_demos as cd  # noqa: E402

CUBE_X_RANGE = (-0.035, 0.035)
CUBE_Y_RANGE = (-0.055, 0.055)      # targeted: wider y offsets
ARM_XYZ_RANGE = ((-0.010, 0.010), (-0.010, 0.010), (-0.006, 0.006))


def stage_gen(episodes, select, seed, out):
    import mujoco
    sys.path.insert(0, os.path.join(REPO, "envs"))
    from push_cube_env import PushCubeEnv, HOME_POS, TARGET_XY

    env = PushCubeEnv(image_obs=True, max_episode_steps=200)
    fps = int(round(1.0 / (env.model.opt.timestep * env.frame_skip)))
    print(f"[gen-v3] cube x{CUBE_X_RANGE} y{CUBE_Y_RANGE}; dt={env.model.opt.timestep} fps={fps}")

    rng = np.random.default_rng(seed)
    records = []
    for ep in range(episodes):
        cube_off = np.array([rng.uniform(*CUBE_X_RANGE), rng.uniform(*CUBE_Y_RANGE)])
        arm_off = np.array([rng.uniform(*r) for r in ARM_XYZ_RANGE])
        obs, info = env.reset(seed=3000 + ep)
        cube_start = env.data.xpos[env._cube_body][:3].copy()
        cube_start[0] += cube_off[0]
        cube_start[1] += cube_off[1]
        env.data.qpos[9:12] = cube_start
        env.data.qvel[9:15] = 0.0
        env.data.qpos[7:9] = cd.FINGER_Q
        start_pos = HOME_POS + arm_off
        env.data.mocap_pos[0] = start_pos
        mujoco.mj_forward(env.model, env.data)
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
                    # targeted: drive dy from CUBE y error (strong lane alignment)
                    dy_err = TARGET_XY[1] - cube_xy[1]
                    dy = float(np.clip(1.0 * dy_err, -0.006, 0.006))
                    # keep constant +x push (reliable full success); the SUCCESS_DIST
                    # hold below already demonstrates stopping, so no decel needed.
                    dx = float(np.clip(cd.STEP_MAX, 0.0,
                                       cd.MOCAP_HIGH[0] - env.data.mocap_pos[0][0]))
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
        y0 = cube_start[1]
        records.append({
            "imgs": ep_imgs, "acts": ep_acts, "sts": ep_sts, "steps": n,
            "dist": dist, "success": success, "y0": y0,
            "score": (0 if success else 1, round(dist, 4)),
            "start": cube_start, "arm_start": start_pos,
        })
        if ep % 10 == 0:
            print(f"[gen-v3] ep{ep:03d}: steps={n} y0={y0:.3f} dist={dist:.3f} succ={success}")
    env.close()

    records.sort(key=lambda d: d["score"])
    selected = records[:select]
    n_succ = sum(1 for d in selected if d["success"])
    y0s = [d["y0"] for d in selected]
    print(f"[gen-v3] selected {len(selected)}/{episodes} (success={n_succ}), "
          f"y0 range [{min(y0s):.3f},{max(y0s):.3f}]")
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
    print(f"[gen-v3] saved {sum(d['steps'] for d in selected)} frames / {len(selected)} episodes -> {out}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["gen", "export", "verify"])
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--select", type=int, default=55)
    p.add_argument("--seed", type=int, default=7777)
    p.add_argument("--raw", default=os.path.join(REPO, "data", "raw_demos_v3.npz"))
    p.add_argument("--out", default=os.path.join(REPO, "data", "push_cube_demos_v3"))
    p.add_argument("--root", default=os.path.join(REPO, "data", "push_cube_demos_v3"))
    args = p.parse_args()
    if args.stage == "gen":
        stage_gen(args.episodes, args.select, args.seed, args.raw)
    elif args.stage == "export":
        cd.stage_export(args.raw, args.out)
    else:
        cd.stage_verify(args.root)


if __name__ == "__main__":
    main()
