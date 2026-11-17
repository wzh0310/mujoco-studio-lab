#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Full-pipeline stress test: env -> interaction -> data generation.

Run inside the container (mujoco conda env, EGL renderer):

    python scripts/stress_test.py

Checks:
  A. 10 consecutive scripted expert episodes: no crash / NaN / black frame,
     penetration depths logged via the correct MuJoCo 4.0 contact API.
  B. raw demo temporal alignment (frame counts, fps, bounded frame-to-frame
     cube motion, changing images) via utils.validation.
  C. reset stability across 20 resets (arm + cube return to identical pose).

Exit code 0 iff all hard invariants pass. Prints a JSON report at the end.
"""
import os
import sys
import json
import numpy as np
import mujoco

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "envs"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)

from push_cube_env import PushCubeEnv, HOME_POS, TARGET_XY  # noqa: E402
import collect_demos as cd  # noqa: E402
from utils import validation as uv  # noqa: E402
from utils import vision as uvis  # noqa: E402


def run_episodes(env, n=10):
    """Replay the expert trajectory n times; collect health metrics."""
    m, d = env.model, env.data
    rng = np.random.default_rng(2026)
    eps, crashes = [], []
    total_frames = nan_steps = black_frames = 0
    worst_pen = {"cube_table": 0.0, "arm_table": 0.0, "cube_finger": 0.0}

    for ep in range(n):
        off = np.array([rng.uniform(0.0, 0.02), rng.uniform(-0.02, 0.02)])
        try:
            obs, _ = env.reset(seed=100 + ep)
            cube = d.xpos[env._cube_body][:3].copy() + np.array([off[0], off[1], 0])
            d.qpos[9:12] = cube
            d.qvel[9:15] = 0.0
            d.qpos[7:9] = cd.FINGER_Q
            mujoco.mj_forward(m, d)
            obs = env._get_obs()  # FIX: refresh after offset injection
            env._prev_dist = env._cube_target_dist()

            rear = np.array([cube[0] - 0.08, cube[1], cd.PUSH_Z])
            fracs = cd.ease_frac(cd.APPROACH_STEPS)
            tgt, hold, steps = HOME_POS.copy(), 0, 0
            ep_nan = ep_black = 0
            ep_pen = dict(worst_pen)

            for step in range(cd.APPROACH_STEPS + cd.PUSH_STEPS_MAX + cd.HOLD_STEPS):
                if step < cd.APPROACH_STEPS:
                    goal = HOME_POS + (rear - HOME_POS) * fracs[step]
                    a3 = np.clip(goal - tgt, -cd.STEP_MAX, cd.STEP_MAX)
                else:
                    dist = np.linalg.norm(d.xpos[env._cube_body][:2] - TARGET_XY)
                    if dist < cd.SUCCESS_DIST:
                        hold += 1
                        a3 = np.zeros(3)
                        if hold >= cd.HOLD_STEPS:
                            break
                    else:
                        dx = max(min(cd.STEP_MAX, cd.MOCAP_HIGH[0] -
                                     d.mocap_pos[0][0]), 0.0)
                        dy = float(np.clip(0.5 * (TARGET_XY[1] -
                                                  d.mocap_pos[0][1]), -0.002, 0.002))
                        a3 = np.array([dx, dy, 0.0])
                obs, _, _, _, _ = env.step(np.concatenate([a3, [0.0]]).astype(np.float32))
                steps += 1
                if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
                    ep_nan += 1
                if uvis.is_black_image(obs["image"]):
                    ep_black += 1
                pen = uv.contact_penetrations(m, d)
                for k in ep_pen:
                    ep_pen[k] = max(ep_pen[k], pen[k])
                    worst_pen[k] = max(worst_pen[k], pen[k])
                tgt = d.mocap_pos[0].copy()

            final_dist = float(np.linalg.norm(d.xpos[env._cube_body][:2] - TARGET_XY))
            eps.append({"ep": ep, "steps": steps, "final_dist": round(final_dist, 4),
                        "success": bool(final_dist < 0.03), "nan": ep_nan,
                        "black": ep_black,
                        "pen": {k: round(v, 4) for k, v in ep_pen.items()}})
            total_frames += steps
            nan_steps += ep_nan
            black_frames += ep_black
        except Exception as exc:  # noqa: BLE001
            crashes.append({"ep": ep, "error": repr(exc)})

    return {"episodes": eps, "crashes": crashes, "total_frames": total_frames,
            "nan_steps": nan_steps, "black_frames": black_frames,
            "worst_penetration": {k: round(v, 4) for k, v in worst_pen.items()}}


def main():
    report = {}
    env = PushCubeEnv(image_obs=True, max_episode_steps=200)
    report["A_rollouts"] = run_episodes(env, n=10)
    dt = env.model.opt.timestep * env.frame_skip
    env.close()

    raw_path = os.path.join(REPO, "data", "raw_demos.npz")
    if not os.path.exists(raw_path):
        # raw_demos.npz is gitignored, so a fresh clone won't have it. Generate
        # a small expert batch here so check B always runs and the stress test
        # is self-contained / reproducible straight from a clean checkout.
        # (The full 5-episode LeRobot dataset is committed separately.)
        print("[stress] raw_demos.npz not found; generating a small expert "
              "batch for the alignment check...")
        cd.stage_gen(episodes=2, select=1)
    if os.path.exists(raw_path):
        raw = np.load(raw_path, allow_pickle=False)
        report["B_alignment"] = uv.check_episode_alignment(
            raw["images"], raw["actions"], raw["states"],
            raw["episode_lengths"], int(raw["fps"]), dt)

    env2 = PushCubeEnv(image_obs=False, max_episode_steps=50)
    samples = []
    for s in range(20):
        env2.reset(seed=s)
        samples.append(np.concatenate([
            env2.data.xpos[env2._cube_body],
            env2.data.xpos[env2._hand_body],
            env2.data.qpos.copy(),
        ]))
    env2.close()
    samples = np.array(samples)
    report["C_reset"] = {"n_resets": len(samples),
                         "max_dev": float(np.abs(samples - samples.mean(0)).max())}

    a = report["A_rollouts"]
    hard_ok = (not a["crashes"] and a["nan_steps"] == 0 and
               a["black_frames"] == 0 and
               report.get("B_alignment", {}).get("lengths_match", False) and
               report["C_reset"]["max_dev"] < 1e-5)
    report["PASS"] = bool(hard_ok)

    print("===== STRESS REPORT =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("PASS" if hard_ok else "FAIL")
    sys.exit(0 if hard_ok else 1)


if __name__ == "__main__":
    main()
