#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scripted expert demonstrations for the push-cube task -> LeRobot dataset.

Two stages (run in different conda envs):

  gen    (mujoco env):  simulate N expert push episodes with smooth, clipped
         mocap trajectories and render 224x224 top_cam frames; save raw data
         to data/raw_demos.npz.

  export (lerobot env): convert the raw npz into a LeRobot-format dataset
         (parquet + mp4 per episode) via LeRobotDataset.create /
         add_frame / save_episode.

  verify (lerobot env): load the dataset back and sanity-check fields.

Usage:
  python scripts/collect_demos.py gen    --episodes 5
  python scripts/collect_demos.py export --raw data/raw_demos.npz \
         --out data/push_cube_demos
  python scripts/collect_demos.py verify --root data/push_cube_demos
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ---- expert trajectory constants (tuned empirically on this scene) ----------
PUSH_Z = 0.86            # hand height while pushing (fingertips at cube top)
FINGER_Q = 0.025         # pre-tightened fingers (gap ~= cube width)
APPROACH_STEPS = 24      # home -> behind-cube approach, eased + clipped
PUSH_STEPS_MAX = 18      # closed-loop push budget
HOLD_STEPS = 3           # settle steps once the cube is on target
SUCCESS_DIST = 0.028     # stop pushing below this cube-target distance
STEP_MAX = 0.02          # per-step mocap increment (matches action space)
MOCAP_HIGH = np.array([0.85, 0.25, 1.15])
MOCAP_LOW = np.array([0.10, -0.25, 0.82])

TASK = 'push the cube onto the red target disc'


def ease_frac(n, ramp=2):
    """Cumulative displacement fractions: linear ramp with eased ends."""
    w = np.ones(n)
    w[:ramp] = np.linspace(0.4, 0.9, ramp)
    w[-ramp:] = np.linspace(0.9, 0.4, ramp)
    return np.cumsum(w / w.sum())


# ============================================================================
# stage: gen (mujoco env)
# ============================================================================
def stage_gen(episodes, select=None):
    """Simulate `select`-quality expert demos from `episodes` (or 2x) tries.

    Generates candidate episodes, then keeps the best `select` episodes
    (success first, then fewest over-budget steps, then smallest final
    distance). If select is None, keeps all.
    """
    import mujoco
    sys.path.insert(0, os.path.join(REPO, 'envs'))
    from push_cube_env import PushCubeEnv, HOME_POS, TARGET_XY

    n_try = episodes if select is None else min(episodes, 2 * select)
    keep = select if select is not None else n_try
    env = PushCubeEnv(image_obs=True, max_episode_steps=200)
    fps = int(round(1.0 / (env.model.opt.timestep * env.frame_skip)))
    print(f'[gen] physics dt={env.model.opt.timestep} frame_skip={env.frame_skip} '
          f'-> fps={fps}')

    rng = np.random.default_rng(42)
    episodes_data = []

    for ep in range(n_try):
        # small random initial cube offset (simulates demo variability);
        # dx biased away from the arm so the push distance stays within the
        # arm's reachable range
        off = np.array([rng.uniform(0.0, 0.02), rng.uniform(-0.02, 0.02)])
        obs, info = env.reset(seed=ep)
        cube_start = env.data.xpos[env._cube_body][:3].copy() + \
            np.array([off[0], off[1], 0.0])
        env.data.qpos[9:12] = cube_start
        env.data.qvel[9:15] = 0.0
        env.data.qpos[7:9] = FINGER_Q            # pre-tighten fingers
        mujoco.mj_forward(env.model, env.data)

        lane_y = float(cube_start[1])
        rear = np.array([cube_start[0] - 0.08, lane_y, PUSH_Z])
        fracs = ease_frac(APPROACH_STEPS)

        ep_imgs, ep_acts, ep_sts = [], [], []
        tgt = HOME_POS.copy()
        hold = 0
        for step in range(APPROACH_STEPS + PUSH_STEPS_MAX + HOLD_STEPS):
            obs_t = obs                          # frame observation (pre-action)
            if step < APPROACH_STEPS:
                goal = HOME_POS + (rear - HOME_POS) * fracs[step]
                a3 = np.clip(goal - tgt, -STEP_MAX, STEP_MAX)
            else:
                cube_xy = env.data.xpos[env._cube_body][:2]
                dist = float(np.linalg.norm(cube_xy - TARGET_XY))
                if dist < SUCCESS_DIST:
                    hold += 1
                    a3 = np.zeros(3)
                    if hold >= HOLD_STEPS:
                        break
                else:
                    dx = max(min(STEP_MAX, MOCAP_HIGH[0] -
                                 env.data.mocap_pos[0][0]), 0.0)
                    dy = float(np.clip(0.5 * (TARGET_XY[1] -
                                              env.data.mocap_pos[0][1]),
                                       -0.002, 0.002))
                    a3 = np.array([dx, dy, 0.0])
            action = np.concatenate([a3, [0.0]]).astype(np.float32)
            obs, r, term, trunc, info = env.step(action)

            ep_imgs.append(obs_t['image'].copy())
            ep_acts.append(action)
            ep_sts.append(obs_t['state'].copy())
            tgt = env.data.mocap_pos[0].copy()

        n = len(ep_acts)
        final = env.data.xpos[env._cube_body]
        dist = float(np.linalg.norm(final[:2] - TARGET_XY))
        success = dist < 0.03
        # ranking: success first, then stay within the step budget, then dist
        score = (0 if success else 1, max(0, n - 40), round(dist, 3))
        episodes_data.append({'imgs': ep_imgs, 'acts': ep_acts, 'sts': ep_sts,
                              'steps': n, 'dist': dist, 'success': success,
                              'score': score, 'start': cube_start})
        print(f'[gen] ep{ep}: steps={n} cube=({final[0]:.3f},{final[1]:.3f}) '
              f'final_dist={dist:.3f} success={success}')

    env.close()
    episodes_data.sort(key=lambda d: d['score'])
    selected = episodes_data[:keep]
    print(f'[gen] selected {len(selected)}/{n_try} episodes:')
    for d in selected:
        print(f'[gen]   steps={d["steps"]} dist={d["dist"]:.3f} '
              f'success={d["success"]}')

    out = os.path.join(REPO, 'data', 'raw_demos.npz')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        images=np.stack([f for d in selected for f in d['imgs']]).astype(np.uint8),
        actions=np.stack([f for d in selected for f in d['acts']]).astype(np.float32),
        states=np.stack([f for d in selected for f in d['sts']]).astype(np.float32),
        episode_lengths=np.array([d['steps'] for d in selected], dtype=np.int64),
        fps=fps,
        cube_starts=np.stack([d['start'] for d in selected]).astype(np.float32),
        task=TASK,
    )
    print(f'[gen] saved {sum(d["steps"] for d in selected)} frames / '
          f'{len(selected)} episodes -> {out}')
    return out


# ============================================================================
# stage: export (lerobot env)
# ============================================================================
def stage_export(raw_path, out_dir, repo_id='push_cube_expert'):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw = np.load(raw_path, allow_pickle=False)
    images = raw['images']
    actions = raw['actions']
    states = raw['states']
    lengths = raw['episode_lengths']
    fps = int(raw['fps'])
    task = str(raw['task'])

    H, W, C = images.shape[1:]
    features = {
        'observation.images.main': {
            'dtype': 'video', 'shape': (H, W, C),
            'names': ['height', 'width', 'channels']},
        'observation.state': {
            'dtype': 'float32', 'shape': states.shape[1:],
            'names': [f'joint_{i}' for i in range(7)] +
                     ['cube_x', 'cube_y', 'cube_z',
                      'target_x', 'target_y', 'target_z']},
        'action': {
            'dtype': 'float32', 'shape': actions.shape[1:],
            'names': ['dx', 'dy', 'dz', 'gripper']},
    }
    print(f'[export] {len(lengths)} episodes, {images.shape[0]} frames, fps={fps}')
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features,
                               root=out_dir, robot_type=None, use_videos=True)

    idx = 0
    for ep, length in enumerate(lengths):
        for _ in range(int(length)):
            ds.add_frame({
                'observation.images.main': images[idx],
                'observation.state': states[idx].astype(np.float32),
                'action': actions[idx].astype(np.float32),
                'task': task,
            })
            idx += 1
        ds.save_episode()
        print(f'[export] episode {ep} saved ({int(length)} frames)')
    ds.finalize()
    print(f'[export] dataset finalized at {out_dir}')
    return out_dir


# ============================================================================
# stage: verify (lerobot env)
# ============================================================================
def stage_verify(root, repo_id='push_cube_expert'):
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root, video_backend='pyav')
    try:
        n_ep, n_fr = ds.num_episodes, ds.num_frames
    except AttributeError:
        n_ep = ds.meta.info['total_episodes']
        n_fr = ds.meta.info['total_frames']
    print(f'[verify] num_episodes={n_ep} num_frames={n_fr} fps={ds.fps}')

    sample = ds[0]
    for k, v in sample.items():
        shape = tuple(v.shape) if hasattr(v, 'shape') else v
        print(f'[verify] sample[{k}]: {type(v).__name__} {shape}')

    img = sample['observation.images.main']
    assert tuple(img.shape[-3:]) == (3, 224, 224), 'image shape mismatch'
    assert tuple(sample['action'].shape) == (4,), 'action shape mismatch'
    assert tuple(sample['observation.state'].shape) == (13,), 'state shape'

    # parquet-level fields: timestamp / episode_index / frame_index
    import glob
    pqs = sorted(glob.glob(os.path.join(str(root), 'data', 'chunk-*',
                                        '*.parquet')))
    assert pqs, 'no parquet files found'
    df = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
    cols = ['timestamp', 'episode_index', 'frame_index', 'task_index']
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f'missing columns: {missing}'
    print(f'[verify] parquet cols ok; ts[0:3]={df["timestamp"].tolist()[:3]} '
          f'episodes={df["episode_index"].unique().tolist()} '
          f'frames_per_ep={df.groupby("episode_index").size().tolist()}')

    # decode one frame from every episode (evenly spaced indices)
    for i in np.linspace(0, int(n_fr) - 1, int(n_ep)).astype(int):
        _ = ds[int(i)]['observation.images.main']
    print('[verify] all episodes decodable')
    print('VERIFY OK')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['gen', 'export', 'verify'])
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--select', type=int, default=5)
    p.add_argument('--raw', default=os.path.join(REPO, 'data', 'raw_demos.npz'))
    p.add_argument('--out', default=os.path.join(REPO, 'data',
                                                 'push_cube_demos'))
    p.add_argument('--root', default=os.path.join(REPO, 'data',
                                                  'push_cube_demos'))
    args = p.parse_args()
    if args.stage == 'gen':
        stage_gen(args.episodes, select=args.select)
    elif args.stage == 'export':
        stage_export(args.raw, args.out)
    else:
        stage_verify(args.root)


if __name__ == '__main__':
    main()
