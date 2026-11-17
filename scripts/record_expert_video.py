#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Record a 10-second expert push demo (offscreen) + 3 key screenshots.

Run inside the container, in the `mujoco` conda env:
    source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mujoco
    export MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless
    python scripts/record_expert_video.py

Approach: run the SAME scripted expert used to generate the dataset (mocap +
weld, no hand-written IK), capture each simulated keyframe as a full state,
then temporally up-sample those true states for a smooth 10s clip. Rendering is
fully offscreen via mujoco.Renderer -- no WebViewer / browser needed.

Outputs (all under assets/):
    expert_push_demo.mp4     10s, 640x480, 24fps, global_cam (whole arm)
    expert_push_demo.gif     lightweight preview for README
    scene_overview.png       global_cam wide shot at the start
    camera_observation.png   top_cam 224x224 observation, mid-push
    goal_reached.png         top_cam close shot, cube on the red target
"""
import os
import subprocess
import sys

import numpy as np
import mujoco

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "envs"))
from push_cube_env import PushCubeEnv, HOME_POS, TARGET_XY  # noqa: E402
from utils.vision import save_png                            # noqa: E402

# ---- expert constants (identical to collect_demos.py, proven to succeed) ----
PUSH_Z = 0.86
FINGER_Q = 0.025
APPROACH_STEPS = 24
PUSH_STEPS_MAX = 18
HOLD_STEPS = 3
SUCCESS_DIST = 0.028
STEP_MAX = 0.02
MOCAP_HIGH = np.array([0.85, 0.25, 1.15])

FPS = 24
DURATION_S = 10
N_FRAMES = FPS * DURATION_S            # 240 frames -> exactly 10s
VIDEO_W, VIDEO_H = 640, 480

ASSETS = os.path.join(REPO, "assets")
TMP_FRAMES = os.path.join("/tmp", "expert_push_frames")


def ease_frac(n, ramp=2):
    w = np.ones(n)
    w[:ramp] = np.linspace(0.4, 0.9, ramp)
    w[-ramp:] = np.linspace(0.9, 0.4, ramp)
    return np.cumsum(w / w.sum())


def run_expert(seed):
    """Run one scripted expert episode; return (states, final_dist, success).

    Each state = (qpos.copy(), mocap_pos.copy(), mocap_quat.copy()).
    """
    env = PushCubeEnv(image_obs=False, frame_skip=10, max_episode_steps=200)
    m, d = env.model, env.data
    cube = env._cube_body

    obs, info = env.reset(seed=seed)
    # Deterministic clean demo: use the scene's default cube start, fingers
    # pre-tightened around the cube (same as dataset generation).
    cube_start = d.xpos[cube][:3].copy()
    d.qpos[7:9] = FINGER_Q
    mujoco.mj_forward(m, d)

    lane_y = float(cube_start[1])
    rear = np.array([cube_start[0] - 0.08, lane_y, PUSH_Z])
    fracs = ease_frac(APPROACH_STEPS)

    states = [(d.qpos.copy(), d.mocap_pos[0].copy(), d.mocap_quat[0].copy())]
    tgt = HOME_POS.copy()
    hold = 0
    for step in range(APPROACH_STEPS + PUSH_STEPS_MAX + HOLD_STEPS):
        if step < APPROACH_STEPS:
            goal = HOME_POS + (rear - HOME_POS) * fracs[step]
            a3 = np.clip(goal - tgt, -STEP_MAX, STEP_MAX)
        else:
            cube_xy = d.xpos[cube][:2]
            dist = float(np.linalg.norm(cube_xy - TARGET_XY))
            if dist < SUCCESS_DIST:
                hold += 1
                a3 = np.zeros(3)
                if hold >= HOLD_STEPS:
                    break
            else:
                dx = max(min(STEP_MAX, MOCAP_HIGH[0] - d.mocap_pos[0][0]), 0.0)
                dy = float(np.clip(0.5 * (TARGET_XY[1] - d.mocap_pos[0][1]),
                                   -0.002, 0.002))
                a3 = np.array([dx, dy, 0.0])
        action = np.concatenate([a3, [0.0]]).astype(np.float32)
        env.step(action)
        states.append((d.qpos.copy(), d.mocap_pos[0].copy(),
                       d.mocap_quat[0].copy()))
        tgt = d.mocap_pos[0].copy()

    final = d.xpos[cube]
    dist = float(np.linalg.norm(final[:2] - TARGET_XY))
    nq = env.model.nq
    env.close()
    return states, dist, dist < 0.03


def interpolate_state(states, f):
    """Linearly interpolate a full state at continuous index f."""
    n = len(states)
    i0 = int(np.floor(f))
    i1 = min(i0 + 1, n - 1)
    w = float(f - i0)
    q = (1 - w) * states[i0][0] + w * states[i1][0]
    mp = (1 - w) * states[i0][1] + w * states[i1][1]
    mq = (1 - w) * states[i0][2] + w * states[i1][2]
    mq /= np.linalg.norm(mq)
    return q, mp, mq


def set_state(m, d, q, mp, mq):
    d.qpos[:] = q
    d.qvel[:] = 0.0
    d.mocap_pos[0] = mp
    d.mocap_quat[0] = mq
    mujoco.mj_forward(m, d)


def render_frame(renderer, m, d, q, mp, mq, camera):
    set_state(m, d, q, mp, mq)
    renderer.update_scene(d, camera=camera)
    return renderer.render()


def main():
    os.makedirs(ASSETS, exist_ok=True)
    os.makedirs(TMP_FRAMES, exist_ok=True)

    # 1. obtain a successful expert trajectory (try a few seeds)
    best = None       # strictly successful
    closest = None    # fallback: smallest final distance seen
    for seed in range(6):
        states, dist, ok = run_expert(seed)
        print(f"[expert] seed={seed} states={len(states)} "
              f"final_dist={dist:.3f} success={ok}")
        if closest is None or dist < closest[1]:
            closest = (states, dist, seed)
        if ok and (best is None or dist < best[1]):
            best = (states, dist, seed)
    if best is None:
        print("[expert] WARNING: no strict success, using closest trajectory")
        best = closest
    states, dist, seed = best
    n_states = len(states)
    print(f"[expert] using seed={seed}: {n_states} states, final_dist={dist:.3f}")

    env = PushCubeEnv(image_obs=False, frame_skip=10, max_episode_steps=200)
    m, d = env.model, env.data

    # 2. sample indices across the trajectory (hold a little at both ends)
    hold_start = int(N_FRAMES * 0.08)
    hold_end = int(N_FRAMES * 0.12)
    idx = []
    for f in range(N_FRAMES):
        if f < hold_start:
            s = 0.0
        elif f >= N_FRAMES - hold_end:
            s = float(n_states - 1)
        else:
            t = (f - hold_start) / (N_FRAMES - hold_start - hold_end)
            s = t * (n_states - 1)
        idx.append(s)

    # 3. render all frames with global_cam (whole arm visible) -> mp4 via pipe
    r_global = mujoco.Renderer(m, height=VIDEO_H, width=VIDEO_W)
    ffmpeg = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{VIDEO_W}x{VIDEO_H}", "-r", str(FPS),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        os.path.join(ASSETS, "expert_push_demo.mp4"),
    ]
    proc = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    shot_overview_saved = False
    for f, s in enumerate(idx):
        q, mp, mq = interpolate_state(states, s)
        frame = render_frame(r_global, m, d, q, mp, mq, "global_cam")
        proc.stdin.write(frame.tobytes())
        if not shot_overview_saved and f == hold_start:
            # wide start scene (offscreen framebuffer is capped at 640x480 by
            # the model's <visual><global offwidth/offheight>, so stay <= that)
            r_wide = mujoco.Renderer(m, height=480, width=640)
            wide = render_frame(r_wide, m, d, q, mp, mq, "global_cam")
            save_png(os.path.join(ASSETS, "scene_overview.png"), wide)
            r_wide.close()
            shot_overview_saved = True
        if f % 48 == 0:
            print(f"[render] global frame {f+1}/{N_FRAMES}")
    proc.stdin.close()
    ferr = proc.stderr.read().decode("utf-8", "replace")
    rc = proc.wait()
    r_global.close()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed rc={rc}: {ferr[-1000:]}")
    print("[render] mp4 done")

    # 4. key screenshots from top_cam
    r_top = mujoco.Renderer(m, height=224, width=224)
    # observation mid-push (~55% along the trajectory)
    f_mid = int(N_FRAMES * 0.55)
    q, mp, mq = interpolate_state(states, idx[f_mid])
    obs = render_frame(r_top, m, d, q, mp, mq, "top_cam")
    save_png(os.path.join(ASSETS, "camera_observation.png"), obs)
    # goal reached (final state), close view (within 640x480 framebuffer cap)
    r_top_big = mujoco.Renderer(m, height=480, width=480)
    q, mp, mq = interpolate_state(states, float(n_states - 1))
    goal = render_frame(r_top_big, m, d, q, mp, mq, "top_cam")
    save_png(os.path.join(ASSETS, "goal_reached.png"), goal)
    r_top.close()
    r_top_big.close()
    env.close()
    print("[render] screenshots done")

    # 5. lightweight GIF preview for README (downscaled, lower fps)
    gif = os.path.join(ASSETS, "expert_push_demo.gif")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", os.path.join(ASSETS, "expert_push_demo.mp4"),
        "-vf", "fps=10,scale=480:-1:flags=lanczos,split[s0][s1];"
               "[s0]palettegen[p];[s1][p]paletteuse",
        gif,
    ], check=True)
    print("[render] gif done ->", gif)

    print("RECORD EXPERT VIDEO DONE")


if __name__ == "__main__":
    main()
