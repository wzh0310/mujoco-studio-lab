# -*- coding: utf-8 -*-
"""Record a dual-view (third-person global + first-person task camera) MP4 of a
successful ACT rollout on push_cube (Day-4 deliverable).

Left panel  = global_cam  (third-person overview of the whole scene)
Right panel = top_cam     (first-person task observation fed to the policy)

Run (lerobot env):
  python scripts/record_dual_view.py --checkpoint /path/to/checkpoints/004000
"""
import argparse
import os
import sys

import mujoco
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "envs"))

from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa
from lerobot.policies.act.configuration_act import ACTConfig  # noqa
from lerobot.envs import make_env_config  # noqa
import lerobot_push_cube_env  # noqa: F401  (registers push_cube EnvConfig)
from push_cube_env import PushCubeEnv, TARGET_XY  # noqa


def load_policy(ckpt_dir):
    pretrained = os.path.join(ckpt_dir, "pretrained_model")
    if not os.path.isdir(pretrained):
        pretrained = ckpt_dir
    cfg = ACTConfig.from_pretrained(pretrained)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    pol = make_policy(cfg=cfg, env_cfg=make_env_config("push_cube"), pretrained_path=pretrained)
    pol.eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=pretrained)
    return pol, pre, post


def rollout_dual(env, pol, pre, post, device, seed, max_steps):
    obs, info = env.reset(seed=seed)
    # match eval: randomize cube start (seed 100, ep 3 was a success)
    rng = np.random.default_rng(seed)
    env.data.qpos[9] += float(rng.uniform(-0.030, 0.030))
    env.data.qpos[10] += float(rng.uniform(-0.035, 0.035))
    env.data.qvel[9:15] = 0.0
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    env._prev_dist = env._cube_target_dist()

    r_global = mujoco.Renderer(env.model, height=224, width=224)
    r_task = mujoco.Renderer(env.model, height=224, width=224)
    frames = []
    for t in range(max_steps):
        img_chw = np.transpose(obs["image"], (2, 0, 1)).astype(np.float32) / 255.0
        canon = {
            "observation.images.main": torch.from_numpy(np.ascontiguousarray(img_chw)[None]).to(device),
            "observation.state": torch.from_numpy(
                np.ascontiguousarray(obs["state"]).astype(np.float32)[None]).to(device),
        }
        with torch.inference_mode():
            a = pol.select_action(pre(canon))
            a = post(a)
        if isinstance(a, dict):
            a = a["action"]
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        raw = np.clip(np.asarray(a).reshape(-1)[:4], env.action_space.low, env.action_space.high)

        # render BOTH views at the current frame (before stepping)
        r_global.update_scene(env.data, camera="global_cam")
        r_task.update_scene(env.data, camera="top_cam")
        g = r_global.render()
        tc = r_task.render()
        frames.append(np.concatenate([g, tc], axis=1))  # (224, 448, 3)

        obs, reward, term, trunc, info = env.step(raw)
        if term or trunc:
            break
    r_global.close()
    r_task.close()
    cube = env.data.xpos[env._cube_body]
    dist = float(np.linalg.norm(cube[:2] - TARGET_XY))
    return frames, dist, dist < 0.03, int(info.get("steps", t + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300])
    ap.add_argument("--out", default="assets/eval_dual_view.mp4")
    ap.add_argument("--max-steps", type=int, default=50)
    args = ap.parse_args()

    pol, pre, post = load_policy(args.checkpoint)
    device = next(pol.parameters()).device
    env = PushCubeEnv(image_obs=True, max_episode_steps=args.max_steps)
    best = None
    for s in args.seeds:
        frames, dist, success, steps = rollout_dual(env, pol, pre, post, device, s, args.max_steps)
        print(f"[dual] seed={s} success={success} steps={steps} dist={dist:.4f}")
        if success and (best is None or dist < best[1]):
            best = (frames, dist, s)
    env.close()

    if best is None:
        print("[dual] no successful rollout; nothing recorded")
        return
    frames, dist, seed = best
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    from lerobot.utils.io_utils import write_video
    write_video(out, np.stack(frames), fps=50)
    print(f"[dual] wrote {out} frames={len(frames)} shape={np.stack(frames).shape} seed={seed} dist={dist:.4f}")
    print("DUAL_VIEW DONE")


if __name__ == "__main__":
    main()
