# -*- coding: utf-8 -*-
"""Automated ACT evaluation for push_cube (Day-4).

Protocol (task spec):
  * 3 fixed seeds x 10 episodes = 30 rollouts.
  * Metrics:
      1) success_rate          - fraction of episodes where cube reaches target
      2) avg_completion_steps  - mean executed steps over SUCCESSFUL episodes
      3) avg_distance_error    - mean final cube-to-target distance (metres)
  * Everything runs OFFSCREEN (MUJOCO_GL=egl); no viewer window.
  * Optionally records videos of a few successful episodes (dual view).

The policy is loaded exactly like LeRobot eval: config from the checkpoint,
weights via make_policy(pretrained_path=...), and its own pre/post processors
so preprocessing is identical to training (no manual re-normalization).

Run (lerobot env):
  python scripts/eval_act.py --checkpoint /path/to/checkpoints/004000 \
      --seeds 100 200 300 --episodes-per-seed 10 \
      --out data/eval_act_metrics.json
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

import lerobot_push_cube_env as lc  # noqa: E402
from lerobot.policies.factory import (  # noqa: E402
    make_policy, make_pre_post_processors,
)
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from push_cube_env import PushCubeEnv, TARGET_XY  # noqa: E402


def load_policy_from_checkpoint(ckpt_dir):
    """Load ACT policy + its pre/post processors from a saved checkpoint."""
    pretrained = os.path.join(ckpt_dir, "pretrained_model")
    if not os.path.isdir(pretrained):
        pretrained = ckpt_dir
    from lerobot.envs import make_env_config
    cfg = ACTConfig.from_pretrained(pretrained)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    # make_policy needs either ds_meta or env_cfg; the checkpoint config already
    # carries input/output features, but the factory still requires one of them.
    env_cfg = make_env_config("push_cube")
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, pretrained_path=pretrained)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=pretrained,
    )
    return policy, preprocessor, postprocessor, cfg


def run_episode(env, policy, pre, post, device, seed, max_steps, render_frames=False):
    """One rollout with the raw PushCubeEnv; returns metrics + optional frames."""
    obs, info = env.reset(seed=seed)
    # Randomize the cube's initial position per episode so the 30 rollouts are
    # genuinely independent and match the training-data distribution. The base
    # env.reset() always uses the fixed home keyframe (cube at [0.62,-0.12]);
    # collect_demos_v2 generated training data with cube offsets in these ranges.
    rng = np.random.default_rng(seed)
    dx = float(rng.uniform(-0.030, 0.030))
    dy = float(rng.uniform(-0.035, 0.035))
    env.data.qpos[9] += dx
    env.data.qpos[10] += dy
    env.data.qvel[9:15] = 0.0
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    env._prev_dist = env._cube_target_dist()
    frames = []
    for t in range(max_steps):
        # Build a batched canonical observation dict for the processor/policy.
        # NOTE: LeRobot's dataset yields images as float32 in [0,1] (uint8/255).
        # The policy processor's normalizer derives stats dtype from the input
        # tensor, so we must pass float32 (not uint8) or stats conversion
        # overflows ("value cannot be converted to type uint8").
        # LeRobot policy processors expect images in CHW layout (B,C,H,W):
        # the normalizer stats for images have shape (C,), applied on dim 1.
        # The env yields uint8 HWC, so transpose to CHW and scale to [0,1].
        img_chw = np.transpose(obs["image"], (2, 0, 1)).astype(np.float32) / 255.0
        canon = {
            "observation.images.main": torch.from_numpy(
                np.ascontiguousarray(img_chw)[None]).to(device),
            "observation.state": torch.from_numpy(
                np.ascontiguousarray(obs["state"]).astype(np.float32)[None]).to(device),
        }
        # policy processor -> policy -> postprocessor -> normalized action.
        # select_action / postprocessor return a tensor (B, action_dim) directly
        # in this LeRobot build (not a dict).
        with torch.inference_mode():
            processed = pre(canon)
            action = policy.select_action(processed)
            action = post(action)
        if isinstance(action, dict):
            action = action["action"]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        a = np.asarray(action).reshape(-1)[:4]
        # IMPORTANT: postprocessor (unnormalizer) already returns RAW actions in
        # the simulator's action_space (verified: dx~0.0177). Do NOT apply
        # a*std+mean again -- that would collapse dy/dz toward the dataset mean
        # and make the arm push straight, missing the target.
        raw = np.clip(a, env.action_space.low, env.action_space.high)
        obs, reward, term, trunc, info = env.step(raw)
        if render_frames:
            frames.append(np.ascontiguousarray(env.render()))
        if term or trunc:
            break
    cube = env.data.xpos[env._cube_body]
    dist = float(np.linalg.norm(cube[:2] - TARGET_XY))
    success = bool(dist < 0.03)
    steps = int(info.get("steps", t + 1))
    return {"success": success, "steps": steps, "dist": dist,
            "total_reward": float(reward)}, frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300])
    ap.add_argument("--episodes-per-seed", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--out", default="data/eval_act_metrics.json")
    ap.add_argument("--record-success", type=int, default=0,
                    help="record up to N successful episodes as MP4")
    ap.add_argument("--video-dir", default="assets")
    args = ap.parse_args()

    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    video_dir = args.video_dir if os.path.isabs(args.video_dir) else os.path.join(REPO, args.video_dir)

    print("=" * 64)
    print("[eval_act] loading policy from", args.checkpoint)
    policy, pre, post, cfg = load_policy_from_checkpoint(args.checkpoint)
    device = next(policy.parameters()).device
    print("[eval_act] device", device, "| seeds", args.seeds,
          "| ep/seed", args.episodes_per_seed)

    env = PushCubeEnv(image_obs=True, max_episode_steps=args.max_steps)
    results = []
    saved_videos = 0
    for seed in args.seeds:
        for ep in range(args.episodes_per_seed):
            rec = (saved_videos < args.record_success)
            metrics, frames = run_episode(
                env, policy, pre, post, device,
                seed=seed * 1000 + ep, max_steps=args.max_steps,
                render_frames=rec)
            metrics["seed"] = seed
            metrics["episode"] = ep
            results.append(metrics)
            if rec and metrics["success"] and frames:
                from lerobot.utils.io_utils import write_video
                os.makedirs(video_dir, exist_ok=True)
                vp = os.path.join(video_dir, "eval_success_%d_%d.mp4" % (seed, ep))
                write_video(vp, np.stack(frames), fps=50)
                print("[eval_act] wrote video", vp)
                saved_videos += 1
            print("[eval_act] seed=%d ep=%d success=%s steps=%d dist=%.4f" % (
                seed, ep, metrics["success"], metrics["steps"], metrics["dist"]))
    env.close()

    successes = [r for r in results if r["success"]]
    n_total = len(results)
    summary = {
        "checkpoint": args.checkpoint,
        "n_rollouts": n_total,
        "success_rate": len(successes) / n_total if n_total else 0.0,
        "avg_completion_steps": float(np.mean([r["steps"] for r in successes])) if successes else None,
        "avg_distance_error": float(np.mean([r["dist"] for r in results])),
        "avg_distance_error_success": float(np.mean([r["dist"] for r in successes])) if successes else None,
        "avg_total_reward": float(np.mean([r["total_reward"] for r in results])),
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes_per_seed,
        "per_episode": results,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("=" * 64)
    print("[eval_act] SUCCESS_RATE = %.3f (%d/%d)" % (
        summary["success_rate"], len(successes), n_total))
    print("[eval_act] avg_completion_steps =", summary["avg_completion_steps"])
    print("[eval_act] avg_distance_error = %.4f m" % summary["avg_distance_error"])
    print("[eval_act] metrics ->", out_path)
    print("EVAL_ACT DONE")


if __name__ == "__main__":
    main()
