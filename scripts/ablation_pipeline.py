# -*- coding: utf-8 -*-
"""Ablation pipeline: generate config variants, train, evaluate, summarize.

Runs INSIDE the container (lerobot env). Each experiment:
  1) load base yaml (configs/act_full_train.yaml), override selected fields
  2) write configs/ablation_<name>.yaml with a unique output_dir
  3) train via scripts/train_act_push_cube.py (subprocess, isolated)
  4) evaluate last checkpoint via scripts/eval_act.py (headless, fixed protocol)
  5) collect success_rate / dist / steps into a results json

Usage:
  python scripts/ablation_pipeline.py --exp chunk12 chunk16 --steps 1600 --eval-eps 5
  python scripts/ablation_pipeline.py --list
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import yaml

REPO = "/data/wzh/robot/mujoco-studio-lab"
BASE_CFG = os.path.join(REPO, "configs", "act_full_train.yaml")
TRAINER = os.path.join(REPO, "scripts", "train_act_push_cube.py")
EVALUATOR = os.path.join(REPO, "scripts", "eval_act.py")
RESULTS = os.path.join(REPO, "data", "ablation_results.json")
OUTPUTS = "/data/wzh/robot/outputs"

# ---- experiment definitions: name -> dict of overrides -----------------------
# structure mirrors the nested yaml; we do a deep-merge onto the base config.
EXPERIMENTS = {
    # ---- quick ablations (low cost) ----
    "base_s": {},   # short-training baseline (chunk20 / lr1e-5 / no aug) as control
    "chunk10": {"policy": {"chunk_size": 10, "n_action_steps": 10}},
    "augment": {"dataset": {"image_transforms": {"enable": True, "backend": "dataloader",
                                                 "max_num_transforms": 3}}},
    # ---- chunk length ----
    "chunk8": {"policy": {"chunk_size": 8, "n_action_steps": 8}},
    "chunk12": {"policy": {"chunk_size": 12, "n_action_steps": 12}},
    "chunk16": {"policy": {"chunk_size": 16, "n_action_steps": 16}},
    # ---- learning rate ----
    "lr1e4": {"policy": {"optimizer_lr": 1e-4, "optimizer_lr_backbone": 1e-4}},
    "lr3e4": {"policy": {"optimizer_lr": 3e-4, "optimizer_lr_backbone": 3e-4}},
    "lr5e4": {"policy": {"optimizer_lr": 5e-4, "optimizer_lr_backbone": 5e-4}},
    # ---- action smoothing (temporal ensembling as ACT-equivalent) ----
    "smooth": {"policy": {"temporal_ensemble_coeff": 0.01, "n_action_steps": 1,
                          "chunk_size": 20}},
    # ---- KEY: closed-loop inference (re-plan every K steps instead of 20) ----
    "nas1": {"policy": {"chunk_size": 20, "n_action_steps": 1}},
    "nas4": {"policy": {"chunk_size": 20, "n_action_steps": 4}},
    "nas10": {"policy": {"chunk_size": 20, "n_action_steps": 10}},
    # ---- combined: best LR + augment (used if chunk/lr winner found) ----
    "aug_lr1e4": {"dataset": {"image_transforms": {"enable": True, "backend": "dataloader",
                                                    "max_num_transforms": 3}},
                  "policy": {"optimizer_lr": 1e-4, "optimizer_lr_backbone": 1e-4}},
}


def deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def build_config(name, steps, save_freq, eval_steps):
    with open(BASE_CFG) as fh:
        base = yaml.safe_load(fh)
    cfg = deep_merge(base, EXPERIMENTS[name])
    out_dir = os.path.join(OUTPUTS, "ablation_" + name)
    cfg["output_dir"] = out_dir
    cfg["job_name"] = "ablation_" + name
    cfg["steps"] = steps
    cfg["save_freq"] = save_freq
    cfg["eval_steps"] = eval_steps
    cfg["env_eval_freq"] = 10**9       # disable built-in rollout during ablation
    cfg["trackio"] = {"enable": False, "project": "lerobot"}
    # keep eval metrics cheap
    cfg["eval"] = {"n_episodes": 1, "batch_size": 1, "use_async_envs": False,
                   "recording": False, "max_episode_steps": 50}
    cfg_path = os.path.join(REPO, "configs", "ablation_" + name + ".yaml")
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    return cfg_path, out_dir


def run_one(name, args):
    cfg_path, out_dir = build_config(name, args.steps, args.save_freq, args.eval_steps)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    log = os.path.join(OUTPUTS, "ablation_" + name + "_train.log")
    print("=" * 70)
    print("[ablation] TRAIN", name, "->", out_dir)
    print("[ablation] cfg:", cfg_path, "| steps:", args.steps)
    t0 = time.time()
    with open(log, "w") as lf:
        p = subprocess.run([sys.executable, TRAINER, "--config_path=" + cfg_path],
                           cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                           env={**os.environ, "MUJOCO_GL": "egl",
                                "LIBGL_ALWAYS_SOFTWARE": "1", "EGL_PLATFORM": "surfaceless"})
    dt = time.time() - t0
    print("[ablation] train rc=%d in %.1fs (log %s)" % (p.returncode, dt, log))
    if p.returncode != 0:
        print("[ablation] TRAIN FAILED; tail of log:")
        print("\n".join(open(log).read().splitlines()[-25:]))
        return {"exp": name, "status": "TRAIN_FAILED", "train_sec": round(dt, 1)}

    ckpt = os.path.join(out_dir, "checkpoints", "last")
    if not os.path.isdir(ckpt):
        cands = sorted(os.path.join(out_dir, "checkpoints", d)
                       for d in os.listdir(os.path.join(out_dir, "checkpoints")))
        ckpt = cands[-1] if cands else None
    if not ckpt:
        return {"exp": name, "status": "NO_CKPT", "train_sec": round(dt, 1)}
    print("[ablation] ckpt:", ckpt)

    metrics_path = os.path.join(REPO, "data", "ablation_eval_" + name + ".json")
    ev_log = os.path.join(OUTPUTS, "ablation_" + name + "_eval.log")
    cmd = [sys.executable, EVALUATOR, "--checkpoint", ckpt,
           "--seeds", "100", "200", "300", "--episodes-per-seed", str(args.eval_eps),
           "--max-steps", "50", "--out", metrics_path]
    with open(ev_log, "w") as lf:
        pe = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                            env={**os.environ, "MUJOCO_GL": "egl",
                                 "LIBGL_ALWAYS_SOFTWARE": "1", "EGL_PLATFORM": "surfaceless"})
    if pe.returncode != 0 or not os.path.isfile(metrics_path):
        print("[ablation] EVAL FAILED; tail:")
        print("\n".join(open(ev_log).read().splitlines()[-20:]))
        return {"exp": name, "status": "EVAL_FAILED", "train_sec": round(dt, 1)}
    m = json.load(open(metrics_path))
    rec = {
        "exp": name, "status": "OK", "train_sec": round(dt, 1),
        "steps": args.steps, "checkpoint": ckpt,
        "n_rollouts": m["n_rollouts"],
        "success_rate": m["success_rate"],
        "avg_distance_error": round(m["avg_distance_error"], 4),
        "avg_distance_error_success": (round(m["avg_distance_error_success"], 4)
                                       if m.get("avg_distance_error_success") else None),
        "avg_completion_steps": (round(m["avg_completion_steps"], 2)
                                 if m.get("avg_completion_steps") else None),
    }
    print("[ablation] RESULT", name, "success=%.3f" % rec["success_rate"],
          "dist=%.4f" % rec["avg_distance_error"])
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", nargs="+", help="experiment names (see --list)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--save-freq", type=int, default=800)
    ap.add_argument("--eval-steps", type=int, default=400)
    ap.add_argument("--eval-eps", type=int, default=5)
    args = ap.parse_args()
    if args.list:
        for k in EXPERIMENTS:
            print(k, "->", json.dumps(EXPERIMENTS[k], ensure_ascii=False))
        return
    names = args.exp
    results = {}
    if os.path.isfile(RESULTS):
        results = json.load(open(RESULTS))
    for name in names:
        if name not in EXPERIMENTS:
            print("[ablation] unknown exp", name); continue
        rec = run_one(name, args)
        results[name] = rec
        with open(RESULTS, "w") as fh:
            json.dump(results, fh, indent=2)
    print("=" * 70)
    print("[ablation] SUMMARY")
    for k, v in results.items():
        print("  %-12s %-12s succ=%s dist=%s" % (
            k, v.get("status"), v.get("success_rate"), v.get("avg_distance_error")))
    print("[ablation] results ->", RESULTS)
    print("ABLATION DONE")


if __name__ == "__main__":
    main()
