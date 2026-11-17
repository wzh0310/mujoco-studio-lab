# -*- coding: utf-8 -*-
"""Dataset / observation validation helpers used by the stress test.

All checks are pure functions over numpy arrays or MuJoCo objects, so they can
be unit-tested and reused by both `scripts/stress_test.py` and notebooks.
"""
import numpy as np
import mujoco


def contact_penetrations(model, data):
    """Max contact penetration (m) grouped by task-relevant body pairs.

    MuJoCo 4.0's MjContact has no ``body1``/``body2`` attributes; body ids must
    be resolved through ``model.geom_bodyid[contact.geom1/geom2]``. This helper
    encapsulates that so callers don't repeat the (easy-to-get-wrong) mapping.

    Returns a dict with keys: cube_table, arm_table, cube_finger.
    """
    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    cube = bid("cube")
    table = bid("table")
    fingers = {bid("left_finger"), bid("right_finger")}
    arm = {bid(f"link{i}") for i in range(8)} | {bid("hand")}
    geom2body = np.array([int(model.geom_bodyid[i]) for i in range(model.ngeom)])

    out = {"cube_table": 0.0, "arm_table": 0.0, "cube_finger": 0.0}
    for i in range(data.ncon):
        c = data.contact[i]
        depth = -float(c.dist)
        if depth <= 0:
            continue
        b1 = int(geom2body[c.geom1])
        b2 = int(geom2body[c.geom2])
        pair = {b1, b2}
        if cube in pair and table in pair:
            out["cube_table"] = max(out["cube_table"], depth)
        if table in pair and (pair & arm):
            out["arm_table"] = max(out["arm_table"], depth)
        if cube in pair and (pair & fingers):
            out["cube_finger"] = max(out["cube_finger"], depth)
    return out


def check_episode_alignment(images, actions, states, lengths, fps, physics_dt):
    """Sanity-check observation/action temporal alignment for raw demos.

    Returns a report dict. Key invariants:
      - len(images) == len(actions) == len(states) == sum(lengths)
      - fps matches 1 / physics_dt (timestamp step used by LeRobot export)
      - cube-position state jumps are bounded within an episode (no misframe)
      - images change during episodes (not a frozen/black stream)
    """
    rep = {}
    rep["lengths_match"] = (
        len(images) == len(actions) == len(states) == int(np.sum(lengths)))
    rep["n_frames"] = len(images)
    rep["fps"] = int(fps)
    rep["physics_dt"] = float(physics_dt)
    rep["fps_consistent"] = abs(int(fps) - 1.0 / float(physics_dt)) < 0.6

    cube_cols = slice(7, 10)
    jumps = np.linalg.norm(np.diff(np.asarray(states)[:, cube_cols], axis=0),
                           axis=1)
    bounds = np.cumsum(np.asarray(lengths, dtype=int))[:-1]
    interior = np.ones(len(jumps), dtype=bool)
    if len(bounds):
        interior[bounds - 1] = False  # ignore episode-boundary resets
    rep["cube_max_jump_interior"] = (
        float(jumps[interior].max()) if interior.any() else 0.0)
    rep["cube_mean_jump"] = float(jumps.mean()) if len(jumps) else 0.0

    imgs = np.asarray(images)
    if len(imgs) > 1:
        diffs = np.abs(imgs[1:].astype(np.int16) -
                       imgs[:-1].astype(np.int16)).mean(axis=(1, 2, 3))
        idiff = diffs[interior] if len(diffs) == len(interior) else diffs
        rep["image_mean_diff_interior"] = float(idiff.mean())
    else:
        rep["image_mean_diff_interior"] = 0.0
    return rep


def check_reset_stability(reset_fn, n=20, seed_start=0):
    """Call reset_fn(seed) n times; report max deviation of key qpos/positions.

    reset_fn must return (obs, info) and leave the MuJoCo data populated. It may
    also attach the data object via an attribute if needed; here we instead
    accept a callable returning (obs, cube_xyz, hand_xyz, qpos, mocap_pos).
    """
    rows = np.array([np.concatenate(reset_fn(seed_start + s)) for s in range(n)])
    return {
        "n_resets": int(n),
        "max_dev": float(np.abs(rows - rows.mean(axis=0)).max()),
        "mean": rows.mean(axis=0).tolist(),
    }
