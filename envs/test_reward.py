# -*- coding: utf-8 -*-
"""Reward & termination debug checks.

1. Inject four key states (initial / mid / success / fail) and verify the
   reward components match hand-computed expectations.
2. Run full random episodes and verify truncation at max_episode_steps.
"""
import numpy as np
from push_cube_env import PushCubeEnv

env = PushCubeEnv()
CUBE_QPOS = slice(9, 16)          # freejoint: pos(3) + quat(4)
ok = True


def check(name, got, want, tol=1e-6):
    global ok
    good = abs(got - want) <= tol
    ok = ok and good
    print(f'  [{"PASS" if good else "FAIL"}] {name}: got {got:+.4f}  want {want:+.4f}')


def inject_cube(pos):
    env.data.qpos[CUBE_QPOS] = [pos[0], pos[1], pos[2], 1, 0, 0, 0]
    env.data.qvel[9:15] = 0.0
    mujoco_forward()


def mujoco_forward():
    import mujoco
    mujoco.mj_forward(env.model, env.data)


print('=== 1. initial state (right after reset) ===')
obs, info = env.reset(seed=0)
# after reset _prev_dist == dist -> shaping 0; r = -K_EE*ee - TIME
r, comps, term = env._compute_reward()
print(f'  dist={comps["cube_target_dist"]:.3f} ee={comps["ee_cube_dist"]:.3f}')
check('r_shaping == 0', comps['r_shaping'], 0.0)
check('r_time == -0.01', comps['r_time'], -0.01)
check('r_ee == -0.1*ee', comps['r_ee'], -0.1 * comps['ee_cube_dist'])
check('terminated == False', float(term), 0.0)

print('=== 2. mid state: cube pushed halfway, distance reduced ===')
inject_cube([0.72, -0.12, 0.775])
env._prev_dist = 0.20                      # simulated previous distance
r, comps, term = env._compute_reward()
d = comps['cube_target_dist']
print(f'  dist={d:.3f} (was 0.20)')
check('r_shaping == -10*(d-0.20)', comps['r_shaping'], -10.0 * (d - 0.20))
check('shaping positive (improved)', float(comps['r_shaping'] > 0), 1.0)
check('terminated == False', float(term), 0.0)

print('=== 3. success state: cube on target centre ===')
inject_cube([0.82, -0.12, 0.775])
env._prev_dist = 0.10
r, comps, term = env._compute_reward()
print(f'  dist={comps["cube_target_dist"]:.4f}')
check('dist < 0.03', float(comps['cube_target_dist'] < 0.03), 1.0)
check('r_sparse == +10', comps['r_sparse'], 10.0)
check('terminated == True', float(term), 1.0)

print('=== 4. fail state: cube fell off the table (y out of bounds) ===')
inject_cube([0.62, 0.45, 0.40])
env._prev_dist = 0.10
r, comps, term = env._compute_reward()
check('r_fail == -1', comps['r_fail'], -1.0)
check('terminated == True', float(term), 1.0)
print('=== 4b. fail state: cube below tabletop (z drop) ===')
inject_cube([0.70, 0.00, 0.40])
env._prev_dist = 0.10
r, comps, term = env._compute_reward()
check('r_fail == -1', comps['r_fail'], -1.0)
check('terminated == True', float(term), 1.0)

print('=== 5. full random episodes: truncation at 50 steps ===')
env.reset(seed=1)
total, done, step = 0.0, False, 0
while not done:
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    total += r
    step += 1
    done = term or trunc
print(f'  episode ended at step {step} (term={term} trunc={trunc}) '
      f'total_reward={total:+.3f}')
check('truncated at max_episode_steps', float(step == env.max_episode_steps), 1.0)
check('not terminated by physics', float(term), 0.0)
check('total reward finite', float(np.isfinite(total)), 1.0)

print('=== 6. second random episode (fresh reset works) ===')
env.reset(seed=2)
done, step = False, 0
while not done:
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    step += 1
    done = term or trunc
print(f'  episode 2 ended at step {step} (term={term} trunc={trunc})')
env.close()
print('ALL PASS' if ok else 'SOME CHECKS FAILED')
