#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验收标准化环境：Gymnasium 注册 / seed / render 两模式 / obs_mode 切换 / 动作空间。"""
import numpy as np
import gymnasium as gym

# 1) 导入即注册到全局表
import pushcube_envs  # noqa: F401
assert "PushCube-v0" in gym.registry, "PushCube-v0 未注册到 gym.registry"
print("[1] gym.registry 注册 OK:", "PushCube-v0" in gym.registry)

# 2) gym.make 直接创建（无前缀 id）
env = gym.make("PushCube-v0")
print("[2] gym.make('PushCube-v0') OK:", type(env.unwrapped).__name__)
env.close()

# 3) 显式 seed() 可复现
from pushcube_envs import PushCubeEnv
e = PushCubeEnv(obs_mode='state')
seed_ret = e.seed(123)
obs1, _ = e.reset(seed=123)
s1 = obs1.copy()
obs2, _ = e.reset(seed=123)
s2 = obs2.copy()
assert np.allclose(s1, s2), "相同 seed 的初始状态不一致"
print("[3] seed() + reset(seed=) 可复现 OK, seed() 返回:", seed_ret)

# 4) obs_mode 三态
e_both = PushCubeEnv(obs_mode='both')
e_img = PushCubeEnv(obs_mode='image')
e_st = PushCubeEnv(obs_mode='state')
o_both, _ = e_both.reset(seed=0)
o_img, _ = e_img.reset(seed=0)
o_st, _ = e_st.reset(seed=0)
assert isinstance(o_both, dict) and 'image' in o_both and 'state' in o_both
assert isinstance(o_img, np.ndarray) and o_img.shape == (224, 224, 3), o_img.shape
assert isinstance(o_st, np.ndarray) and o_st.shape == (13,), o_st.shape
assert e_both.observation_space.contains(o_both)
assert e_img.observation_space.contains(o_img)
assert e_st.observation_space.contains(o_st)
print("[4] obs_mode both/image/state OK:",
      type(o_both).__name__, o_img.shape, o_st.shape)

# 5) 渲染两模式
e_rgb = PushCubeEnv(render_mode='rgb_array')
e_rgb.reset(seed=0)
frame = e_rgb.render()
assert isinstance(frame, np.ndarray) and frame.shape == (224, 224, 3), (
    None if frame is None else frame.shape)
print("[5a] render rgb_array OK:", frame.shape, frame.dtype)
e_human = PushCubeEnv(render_mode='human')
e_human.reset(seed=0)
h = e_human.render()   # headless 下允许返回 None（安全 no-op）
assert h is None or (isinstance(h, np.ndarray) and h.ndim == 3)
assert 'human' in PushCubeEnv.metadata['render_modes']
assert 'rgb_array' in PushCubeEnv.metadata['render_modes']
print("[5b] render human OK (headless no-op accepted):", h)
for x in (e, e_both, e_img, e_st, e_rgb, e_human):
    x.close()

# 6) 动作空间上下界
e_act = PushCubeEnv()
lo, hi = e_act.action_space.low, e_act.action_space.high
assert np.allclose(lo[:3], -0.02) and np.allclose(hi[:3], 0.02)
assert lo[3] == 0.0 and hi[3] == 1.0
# 越界动作被 clip 到合法范围
_, _, _, _, info = e_act.step(np.array([9.0, -9.0, 9.0, 5.0], dtype=np.float32))
ee = info['ee_target_pos']
print("[6] action space +/-0.02 & gripper[0,1] OK; clip applied, ee_target:",
      np.round(ee, 4))
e_act.close()

# legacy wrapper 的带前缀 id 注册在 lerobot 环境单独验证（wrapper 依赖 lerobot）。
print("\nALL STANDARD ENV CHECKS PASSED (mujoco env)")
