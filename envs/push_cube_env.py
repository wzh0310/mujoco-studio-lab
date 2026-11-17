# -*- coding: utf-8 -*-
"""Gymnasium environment for the push-cube task (mocap + weld driven).

Standard Env interface on top of `assets/scenes/push_cube.xml`:

- **Observation** (`spaces.Dict`):
  - `image`: Box(224, 224, 3) uint8, fixed `top_cam` RGB view (core obs)
  - `state`: Box(13,) float32 -- arm joint angles (7) + cube pos (3) + target pos (3)
- **Action**: Box(4,) -- `[dx, dy, dz, gripper]` per-step increments.
  `a[:3]` is added to the mocap body target (clamped to +-0.02 m / step and to a
  reachable workspace box); no joint torques are touched. `a[3]` is the gripper
  (0..1, reserved for future grasp tasks; kept closed here).
- **Reward**: dense `-||cube_xy - target_xy||` + success bonus when the cube is
  pushed onto the red target disc.

Example:
    env = PushCubeEnv()
    obs, info = env.reset(seed=0)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
"""
import os

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

DEFAULT_XML = os.path.join(os.path.dirname(__file__), '..', 'assets', 'scenes', 'push_cube.xml')

# mocap (EE target) home pose == hand pose at the all-zero home keyframe
HOME_POS = np.array([0.088, 0.0, 0.926])
HOME_QUAT = np.array([0.0, 0.92388, 0.382683, 0.0])
# reachable workspace clamp for the mocap target (keeps the arm away from table collisions)
MOCAP_LOW = np.array([0.10, -0.25, 0.82])
MOCAP_HIGH = np.array([0.85, 0.25, 1.15])

TARGET_XY = np.array([0.82, -0.12])   # red disc centre (on the table)
SUCCESS_RADIUS = 0.06                  # disc radius
SUCCESS_BONUS = 5.0
IMG_SIZE = 224

STATE_LOW = np.array([-3.0] * 7 + [0.3, -0.4, 0.70] + [0.3, -0.4, 0.70], dtype=np.float32)
STATE_HIGH = np.array([3.0] * 7 + [1.3, 0.4, 0.90] + [1.3, 0.4, 0.90], dtype=np.float32)


class PushCubeEnv(gym.Env):
    """Push the 0.1kg cube from its start position onto the red target disc."""

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 50}

    def __init__(self, xml_path=DEFAULT_XML, frame_skip=10, image_obs=True,
                 max_episode_steps=100, render_mode=None):
        super().__init__()
        self.xml_path = os.path.abspath(xml_path)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.image_obs = bool(image_obs)
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self._home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'push_home')
        self._cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        self._renderer = None
        if self.image_obs:
            try:
                self._renderer = mujoco.Renderer(self.model, height=IMG_SIZE, width=IMG_SIZE)
            except Exception as e:                       # headless without EGL etc.
                import warnings
                warnings.warn(f'image observation disabled (renderer init failed: {e})')
                self._renderer = None

        # spaces
        self.action_space = spaces.Box(
            low=np.array([-0.02, -0.02, -0.02, 0.0], dtype=np.float32),
            high=np.array([0.02, 0.02, 0.02, 1.0], dtype=np.float32),
            dtype=np.float32)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(IMG_SIZE, IMG_SIZE, 3),
                                dtype=np.uint8) if self._renderer else
                     spaces.Box(low=0, high=0, shape=(1,), dtype=np.uint8),
            'state': spaces.Box(low=STATE_LOW, high=STATE_HIGH, dtype=np.float32),
        })

    # ------------------------------------------------------------------ helpers
    def _get_image(self):
        if self._renderer is None:
            return np.zeros((1,), dtype=np.uint8)
        self._renderer.update_scene(self.data, camera='top_cam')
        return self._renderer.render()          # uint8 RGB, (224, 224, 3)

    def _get_state(self):
        return np.concatenate([
            self.data.qpos[:7].copy(),          # arm joint angles
            self.data.xpos[self._cube_body].copy(),   # cube position
            np.array([TARGET_XY[0], TARGET_XY[1], 0.752]),  # target position
        ]).astype(np.float32)

    def _get_obs(self):
        return {'image': self._get_image(), 'state': self._get_state()}

    def _cube_target_dist(self):
        c = self.data.xpos[self._cube_body]
        return float(np.linalg.norm(c[:2] - TARGET_XY))

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        self.data.mocap_pos[0] = HOME_POS
        self.data.mocap_quat[0] = HOME_QUAT
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        obs = self._get_obs()
        info = {'cube_target_dist': self._cube_target_dist(), 'success': False}
        return obs, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64),
                         self.action_space.low, self.action_space.high)
        # per-step EE increment on the mocap target (weld constraint does the rest)
        target = self.data.mocap_pos[0] + action[:3]
        self.data.mocap_pos[0] = np.clip(target, MOCAP_LOW, MOCAP_HIGH)
        # action[3] = gripper 0..1: interface reserved, kept closed for pushing

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        dist = self._cube_target_dist()
        success = dist < SUCCESS_RADIUS
        reward = -dist + (SUCCESS_BONUS if success else 0.0)
        terminated = bool(success)
        truncated = bool(self._steps >= self.max_episode_steps)
        info = {'cube_target_dist': dist, 'success': success,
                'ee_target_pos': self.data.mocap_pos[0].copy(),
                'cube_pos': self.data.xpos[self._cube_body].copy()}
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        return self._get_image()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
