# -*- coding: utf-8 -*-
"""Industrial offscreen image capture for the PushCube task.

Pure MuJoCo GLContext (EGL / OSMesa) offscreen rendering -- no WebViewer and no
on-screen window required. Two quality presets:

* ``train``: 84x84 grayscale, CHW, float32 [0,1], rendered with a lightweight
  geometry-proxy scene (cube / target / table / EE proxy). The proxy is synced
  from the real simulation data every frame so it stays geometrically aligned,
  while avoiding the expensive Panda mesh -> >200 fps on CPU-only hosts.
* ``demo``: 224x224 RGB, CHW, float32 [0,1], rendered from the FULL Panda scene
  so it is faithful to the real task and aligned with the VLA observation space.

Physics is NEVER changed: the environment keeps simulating the full scene;
this module only renders pictures. Output layout follows SB3 CNN convention
(CHW, normalized [0,1]); pass ``chw=False`` / ``normalize=False`` for HWC uint8.
"""
import os

import mujoco
import numpy as np

LITE_XML = os.path.join(os.path.dirname(__file__), '..', 'assets', 'scenes',
                        'push_cube_lite.xml')

# (height, width, grayscale)
PRESETS = {
    'train': (84, 84, True),
    'demo': (224, 224, False),
}


class OffscreenRenderer:
    """Reusable offscreen renderer with a fast proxy and a faithful full path."""

    def __init__(self, full_model, full_data, cube_body, hand_body,
                 lite_xml=LITE_XML, disable_shadow=True):
        self.full_model = full_model
        self.full_data = full_data
        self.cube_body = cube_body
        self.hand_body = hand_body

        self._lite_xml = lite_xml
        self._disable_shadow = disable_shadow
        # NOTE: GL contexts are created LAZILY on first render, not here.
        # Creating a MuJoCo EGL context before torch initialises a CNN can
        # segfault on hosts with a software EGL stack. Deferring keeps the
        # environment import/construct safe alongside SB3/torch.
        self._full = None
        self._lite = None
        self.lite_model = None
        self.lite_data = None
        self._lite_cube_qadr = None

    def _ensure_init(self):
        if self._full is not None:
            return
        self._full = mujoco.Renderer(self.full_model, height=224, width=224)
        self.lite_model = mujoco.MjModel.from_xml_path(self._lite_xml)
        self.lite_data = mujoco.MjData(self.lite_model)
        if self._disable_shadow:
            for i in range(self.lite_model.nlight):
                self.lite_model.light_castshadow[i] = 0
        self._lite = mujoco.Renderer(self.lite_model, height=84, width=84)
        self._lite_cube_qadr = self.lite_model.jnt_qposadr[
            mujoco.mj_name2id(self.lite_model, mujoco.mjtObj.mjOBJ_JOINT,
                              'cube_free')]

    def _sync_lite(self):
        """Copy the real cube pose + hand position into the proxy scene."""
        cpos = self.full_data.xpos[self.cube_body]
        cquat = self.full_data.xquat[self.cube_body]
        q = self._lite_cube_qadr
        self.lite_data.qpos[q:q + 3] = cpos
        self.lite_data.qpos[q + 3:q + 7] = cquat
        self.lite_data.mocap_pos[0] = self.full_data.xpos[self.hand_body]
        mujoco.mj_forward(self.lite_model, self.lite_data)

    @staticmethod
    def _postprocess(img, gray, normalize, chw):
        if gray:
            img = (0.299 * img[..., 0] + 0.587 * img[..., 1] +
                   0.114 * img[..., 2])
        if normalize:
            img = img.astype(np.float32) / 255.0
            if gray:
                img = img[None, ...]          # (1, H, W)
            else:
                img = np.transpose(img, (2, 0, 1))  # (3, H, W)
        elif chw:
            if gray:
                img = img[None, ...]
            else:
                img = np.transpose(img, (2, 0, 1))
        return img

    def get_rgb_image(self, preset='train', camera='top_cam', size=None,
                      gray=None, chw=True, normalize=True, sync=True):
        """Capture one offscreen RGB observation.

        Args:
            preset: 'train' (84x84 gray proxy) or 'demo' (224x224 full scene).
            camera: camera name in the corresponding scene.
            size: optional (h, w) override (renderer fixed at preset size).
            gray: override grayscale flag.
            chw: return CHW layout (SB3 CNN convention) when True.
            normalize: scale to float32 [0,1] when True.
            sync: for the train preset, sync proxy from real data (default True).

        Returns:
            ndarray: CHW float32 [0,1] by default; HWC uint8 if normalize=False.
        """
        self._ensure_init()
        h, w, g = PRESETS[preset]
        if gray is not None:
            g = gray
        if preset == 'train':
            if sync:
                self._sync_lite()
            self._lite.update_scene(self.lite_data, camera=camera)
            raw = self._lite.render()
        else:
            self._full.update_scene(self.full_data, camera=camera)
            raw = self._full.render()
        return self._postprocess(raw, g, normalize, chw)

    # Convenience aliases ----------------------------------------------------
    def get_train_image(self, **kwargs):
        return self.get_rgb_image(preset='train', **kwargs)

    def get_demo_image(self, **kwargs):
        return self.get_rgb_image(preset='demo', **kwargs)

    def close(self):
        if self._full is not None:
            self._full.close()
        if self._lite is not None:
            self._lite.close()
