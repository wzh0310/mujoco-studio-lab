# -*- coding: utf-8 -*-
"""Visualization helpers for MuJoCo offscreen rendering and PNG export.

Kept dependency-free (only numpy + stdlib) so it works in the `mujoco` conda
env where PIL/cv2 are not installed. Used by stress tests and demo rendering.
"""
import struct
import zlib

import numpy as np


def render_camera(model, data, renderer, camera="top_cam"):
    """Render a camera to an RGB uint8 array of shape (H, W, 3)."""
    renderer.update_scene(data, camera=camera)
    return renderer.render()


def save_png(path, rgb):
    """Write an (H, W, 3) uint8 RGB array to a PNG file (stdlib only)."""
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:  # grayscale -> RGB
        rgb = np.stack([rgb] * 3, axis=-1)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + np.ascontiguousarray(rgb[r]).tobytes()
                   for r in range(h))

    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload +
                struct.pack(">I", zlib.crc32(tag + payload) & 0xffffffff))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", ihdr))
        fh.write(chunk(b"IDAT", zlib.compress(raw)))
        fh.write(chunk(b"IEND", b""))


def is_black_image(image, threshold=12):
    """True if an image observation looks like a black/blank render."""
    arr = np.asarray(image)
    return arr.size > 0 and float(arr.mean()) < threshold


def frame_diff_mean(img_a, img_b):
    """Mean per-channel absolute pixel difference between two RGB frames."""
    a = np.asarray(img_a, dtype=np.int16)
    b = np.asarray(img_b, dtype=np.int16)
    return float(np.abs(a - b).mean())
