#!/usr/bin/env python
"""Minimal MuJoCo Studio SimPlugin demo.

Runs a bouncing/oscillating ball in a browser WebViewer. This is the
simplest possible Studio plugin: on every PreStepEvent (i.e. right before
mj_step) it writes a sinusoidal target into the position actuator.

Why PreStepEvent: the viewer's mouse perturbation has already been applied
and physics has not advanced yet, so this is the correct place to inject
control. Never call mj_step() yourself here -- StepControl does that.

Run (in the `mujoco` conda env):
    MUJOCO_GL=egl python -m mujoco.studio.viewer \
        --model=model.xml --plugins=simulate.py --gfx=web --port=8080

Then open http://localhost:8080 in a browser. Edit `model.xml` (e.g. the
sphere's rgba) and save -- the page reloads automatically (hot reload).
"""
import math

from mujoco.studio import messages, plugin


class BallOscillator(plugin.SimPlugin):
    """Drive the ball's slide joint with a sine wave via a position servo."""

    def __init__(self):
        self.t = 0.0

    @messages.handler(priority=messages.Priority.USER)
    def on_pre_step(self, event: messages.PreStepEvent) -> None:
        self.t += event.model.opt.timestep
        # 0.2 ~ 2.2 m target, 3-second period, centered at 1.2 m
        event.data.ctrl[0] = 1.2 + 1.0 * math.sin(2 * math.pi * self.t / 3.0)
