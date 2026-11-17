#!/usr/bin/env python
"""MuJoCo Studio SimPlugin: auto push-cube demo (mocap-driven).

Drives the `mocap` body along a closed-loop waypoint trajectory. The arm
follows through the weld constraint (`ee_weld` in push_cube.xml) -- no
hand-written joint control anywhere; MuJoCo's constraint solver does the
internal IK / position control.

Cycle (repeats forever, cube is reset at cycle start):
  home -> approach above cube -> descend -> push 0.2m toward the target
  -> lift -> return home

Action space (reserved interface, 4-D):
  a = [dx, dy, dz, gripper]
  * a[:3]  end-effector translation increment (applied to mocap_pos)
  * a[3]   gripper open/close 0..255 (kept closed = 0 for pushing; the
           interface is reserved for future grasp tasks)

Run (in the `mujoco` conda env):
    ./scripts/start_push.sh
    # or: mujoco-studio --model=assets/scenes/push_cube.xml \
    #        --plugins=push_sim.py --gfx=web --port=8083 --hot_reload=all
"""
import math
from mujoco.studio import messages, plugin

# mocap waypoints (world XYZ for the hand target)
HOME     = [0.088,  0.000, 0.926]
ABOVE    = [0.620, -0.120, 0.900]
CONTACT  = [0.620, -0.120, 0.852]
PUSH_END = [0.800, -0.120, 0.845]
LIFT     = [0.800, -0.120, 0.950]

# cube initial freejoint pose (matches the push_home keyframe)
CUBE_QPOS = [0.62, -0.12, 0.775, 1.0, 0.0, 0.0, 0.0]

# (target, seconds) segments; smoothstep interpolation inside each segment
TRAJECTORY = [
    (ABOVE,    1.2),
    (CONTACT,  1.0),
    (PUSH_END, 2.0),   # slow push: cube slides toward the red target disc
    (LIFT,     0.8),
    (HOME,     1.5),
]

def _smooth(t):
    return t * t * (3.0 - 2.0 * t)

class PushCubeDemo(plugin.SimPlugin):
    """Drive the mocap EE target along the push trajectory."""

    def __init__(self):
        self.t = 0.0          # time inside the current segment
        self.seg = 0          # current segment index
        self.start = list(HOME)

    def _reset_cube(self, data):
        data.qpos[9:16] = CUBE_QPOS
        data.qvel[9:15] = 0.0

    def apply_action(self, data, a):
        """Reserved 4-D action interface: [dx, dy, dz, gripper]."""
        data.mocap_pos[0][0] += a[0]
        data.mocap_pos[0][1] += a[1]
        data.mocap_pos[0][2] += a[2]
        # a[3]: gripper 0..255 -> finger actuator would go here (kept closed)

    @messages.handler(priority=messages.Priority.USER)
    def on_pre_step(self, event: messages.PreStepEvent) -> None:
        m, d = event.model, event.data
        self.t += m.opt.timestep
        target, dur = TRAJECTORY[self.seg]
        k = _smooth(min(self.t / dur, 1.0))
        for i in range(3):
            d.mocap_pos[0][i] = self.start[i] + k * (target[i] - self.start[i])
        if self.t >= dur:
            self.seg = (self.seg + 1) % len(TRAJECTORY)
            self.t = 0.0
            self.start = list(d.mocap_pos[0])
            if self.seg == 0:               # new cycle: reset the cube
                self._reset_cube(d)
