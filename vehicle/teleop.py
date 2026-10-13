"""Drive the 4-module vehicle by hand, with the keyboard (in the viewer window) or a gamepad.

The body-frame velocity command (vx forward, vy left, wz yaw rate) is smoothed with
acceleration limits, integrated into a reference pose, and tracked by the same
QuadController that runs the scripted scenarios.  The reference is kept on a short leash
to the simulated pose, so a command the vehicle cannot follow (e.g. while the modules are
still re-steering) does not wind up.

Keyboard (viewer window must have focus; the viewer only reports key presses, so the
commands are increments, like ROS teleop_twist_keyboard):

    Up / Down      forward speed  +/- 0.05 m/s        , / .   yaw rate  +/- 0.2 rad/s (left / right)
    Left / Right   left speed     +/- 0.05 m/s        0       stop (all commands to zero)
    = / -          scale the current command by 1.1 / 0.9

Avoid Space and Backspace: the viewer itself uses them (pause, reset).

Gamepad: left stick = translation, right stick X = yaw, any button = stop.  Uses pygame if it
is installed, otherwise the Linux joystick device (/dev/input/js0) with no extra packages.

    python teleop.py                       # keyboard
    python teleop.py --joystick auto       # gamepad (first device found) + keyboard
    python teleop.py --test                # headless canned sequence, prints a summary (CI smoke test)
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import select
import struct
import sys
import time

import numpy as np
import mujoco

from build_quad import MODULE_NAMES, build_vehicle
from controller import QuadController, wrap, yaw_of
from geometry import VehicleGeometry


# ------------------------------------------------------------------ reference from a twist
class TeleopReference:
    """pose(t)/vel(t) interface of scenarios.Reference, fed by a live body-frame twist."""

    name, notes, flip = "teleop", "manual drive", True
    T = float("inf")

    def __init__(self, v_max=0.4, w_max=1.5, a_max=0.8, alpha_max=4.0,
                 leash_pos=0.05, leash_yaw=math.radians(10)):
        self.v_max, self.w_max, self.a_max, self.alpha_max = v_max, w_max, a_max, alpha_max
        self.leash_pos, self.leash_yaw = leash_pos, leash_yaw
        self.cmd = np.zeros(3)          # requested body twist
        self.tw = np.zeros(3)           # smoothed body twist actually used
        self.p = np.zeros(3)            # reference pose (x, y, yaw), world frame
        self.t = 0.0                    # controller clock of the current pose
        self.leash_hits = 0

    def set_command(self, vx, vy, wz):
        v = np.array([vx, vy], float)
        n = np.linalg.norm(v)
        if n > self.v_max:
            v *= self.v_max/n
        self.cmd[:] = (v[0], v[1], float(np.clip(wz, -self.w_max, self.w_max)))

    def step(self, t_now, dt, actual_pose):
        """Advance the reference to controller time t_now; actual_pose = (x, y, yaw)."""
        dv = self.cmd[:2] - self.tw[:2]
        n = np.linalg.norm(dv)
        if n > self.a_max*dt:
            dv *= self.a_max*dt/n
        self.tw[:2] += dv
        self.tw[2] += float(np.clip(self.cmd[2] - self.tw[2], -self.alpha_max*dt, self.alpha_max*dt))
        c, s = math.cos(self.p[2]), math.sin(self.p[2])
        vx, vy, wz = self.tw
        self.p += dt*np.array([c*vx - s*vy, s*vx + c*vy, wz])
        # leash: never let the reference run further than the controller can trim
        d = self.p[:2] - actual_pose[:2]
        dn = np.linalg.norm(d)
        if dn > self.leash_pos:
            self.p[:2] = actual_pose[:2] + d*self.leash_pos/dn
            self.leash_hits += 1
        e = wrap(self.p[2] - actual_pose[2])
        if abs(e) > self.leash_yaw:
            self.p[2] -= e - math.copysign(self.leash_yaw, e)
            self.leash_hits += 1
        self.t = t_now

    def _extrapolate(self, dt):
        """Pose after dt at the current body twist (exact arc)."""
        vx, vy, wz = self.tw
        if abs(wz) < 1e-9:
            dxb, dyb = vx*dt, vy*dt
        else:
            sa, ca = math.sin(wz*dt), 1 - math.cos(wz*dt)
            dxb, dyb = (vx*sa - vy*ca)/wz, (vx*ca + vy*sa)/wz
        c, s = math.cos(self.p[2]), math.sin(self.p[2])
        return np.array([self.p[0] + c*dxb - s*dyb, self.p[1] + s*dxb + c*dyb, self.p[2] + wz*dt])

    def pose(self, t):
        return self.p.copy() if t <= self.t else self._extrapolate(t - self.t)

    def vel(self, t):
        yaw = self.p[2] + max(0.0, t - self.t)*self.tw[2]
        c, s = math.cos(yaw), math.sin(yaw)
        vx, vy, wz = self.tw
        return np.array([c*vx - s*vy, s*vx + c*vy, wz])


# ------------------------------------------------------------------------------- inputs
class Keyboard:
    """Increment-style commands from the viewer's key callback (GLFW key codes)."""

    STEP_V, STEP_W = 0.05, 0.2
    KEYS = {265: "up", 264: "down", 263: "left", 262: "right", 44: "comma", 46: "period",
            48: "zero", 61: "equal", 45: "minus"}

    def __init__(self, ref: TeleopReference):
        self.ref = ref
        self.v = np.zeros(3)

    def key(self, code):
        k = self.KEYS.get(code)
        if k is None:
            return
        if k == "up":
            self.v[0] += self.STEP_V
        elif k == "down":
            self.v[0] -= self.STEP_V
        elif k == "left":
            self.v[1] += self.STEP_V
        elif k == "right":
            self.v[1] -= self.STEP_V
        elif k == "comma":
            self.v[2] += self.STEP_W
        elif k == "period":
            self.v[2] -= self.STEP_W
        elif k == "zero":
            self.v[:] = 0
        elif k == "equal":
            self.v *= 1.1
        elif k == "minus":
            self.v *= 0.9
        self.ref.set_command(*self.v)

    def poll(self):
        pass


class LinuxJoystick:
    """Raw Linux joystick API (/dev/input/jsN), no dependencies."""

    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.axes, self.buttons = {}, {}

    def poll(self):
        while select.select([self.fd], [], [], 0)[0]:
            buf = os.read(self.fd, 8)
            if len(buf) < 8:
                break
            _, value, typ, num = struct.unpack("IhBB", buf)
            typ &= 0x7F                               # drop the 'initial state' flag
            if typ == 2:
                self.axes[num] = value/32767.0
            elif typ == 1:
                self.buttons[num] = value


class PygameJoystick:
    def __init__(self, index=0):
        import pygame
        pygame.init(); pygame.joystick.init()
        self.js = pygame.joystick.Joystick(index); self.js.init()
        self.pygame = pygame
        self.axes, self.buttons = {}, {}

    def poll(self):
        self.pygame.event.pump()
        self.axes = {i: self.js.get_axis(i) for i in range(self.js.get_numaxes())}
        self.buttons = {i: self.js.get_button(i) for i in range(self.js.get_numbuttons())}


class Gamepad:
    """Left stick = translation, right stick X = yaw rate, any button = stop.

    Stick values are shaped with x*|x| for fine control near the centre; axis indices
    follow the Linux xpad layout (0 = left X, 1 = left Y, 3 = right X) unless overridden."""

    def __init__(self, ref: TeleopReference, device="auto", axes=(0, 1, 3), deadzone=0.1):
        self.ref, self.axes, self.deadzone = ref, axes, deadzone
        self.dev = None
        try:
            self.dev = PygameJoystick()
            print("gamepad: pygame joystick 0")
        except Exception:
            paths = sorted(glob.glob("/dev/input/js*")) if device == "auto" else [device]
            if not paths:
                raise SystemExit("no gamepad found (no /dev/input/js* and pygame not installed)")
            self.dev = LinuxJoystick(paths[0])
            print(f"gamepad: {paths[0]} (Linux joystick API)")

    def _shape(self, x):
        if abs(x) < self.deadzone:
            return 0.0
        x = (abs(x) - self.deadzone)/(1 - self.deadzone)*math.copysign(1.0, x)
        return x*abs(x)

    def poll(self):
        self.dev.poll()
        ax = self.dev.axes
        lx, ly, rx = (ax.get(i, 0.0) for i in self.axes)
        if any(self.dev.buttons.values()):
            self.ref.set_command(0.0, 0.0, 0.0)
            return
        vx = -self._shape(ly)*self.ref.v_max          # stick up = forward
        vy = -self._shape(lx)*self.ref.v_max          # stick left = left
        wz = -self._shape(rx)*self.ref.w_max          # stick left = turn left (CCW)
        self.ref.set_command(vx, vy, wz)


class Script:
    """Canned command sequence for --test (headless)."""

    SEQ = [(0.0, (0.3, 0.0, 0.0)), (2.5, (0.0, 0.3, 0.0)), (5.0, (0.0, 0.0, 1.0)),
           (7.0, (0.25, 0.25, 0.8)), (9.5, (0.0, 0.0, 0.0))]
    T_END = 11.0

    def __init__(self, ref: TeleopReference):
        self.ref = ref
        self.t = 0.0

    def poll(self):
        cmd = (0.0, 0.0, 0.0)
        for t0, c in self.SEQ:
            if self.t >= t0:
                cmd = c
        self.ref.set_command(*cmd)


# --------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joystick", default=None, help="'auto' or a /dev/input/jsN path; omit for keyboard only")
    ap.add_argument("--axes", default="0,1,3", help="gamepad axis indices: left X, left Y, right X")
    ap.add_argument("--speed", type=float, default=0.4, help="max translation speed [m/s]")
    ap.add_argument("--turn", type=float, default=1.5, help="max yaw rate [rad/s]")
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--test", action="store_true", help="headless canned sequence, print a summary")
    args = ap.parse_args(argv)

    vg = VehicleGeometry()
    m = mujoco.MjModel.from_xml_string(build_vehicle(vg, dt=args.dt))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ref = TeleopReference(v_max=args.speed, w_max=args.turn)
    ctl = QuadController(m, ref, vg, flip=True)
    while d.time < 0.5:                                # settle on the tyres
        ctl.hold(d)
        mujoco.mj_step(m, d)
    ctl.t_offset = d.time
    jcr = [m.joint(f"{n}_crank_spin").qposadr[0] for n in MODULE_NAMES]

    inputs = []
    kb = Keyboard(ref)
    viewer = None
    if args.test:
        inputs.append(Script(ref))
    else:
        from mujoco import viewer as mv
        viewer = mv.launch_passive(m, d, key_callback=kb.key)
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 140, -22, 1.6
        inputs.append(kb)
        if args.joystick:
            inputs.append(Gamepad(ref, args.joystick, tuple(int(a) for a in args.axes.split(","))))
        print(__doc__.split("Gamepad:")[0].split("Keyboard")[1])

    dt = m.opt.timestep
    wall0 = time.monotonic() - d.time
    next_sync, next_status = 0.0, 0.0
    log = []
    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break
            t = d.time - ctl.t_offset
            if args.test:
                inputs[0].t = t
                if t > Script.T_END:
                    break
            for src in inputs:
                src.poll()
            pose = np.array([d.qpos[0], d.qpos[1], yaw_of(d.qpos[3:7])])
            ref.step(t, dt, pose)
            ctl.update(d, dt)
            mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all():
                raise RuntimeError(f"non-finite state at t={d.time:.3f}")
            if d.time >= next_sync:
                next_sync += 1/60
                if viewer is not None:
                    viewer.cam.lookat[:] = (d.qpos[0], d.qpos[1], 0.12)
                    viewer.sync()
                    lag = d.time - (time.monotonic() - wall0)
                    if lag > 0:
                        time.sleep(min(lag, 0.05))
            if d.time >= next_status:
                next_status += 0.2
                err = np.hypot(*(ref.p[:2] - pose[:2]))
                log.append((t, *ref.tw, *pose, err))
                if viewer is not None:
                    print(f"\r cmd vx={ref.cmd[0]:+.2f} vy={ref.cmd[1]:+.2f} wz={ref.cmd[2]:+.2f} | "
                          f"x={pose[0]:+.2f} y={pose[1]:+.2f} yaw={math.degrees(pose[2]):+6.1f} deg | "
                          f"ref err {1000*err:4.0f} mm | cranks " +
                          " ".join(f"{d.qpos[j]/(2*math.pi):+.1f}" for j in jcr) + " turns   ",
                          end="", flush=True)
    finally:
        if viewer is not None:
            viewer.close()
        print()
    if args.test:
        L = np.array(log)
        moving = L[np.linalg.norm(L[:, 1:3], axis=1) > 0.05]
        # expected displacement of the scripted sequence, from the smoothed twist actually used
        exp = TeleopReference(v_max=args.speed, w_max=args.turn)
        s = Script(exp)
        for k in range(int(Script.T_END/dt)):
            s.t = k*dt; s.poll(); exp.step(s.t, dt, exp.p)      # leash never engages: ref follows itself
        summary = {"final_pose": [round(float(x), 4) for x in log[-1][4:7]],
                   "expected_pose_no_leash": [round(float(x), 4) for x in exp.p],
                   "ref_tracking_err_mm_max": round(1000*float(L[:, 7].max()), 1),
                   "ref_tracking_err_mm_mean_moving": round(1000*float(moving[:, 7].mean()), 1),
                   "leash_hits": ref.leash_hits,
                   "crank_turns": [round(float(d.qpos[j]/(2*math.pi)), 2) for j in jcr],
                   "warnings": {str(mujoco.mjtWarning(i)): int(w.number) for i, w in enumerate(d.warning) if w.number}}
        import json
        print(json.dumps(summary, indent=2))  # pose = (x, y, yaw[rad]); yaw is compared modulo 2 pi
        fp, ep = np.array(summary["final_pose"]), np.array(summary["expected_pose_no_leash"])
        summary["final_vs_expected"] = {"pos_mm": round(1000*float(np.hypot(*(fp[:2] - ep[:2]))), 1),
                                        "yaw_deg": round(math.degrees(abs(wrap(fp[2] - ep[2]))), 2)}
        ok = (summary["ref_tracking_err_mm_max"] < 80 and not summary["warnings"]
              and summary["final_vs_expected"]["pos_mm"] < 30 and summary["final_vs_expected"]["yaw_deg"] < 3)
        print("teleop test:", "OK" if ok else "FAILED")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
