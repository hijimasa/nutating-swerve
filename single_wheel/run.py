"""Run a torque-driven, closed-loop MuJoCo experiment; optionally view or record."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

# Set an offscreen backend only when rendering without an interactive viewer.
import sys
if "--video" in sys.argv and "--viewer" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from build_model import Geometry, build_xml


def trajectory(t, cycles, period, ramp=0.5):
    total = cycles * period + ramp
    speed = 2 * math.pi / period
    end = 2 * math.pi * cycles
    if t < 0:
        return 0.0, 0.0
    if t < ramp:
        return speed * t*t / (2*ramp), speed*t/ramp
    if t <= total-ramp:
        return speed*(t-ramp/2), speed
    if t < total:
        return end-speed*(total-t)**2/(2*ramp), speed*(total-t)/ramp
    return end, 0.0


def matrix_angle(a, b):
    cosine = np.clip((np.trace(a.T @ b)-1)/2, -1, 1)
    return math.degrees(math.acos(cosine))


class Recorder:
    def __init__(self, model, path, mode, fps=24):
        from PIL import Image, ImageDraw, ImageFont
        self.Image, self.ImageDraw = Image, ImageDraw
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 17)
        except OSError:
            self.font = ImageFont.load_default()
        self.renderer = mujoco.Renderer(model, height=540, width=720)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.azimuth = 140
        self.cam.elevation = -25
        self.cam.distance = 0.55
        self.mode = mode
        self.next = 0.0
        self.fps = fps
        path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "720x540", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(path)
        ], stdin=subprocess.PIPE)
        self.poster = path.with_suffix(".png")
        self.have_poster = False

    def frame(self, model, data, row):
        if data.time + 1e-8 < self.next:
            return
        self.next += 1/self.fps
        self.cam.lookat[:] = data.body("carrier").xpos + np.array([0, 0, 0.035])
        self.renderer.update_scene(data, self.cam)
        image = self.Image.fromarray(self.renderer.render())
        draw = self.ImageDraw.Draw(image)
        draw.rectangle((0, 0, 720, 85), fill=(20, 26, 33))
        draw.text((14, 8), f"{self.mode.upper()} | passive gimbal + crank | t={data.time:.2f}s", font=self.font, fill=(240, 243, 246))
        draw.text((14, 32), f"turns={row['crank_rad']/(2*math.pi):.2f}   tilt={row['tilt_deg']:.2f} deg   closure={row['closure_m']*1000:.3f} mm", font=self.font, fill=(240, 243, 246))
        draw.text((14, 56), f"gimbal X={row['gimbal_x_deg']:+.1f} deg   Y={row['gimbal_y_deg']:+.1f} deg   wheel={row['wheel_rad_s']:+.2f} rad/s", font=self.font, fill=(240, 243, 246))
        self.proc.stdin.write(np.asarray(image).tobytes())
        if not self.have_poster and data.time > 1:
            image.save(self.poster)
            self.have_poster = True

    def close(self):
        self.renderer.close()
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def contact_metrics(model, data):
    floor = model.geom("floor").id
    tire = model.geom("tire").id
    speed, normal_force, count = [], 0.0, 0
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    force = np.zeros(6)
    for con in data.contact:
        if {int(con.geom1), int(con.geom2)} != {floor, tire}:
            continue
        # floor is static. Velocity of the wheel material at the contact point.
        mujoco.mj_jac(model, data, jacp, jacr, con.pos, model.body("wheel").id)
        velocity = jacp @ data.qvel
        speed.append(float(np.linalg.norm(velocity[:2])))
        count += 1
    for i, con in enumerate(data.contact):
        if {int(con.geom1), int(con.geom2)} == {floor, tire}:
            mujoco.mj_contactForce(model, data, i, force)
            normal_force += float(force[0])
    return count, normal_force, float(np.mean(speed)) if speed else float("nan")


def simulate(args):
    g = Geometry(tilt_deg=args.tilt, timestep=args.dt)
    model = mujoco.MjModel.from_xml_string(build_xml(args.mode, g))
    model.geom_friction[model.geom("tire").id, 1] = args.torsion
    model.geom_friction[model.geom("floor").id, 1] = args.torsion
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.xml").write_text(build_xml(args.mode, g), encoding="utf-8")
    # Preserve any contact coefficient overrides in the actual saved MJCF.
    mujoco.mj_saveLastXML(str(output / "model.xml"), model)
    joints = {name: (model.joint(name).qposadr[0], model.joint(name).dofadr[0])
              for name in ("gimbal_x", "gimbal_y", "crank_spin", "wheel_spin")}
    qcrank, vcrank = joints["crank_spin"]
    _, vwheel = joints["wheel_spin"]
    first_pose = data.body("motor_housing").xmat.reshape(3, 3).copy()
    rows = []
    recorder = Recorder(model, Path(args.video), args.mode) if args.video else None
    viewer = None
    if args.viewer:
        from mujoco import viewer as viewer_module
        viewer = viewer_module.launch_passive(model, data)
        with viewer.lock():
            viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 140, -25, 0.55
    run_time = args.cycles * args.period + 0.5 + 1.0
    next_sample = 0.0
    wall_start = time.monotonic()
    try:
        while data.time < run_time - model.opt.timestep/2:
            if viewer and not viewer.is_running():
                break
            phase, phase_rate = trajectory(data.time, args.cycles, args.period)
            if args.hold_steering:
                phase, phase_rate = 0.0, 0.0
            # Continuous angle error, without wrapping. Passive gimbal joints get no torque.
            tau_steer = np.clip(12*(phase-data.qpos[qcrank]) + 0.15*(phase_rate-data.qvel[vcrank]), -3, 3)
            drive_target = args.wheel_speed * min(data.time/0.5, 1.0)
            tau_drive = np.clip(0.12*(drive_target-data.qvel[vwheel]), -0.8, 0.8)
            if viewer:
                with viewer.lock():
                    data.ctrl[:] = (tau_steer, tau_drive)
                    mujoco.mj_step(model, data)
            else:
                data.ctrl[:] = (tau_steer, tau_drive)
                mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("Non-finite dynamics")
            if data.time + 1e-8 >= next_sample:
                # mj_forward refreshes kinematic fields at the integrated state; it does not set qpos.
                if viewer:
                    with viewer.lock():
                        mujoco.mj_forward(model, data)
                else:
                    mujoco.mj_forward(model, data)
                axis = data.body("motor_housing").xmat.reshape(3, 3)[:, 2]
                ncontact, load, slip = contact_metrics(model, data)
                relative = data.body("carrier").xmat.reshape(3, 3).T @ data.body("motor_housing").xmat.reshape(3, 3)
                row = {
                    "time_s": float(data.time), "target_crank_rad": float(phase),
                    "crank_rad": float(data.qpos[qcrank]), "crank_rad_s": float(data.qvel[vcrank]),
                    "wheel_rad_s": float(data.qvel[vwheel]),
                    "gimbal_x_deg": math.degrees(float(data.qpos[joints['gimbal_x'][0]])),
                    "gimbal_y_deg": math.degrees(float(data.qpos[joints['gimbal_y'][0]])),
                    "tilt_deg": math.degrees(math.acos(np.clip(axis[2], -1, 1))),
                    "axis_x": float(axis[0]), "axis_y": float(axis[1]), "axis_z": float(axis[2]),
                    "closure_m": float(np.linalg.norm(data.site('rear_pin').xpos-data.site('crank_pin').xpos)),
                    "base_x_m": float(data.body('carrier').xpos[0]), "base_y_m": float(data.body('carrier').xpos[1]),
                    "base_z_m": float(data.body('carrier').xpos[2]),
                    "steer_torque_Nm": float(tau_steer), "drive_torque_Nm": float(tau_drive),
                    "contact_count": ncontact, "normal_load_N": load, "contact_slip_m_s": slip,
                    "housing_return_error_deg": matrix_angle(first_pose, relative),
                }
                rows.append(row)
                next_sample += 0.01
                if recorder:
                    recorder.frame(model, data, row)
            if viewer:
                with viewer.lock():
                    viewer.cam.lookat[:] = data.body('carrier').xpos + np.array([0, 0, 0.035])
                viewer.sync()
                remaining = data.time - (time.monotonic()-wall_start)
                if remaining > 0:
                    time.sleep(min(remaining, 0.02))
    finally:
        if viewer:
            viewer.close()
        if recorder:
            recorder.close()
    with (output / "trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    active = [r for r in rows if 1.0 < r['time_s'] < run_time-1.5]
    contact_rows = [r for r in active if r['contact_count'] > 0]
    def span(key):
        return [min(r[key] for r in rows), max(r[key] for r in rows)]
    summary = {
        "mujoco_version": mujoco.__version__, "configuration": vars(args),
        "completed_crank_turns": rows[-1]['crank_rad']/(2*math.pi),
        "closure_max_mm": max(r['closure_m'] for r in rows)*1000,
        "tilt_range_deg": span('tilt_deg'), "gimbal_x_range_deg": span('gimbal_x_deg'),
        "gimbal_y_range_deg": span('gimbal_y_deg'),
        "final_housing_return_error_deg": rows[-1]['housing_return_error_deg'],
        "max_abs_steer_torque_Nm": max(abs(r['steer_torque_Nm']) for r in rows),
        "max_abs_drive_torque_Nm": max(abs(r['drive_torque_Nm']) for r in rows),
        "mean_wheel_speed_rad_s": float(np.mean([r['wheel_rad_s'] for r in active])) if active else None,
        "contact_fraction": len(contact_rows)/len(active) if active else None,
        "mean_contact_slip_m_s": float(np.mean([r['contact_slip_m_s'] for r in contact_rows])) if contact_rows else None,
        "mean_normal_load_N": float(np.mean([r['normal_load_N'] for r in contact_rows])) if contact_rows else None,
        "final_base_xy_m": [rows[-1]['base_x_m'], rows[-1]['base_y_m']],
        "warnings": {str(mujoco.mjtWarning(i)): int(item.number) for i, item in enumerate(data.warning) if item.number},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("air", "ground"), default="air")
    p.add_argument("--tilt", type=float, default=45)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--period", type=float, default=4.0, help="seconds per crank revolution")
    p.add_argument("--wheel-speed", type=float, default=0.0, help="relative rotor speed, rad/s")
    p.add_argument("--hold-steering", action="store_true", help="hold crank at zero for a straight-line test")
    p.add_argument("--torsion", type=float, default=0.003, help="torsional friction length, m")
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--out", default="results/air")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--video", default="", help="MP4 path; requires ffmpeg and an offscreen GL backend")
    args = p.parse_args()
    if args.cycles < 1 or args.period <= 0 or args.dt <= 0 or args.torsion < 0:
        p.error("cycles, period, dt must be positive; torsion must be nonnegative")
    return args


if __name__ == "__main__":
    simulate(parse_args())
