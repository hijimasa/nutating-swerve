"""Run one torque-driven scenario of the 4-module vehicle; log, summarise, optionally record."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

if "--video" in sys.argv and "--viewer" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco

from build_quad import MODULE_NAMES, build_vehicle
from controller import QuadController, wrap, yaw_of
from geometry import VehicleGeometry, gimbal_state
from scenarios import SCENARIOS

BUMPS = [(0.55, 0.17)]          # under the left wheel track, 8 mm high


def rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_angle(Ra, Rb):
    c = np.clip((np.trace(Ra.T @ Rb) - 1)/2, -1, 1)
    return math.degrees(math.acos(c))


class Recorder:
    """Two views side by side: the whole vehicle (left) and a close-up of the FL module."""

    def __init__(self, model, path, title, fps=20, w=1280, h=600, closeup_body="fl_mount"):
        from PIL import Image, ImageDraw, ImageFont
        self.Image, self.Draw = Image, ImageDraw
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except OSError:
            self.font = ImageFont.load_default()
        self.w, self.h = w, h
        self.wl = int(w*0.56)
        self.top = 92
        self.rl = mujoco.Renderer(model, height=h - self.top, width=self.wl)
        self.rr = mujoco.Renderer(model, height=h - self.top, width=w - self.wl)
        self.cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model, self.cam)
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 140, -22, 1.35
        self.cam2 = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model, self.cam2)
        self.cam2.azimuth, self.cam2.elevation, self.cam2.distance = 120, -10, 0.55
        self.body = model.body(closeup_body).id
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 1
        self.next, self.fps, self.title = 0.0, fps, title
        path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                                      "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                                      "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "25",
                                      "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        self.poster = path.with_suffix(".png")
        self.poster_done = False

    def frame(self, d, lines, lookat):
        if d.time + 1e-9 < self.next:
            return
        self.next += 1/self.fps
        self.cam.lookat[:] = lookat
        self.rl.update_scene(d, self.cam, self.opt)
        left = self.rl.render()
        self.cam2.lookat[:] = d.xpos[self.body] + np.array([0, 0, 0.04])
        self.cam2.azimuth = 120 + np.degrees(yaw_of(d.qpos[3:7]))
        self.rr.update_scene(d, self.cam2, self.opt)
        right = self.rr.render()
        canvas = np.zeros((self.h, self.w, 3), np.uint8)
        canvas[:self.top] = (22, 27, 34)
        canvas[self.top:, :self.wl] = left
        canvas[self.top:, self.wl:] = right
        img = self.Image.fromarray(canvas)
        dr = self.Draw.Draw(img)
        dr.text((12, 6), self.title, font=self.font, fill=(240, 243, 246))
        for k, ln in enumerate(lines):
            dr.text((12, 28 + 20*k), ln, font=self.font, fill=(215, 222, 230))
        dr.rectangle((self.wl, self.top, self.w, self.top + 26), fill=(22, 27, 34))
        dr.text((self.wl + 10, self.top + 4), "FL module close-up (orange = motor housing, purple = crank)",
                font=self.font, fill=(240, 243, 246))
        self.proc.stdin.write(np.asarray(img).tobytes())
        if not self.poster_done and d.time > 4.5:
            img.save(self.poster)
            self.poster_done = True

    def close(self):
        self.rl.close(); self.rr.close()
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def tire_contacts(m, d, tires, wheel_bodies, axes):
    """Per tire: normal force, slip (lateral along e, longitudinal along t), count."""
    out = [[0.0, 0.0, 0.0, 0] for _ in tires]
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv)); f6 = np.zeros(6)
    for i in range(d.ncon):
        con = d.contact[i]
        for k, tg in enumerate(tires):
            # contacts are detected within a 1 mm margin (gap 1 mm, i.e. no force there);
            # count a tire as touching when it is within 0.2 mm of the surface
            if tg in (con.geom1, con.geom2) and con.dist < 2e-4:
                mujoco.mj_contactForce(m, d, i, f6)
                mujoco.mj_jac(m, d, jacp, jacr, con.pos, wheel_bodies[k])
                v = jacp @ d.qvel
                e, t = axes[k]
                o = out[k]
                o[0] += f6[0]
                o[1] = max(o[1], abs(v @ e)); o[2] = max(o[2], abs(v @ t)); o[3] += 1
    return out


WALL0 = time.monotonic()


def simulate(args):
    global WALL0
    WALL0 = time.monotonic()
    vg = VehicleGeometry(rocker=args.rocker)
    ref = SCENARIOS[args.scenario]()
    bumps = BUMPS if args.bumps else None
    xml = build_vehicle(vg, dt=args.dt, bumps=bumps)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "model.xml").write_text(xml, encoding="utf-8")
    ctl = QuadController(m, ref, vg, flip=(None if args.flip is None else args.flip == "on"),
                         feedback=not args.no_feedback, coupling_ff=not args.no_coupling_ff)
    tires = [m.geom(f"{n}_tire").id for n in MODULE_NAMES]
    wbodies = [m.body(f"{n}_wheel").id for n in MODULE_NAMES]
    hous = [m.body(f"{n}_housing").id for n in MODULE_NAMES]
    mounts = [m.body(f"{n}_mount").id for n in MODULE_NAMES]
    jgx = [m.joint(f"{n}_gimbal_x").qposadr[0] for n in MODULE_NAMES]
    jgy = [m.joint(f"{n}_gimbal_y").qposadr[0] for n in MODULE_NAMES]
    jcr = [m.joint(f"{n}_crank_spin").qposadr[0] for n in MODULE_NAMES]
    jwh = [m.joint(f"{n}_wheel_spin").dofadr[0] for n in MODULE_NAMES]
    # settle on the tires before the reference clock starts (not logged)
    SETTLE = 0.5
    while d.time < SETTLE:
        ctl.hold(d)
        mujoco.mj_step(m, d)
    ctl.t_offset = d.time
    t_start = d.time
    rel0 = [d.xmat[mounts[k]].reshape(3, 3).T @ d.xmat[hous[k]].reshape(3, 3) for k in range(4)]
    rec = None
    if args.video:
        rec = Recorder(m, Path(args.video), f"{ref.name}: {ref.notes}")
    viewer = None
    if args.viewer:
        from mujoco import viewer as mv
        viewer = mv.launch_passive(m, d)
    rows = []
    next_log = t_start
    T = ref.T + t_start
    wall0 = time.monotonic()
    energy = 0.0
    try:
        while d.time < T:
            if viewer and not viewer.is_running():
                break
            ctl.update(d, m.opt.timestep)
            mujoco.mj_step(m, d)
            energy += sum(abs(d.actuator_force[a]*d.actuator_velocity[a]) for a in range(m.nu))*m.opt.timestep
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                raise RuntimeError(f"non-finite state at t={d.time:.3f}")
            if d.time + 1e-9 >= next_log:
                next_log += 0.01
                yaw = yaw_of(d.qpos[3:7])
                rp = ref.pose(d.time - t_start)
                c, s = math.cos(yaw), math.sin(yaw)
                axes = []
                for k in range(4):
                    phi = d.qpos[jcr[k]] + yaw
                    axes.append((np.array([math.cos(phi), math.sin(phi), 0]),
                                 np.array([-math.sin(phi), math.cos(phi), 0])))
                tc = tire_contacts(m, d, tires, wbodies, axes)
                row = {"t": d.time - t_start, "x": d.qpos[0], "y": d.qpos[1], "z": d.qpos[2], "yaw": yaw,
                       "x_ref": rp[0], "y_ref": rp[1], "yaw_ref": rp[2],
                       "pos_err": float(np.hypot(rp[0] - d.qpos[0], rp[1] - d.qpos[1])),
                       "yaw_err_deg": math.degrees(wrap(rp[2] - yaw))}
                if args.rocker:
                    row["rocker_deg"] = math.degrees(d.qpos[m.joint("rocker").qposadr[0]])
                for k, n in enumerate(MODULE_NAMES):
                    rel = d.xmat[mounts[k]].reshape(3, 3).T @ d.xmat[hous[k]].reshape(3, 3)
                    # ideal housing pose for the measured crank angle (a function of phi only)
                    gx, gy, _, _ = gimbal_state(vg.module, d.qpos[jcr[k]])
                    ideal = rotx(gx) @ roty(gy)
                    row.update({
                        f"{n}_phi": d.qpos[jcr[k]], f"{n}_phi_tgt": ctl.phi_tgt[k],
                        f"{n}_gx_deg": math.degrees(d.qpos[jgx[k]]),
                        f"{n}_gy_deg": math.degrees(d.qpos[jgy[k]]),
                        f"{n}_wheel_rad_s": d.qvel[jwh[k]],
                        f"{n}_tau_s": ctl.tau_s[k], f"{n}_tau_d": ctl.tau_d[k],
                        f"{n}_Fn": tc[k][0], f"{n}_slip_lat": tc[k][1], f"{n}_slip_lon": tc[k][2],
                        f"{n}_ncon": tc[k][3],
                        f"{n}_housing_rot_deg": rot_angle(rel0[k], rel),
                        f"{n}_housing_dev_deg": rot_angle(ideal, rel)})
                rows.append(row)
                if rec:
                    turns = (d.qpos[jcr[0]])/(2*math.pi)
                    rec.frame(d, [
                        f"t={d.time:5.2f}s  pos err={row['pos_err']*1000:5.1f} mm  yaw err={row['yaw_err_deg']:+5.2f} deg",
                        f"FL crank turns={turns:+5.2f}  FL motor-housing rotation vs start={row['fl_housing_rot_deg']:5.1f} deg",
                        "loads N: " + "  ".join(f"{n.upper()} {row[n+'_Fn']:5.1f}" for n in MODULE_NAMES)],
                        np.array([d.qpos[0], d.qpos[1], 0.12]))
            if viewer:
                viewer.sync()
                lag = d.time - (time.monotonic() - wall0)
                if lag > 0:
                    time.sleep(min(lag, 0.02))
    finally:
        if rec:
            rec.close()
        if viewer:
            viewer.close()
    with (out / "trace.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    summary = summarize(rows, ref, args, m, d, energy)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def summarize(rows, ref, args, m, d, energy):
    act = [r for r in rows if r["t"] > 0.3]
    S = {"scenario": ref.name, "notes": ref.notes, "rocker": bool(args.rocker),
         "bumps": bool(args.bumps), "dt": args.dt, "sim_time_s": rows[-1]["t"],
         "max_pos_err_mm": 1000*max(r["pos_err"] for r in act),
         "rms_pos_err_mm": 1000*math.sqrt(np.mean([r["pos_err"]**2 for r in act])),
         "final_pos_err_mm": 1000*rows[-1]["pos_err"],
         "max_yaw_err_deg": max(abs(r["yaw_err_deg"]) for r in act),
         "final_yaw_err_deg": rows[-1]["yaw_err_deg"],
         "actuator_energy_J": energy,
         "wall_time_s": time.monotonic() - WALL0,
         "warnings": {str(mujoco.mjtWarning(i)): int(w.number) for i, w in enumerate(d.warning) if w.number},
         "modules": {}}
    if args.rocker:
        S["rocker_range_deg"] = [min(r["rocker_deg"] for r in act), max(r["rocker_deg"] for r in act)]
    for n in MODULE_NAMES:
        S["modules"][n] = {
            "crank_turns": (rows[-1][f"{n}_phi"] - rows[0][f"{n}_phi"])/(2*math.pi),
            "gimbal_x_range_deg": [min(r[f"{n}_gx_deg"] for r in act), max(r[f"{n}_gx_deg"] for r in act)],
            "gimbal_y_range_deg": [min(r[f"{n}_gy_deg"] for r in act), max(r[f"{n}_gy_deg"] for r in act)],
            "max_housing_rot_deg": max(r[f"{n}_housing_rot_deg"] for r in act),
            "max_housing_dev_from_phi_model_deg": max(r[f"{n}_housing_dev_deg"] for r in act),
            "final_housing_rot_deg": rows[-1][f"{n}_housing_rot_deg"],
            "max_abs_tau_steer_Nm": max(abs(r[f"{n}_tau_s"]) for r in act),
            "max_abs_tau_drive_Nm": max(abs(r[f"{n}_tau_d"]) for r in act),
            "contact_fraction": float(np.mean([r[f"{n}_ncon"] > 0 for r in act])),
            "Fn_mean_N": float(np.mean([r[f"{n}_Fn"] for r in act])),
            "Fn_p1_N": float(np.percentile([r[f"{n}_Fn"] for r in act], 1)),
            "Fn_p99_N": float(np.percentile([r[f"{n}_Fn"] for r in act], 99)),
            "Fn_below_2N_fraction": float(np.mean([r[f"{n}_Fn"] < 2.0 for r in act])),
            "slip_lat_rms_mm_s": 1000*math.sqrt(np.mean([r[f"{n}_slip_lat"]**2 for r in act])),
            "slip_lon_rms_mm_s": 1000*math.sqrt(np.mean([r[f"{n}_slip_lon"]**2 for r in act])),
        }
    return S


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="square")
    p.add_argument("--rocker", action="store_true",
                   help="optional front rocker (default: rigid 4-point chassis)")
    p.add_argument("--bumps", action="store_true", help="8 mm bump under the left wheel track")
    p.add_argument("--flip", choices=("on", "off"), default=None, help="override wheel reversal")
    p.add_argument("--no-feedback", action="store_true", help="pure model control, no pose trim")
    p.add_argument("--no-coupling-ff", action="store_true")
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--out", default="results/quad_square")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--video", default="")
    return p.parse_args(argv)


if __name__ == "__main__":
    simulate(parse())
