"""Elastic-cable check on one module (MuJoCo cable plugin, gravity on).

Layouts (--layout):
  gimbal  (default design) motor -> hollow inner trunnion -> coil A around the inner gimbal
          axis -> ring -> hollow outer trunnion -> coil B around the outer axis -> frame.
  pin     (previous design) out of the hollow crank-pin stub, over the crank arm, through
          the hollow crank shaft to the frame.
  crank   conventional reference: the same route as 'pin' but with the cable end turning
          with the steering (as when the motor itself is steered).
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import mujoco
from geometry import ModuleGeometry
from build_quad import build_bench
import check_clearance as cc


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def cable_bodies(m, p="m_cab"):
    ids = [i for i in range(m.nbody) if m.body(i).name.startswith(p)]
    return ids  # composite bodies are created in chain order


def _transport(v, t0, t1):
    """Parallel transport of v from tangent t0 to t1 (minimal rotation)."""
    axis = np.cross(t0, t1)
    sn, cs = np.linalg.norm(axis), float(np.dot(t0, t1))
    if sn < 1e-12:
        return v
    k = axis/sn
    return v*cs + np.cross(k, v)*sn + k*np.dot(k, v)*(1 - cs)


def twist_and_bend(m, d, ids):
    """Discrete-rod torsion: angle of each body's material frame against the
    parallel-transported frame of its neighbour (cable bodies run along local x).
    Returns total twist (rad), minimum bend radius (m), worst local twist rate (rad/m)."""
    tw, bends, tws = 0.0, [], []
    for a, b in zip(ids[:-1], ids[1:]):
        Ra, Rb = d.xmat[a].reshape(3, 3), d.xmat[b].reshape(3, 3)
        ta, tb = Ra[:, 0], Rb[:, 0]
        ya = _transport(Ra[:, 1], ta, tb)
        ang = math.atan2(np.dot(np.cross(ya, Rb[:, 1]), tb), np.dot(ya, Rb[:, 1]))
        tws.append(ang)
        tw += ang
        bends.append(math.acos(np.clip(np.dot(ta, tb), -1, 1)))
    seg = np.linalg.norm(d.xpos[ids[1]] - d.xpos[ids[0]])
    kmax = max(bends)/seg if bends else 0.0
    return tw, (1/kmax if kmax > 1e-9 else float("inf")), np.array(tws)/seg


SPANS = {"axis": ("m_cab",), "hang": ("m_cab",), "hook": ("m_cabA", "m_cabB"), "clamp": ("m_cabA", "m_cabB"), "gimbal": ("m_cabA", "m_cabB"), "pin": ("m_cab",),
         "crank": ("m_cab",)}


def geometry_for(layout, g=None):
    if g is not None:
        return g
    if layout == "hook":             # v4 (adopted): solid trunnions, hook / bridge brackets
        return ModuleGeometry()
    if layout == "axis":             # v5 candidate: one span up the hollow crank shaft
        return ModuleGeometry(cable_layout="axis")
    if layout == "hang":             # candidate: slack cable from the stub end to the plate
        return ModuleGeometry(cable_layout="hang", ring_bar=0.005, pillow_hz=0.014, arm_y=-0.0225)
    if layout == "gimbal":           # v2: hollow trunnions, 10 mm ring with a channel
        return ModuleGeometry(cable_layout="gimbal", ring_bar=0.005, pillow_hz=0.014,
                              trunnion_y_end=0.048, arm_y=-0.0225)
    if layout == "clamp":            # v3 trial: solid trunnions, clamps next to the axes
        return ModuleGeometry(cable_layout="clamp", ring_bar=0.004, pillow_hz=0.010,
                              bearing_x=0.064, arm_y=-0.0225)
    # v1: the through-the-crank route needs the taller module
    return ModuleGeometry(cable_layout="pin", plate_dz=0.10, ring_bar=0.005, pillow_hz=0.014,
                          arm_y=-0.0225)


def struct_names(layout):
    common = ["m_motor", "m_tube", "m_ringy_p", "m_ringy_n", "m_ringx_p", "m_ringx_n",
              "m_pillow_p", "m_pillow_n", "m_arm_p", "m_arm_n", "m_post_p", "m_post_n",
              "m_crank_arm", "m_crank_hub", "m_pin_bearing", "m_bracket_p", "m_bracket_n",
              "m_tire", "m_hub"]
    if layout == "hang":
        common += ["m_truny_p", "m_truny_n", "m_trunx_p", "m_trunx_n", "m_steer_motor",
                   "m_topbearing", "m_crank_shaft"]
    if layout == "axis":
        # the cable runs inside the hub / shaft / top bearing: those are not obstacles
        common = [n for n in common if n != "m_crank_hub"]
        common += ["m_truny_p", "m_truny_n", "m_trunx_p", "m_trunx_n", "m_steer_motor",
                   "m_belt"]
    if layout == "hook":
        common += ["m_ringstrut", "m_truny_p", "m_truny_n", "m_trunx_p", "m_trunx_n",
                   "m_clampF_arm", "m_steer_motor", "m_topbearing", "m_crank_shaft",
                   "m_hookH_h", "m_hookH_v", "m_bridge_d", "m_bridge_h", "m_bridge_u"]
    if layout == "gimbal":
        common += ["m_ringstrut", "m_truny_n", "m_trunx_n", "m_clampF_arm", "m_steer_motor",
                   "m_topbearing", "m_crank_shaft"]
    if layout == "clamp":
        common += ["m_ringlug", "m_bridge_d", "m_bridge_h", "m_bridge_u", "m_truny_p",
                   "m_truny_n", "m_trunx_p", "m_trunx_n", "m_clampF_arm", "m_steer_motor",
                   "m_topbearing", "m_crank_shaft"]
    return common


def material_curvature(m, d, ids, seg):
    """Bending curvature vector of each joint in the material frame of the first body
    (components about local y and z), 1/m."""
    out = []
    for a, b in zip(ids[:-1], ids[1:]):
        q = quat_mul(quat_conj(d.xquat[a]), d.xquat[b])
        if q[0] < 0:
            q = -q
        out.append((2*q[2]/seg, 2*q[3]/seg))
    return np.array(out)


def run(layout="gimbal", turns=5.0, period=2.0, dt=0.0005, out=None, check_every=0.05,
        g: ModuleGeometry | None = None, video: str = ""):
    g = geometry_for(layout, g)
    attach = {"axis": "axis", "hang": "hang", "hook": "hook", "clamp": "clamp", "gimbal": "gimbal", "pin": "housing", "crank": "crank"}[layout]
    xml, info = build_bench(g, dt=dt, cable=attach, gravity=True)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    spans = {name: cable_bodies(m, name) for name in SPANS[layout]}
    qc, vc = m.joint("m_crank_spin").qposadr[0], m.joint("m_crank_spin").dofadr[0]
    vw = m.joint("m_wheel_spin").dofadr[0]
    ref = {k: twist_and_bend(m, d, ids) for k, ids in spans.items()}
    kap = {k: [] for k in spans}
    ramp, w = 0.5, 2*math.pi/period
    T_move = turns*period + ramp
    settle = 1.0
    log = []
    cc.TORUS = (g.wheel_R, g.tire_r)
    struct = [m.geom(n).id for n in struct_names(layout)]
    pairs = []
    for ids in spans.values():
        geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] in ids]
        # ~2.5 cm next to each clamp is covered by a bend-protection sleeve: not checked
        skip = 4
        # spans of the gimbal-type layouts end in a clamp too (the v1 routes end on a weld
        # far above the plate)
        two_clamps = layout in ("gimbal", "hook", "clamp", "axis", "hang")
        inner = geoms[skip:len(geoms) - (skip if two_clamps else 0)]
        pairs += [(a, b) for a in inner for b in struct]
    cl = cc.Clearance(m, pairs)
    min_clear = (1.0, None)
    part_clear = {}
    next_check = 0.0
    tau_peak = 0.0
    rec = None
    if video:
        import subprocess
        from PIL import Image, ImageDraw, ImageFont
        W, H, fps = 720, 720, 20
        rend = mujoco.Renderer(m, height=H, width=W)
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
        lo = layout in ("gimbal", "clamp", "hook", "axis", "hang")
        cam.azimuth, cam.elevation, cam.distance = 125, -14, 0.62 if lo else 0.75
        cam.lookat[:] = d.xpos[m.body("m_mount").id] + np.array([0, 0, 0.06 if lo else 0.12])
        opt = mujoco.MjvOption(); opt.geomgroup[3] = 1
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 17)
        except OSError:
            font = ImageFont.load_default()
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
                                 "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-an", "-c:v",
                                 "libx264", "-pix_fmt", "yuv420p", "-crf", "25", "-movflags",
                                 "+faststart", video], stdin=subprocess.PIPE)
        rec = [rend, cam, opt, font, proc, 0.0, fps]
        label = {"hang": "slack cable from the stub end to a clamp under the plate (candidate)",
                 "axis": "one span: motor inner side -> hollow crank shaft (candidate v5)",
                 "hook": "cable on external brackets, U loop at each gimbal axis",
                 "clamp": "cable clamped next to each gimbal axis (U loop at each axis)",
                 "gimbal": "cable through hollow trunnions (U loop at each axis)",
                 "pin": "cable through the crank-pin stub (previous design)",
                 "crank": "cable end turning with the steering (conventional)"}[layout]
    settled = False
    while d.time < settle + T_move + 0.5:
        t = d.time - settle
        if t >= 0 and not settled:
            # twist reference: the settled cable (its rest shape is straight)
            ref = {k: twist_and_bend(m, d, ids) for k, ids in spans.items()}
            settled = True
        if t < 0:
            ang, rate = 0.0, 0.0
        elif t < ramp:
            ang, rate = w*t*t/(2*ramp), w*t/ramp
        else:
            ang, rate = min(w*(t - ramp/2), 2*math.pi*turns), (w if t < T_move - ramp/2 else 0.0)
        tau = np.clip(20*(ang - d.qpos[qc]) + 0.4*(rate - d.qvel[vc]), -2, 2)
        d.ctrl[m.actuator("m_steer").id] = tau
        d.ctrl[m.actuator("m_drive").id] = -0.05*d.qvel[vw]
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            return {"layout": layout, "diverged_at_s": d.time}
        if not settled and d.time - settle >= 0:
            # the first logged row after `settle` must already use the settled reference
            # (otherwise it compares against the unsettled initial polyline: a spurious
            # local twist-rate spike of ~1e4 deg/m for routes that sag a lot while settling)
            ref = {k: twist_and_bend(m, d, ids) for k, ids in spans.items()}
            settled = True
        tau_peak = max(tau_peak, abs(tau)) if t > ramp else tau_peak
        if d.time >= next_check:
            next_check += check_every
            row = {"t": d.time, "crank_turns": d.qpos[qc]/(2*math.pi)}
            tw_total, rmin, trate = 0.0, float("inf"), 0.0
            for k, ids in spans.items():
                tw, rb, rates = twist_and_bend(m, d, ids)
                row[f"twist_{k}_deg"] = math.degrees(tw - ref[k][0])
                tw_total += tw - ref[k][0]
                rmin = min(rmin, rb)
                if settled and t > ramp:
                    seg = np.linalg.norm(d.xpos[ids[1]] - d.xpos[ids[0]])
                    kap[k].append(material_curvature(m, d, ids, seg))
                trate = max(trate, float(np.max(np.abs(rates - ref[k][2]))))
            row.update({"twist_deg": math.degrees(tw_total), "min_bend_radius_mm": rmin*1000,
                        "max_twist_rate_deg_per_m": math.degrees(trate),
                        "steer_torque_Nm": float(tau)})
            if pairs and settled and int(round(d.time/check_every)) % 4 == 0:
                dists = [(cl.dist(d, a, b, 0.03), m.geom(b).name) for a, b in pairs]
                for dv, nm in dists:
                    part_clear[nm] = min(part_clear.get(nm, 1.0), dv)
                cmin = min(dists)
                row["cable_clearance_mm"] = cmin[0]*1000
                if cmin[0] < min_clear[0]:
                    min_clear = (cmin[0], cmin[1])
            log.append(row)
            if rec:
                rend, cam, opt, font, proc = rec[:5]
                rend.update_scene(d, cam, opt)
                img = Image.fromarray(rend.render())
                dr = ImageDraw.Draw(img)
                dr.rectangle((0, 0, 720, 54), fill=(22, 27, 34))
                dr.text((10, 6), label, font=font, fill=(240, 243, 246))
                dr.text((10, 30), f"steering turns {row['crank_turns']:+5.2f}   cable twist "
                        f"{row['twist_deg']:+7.1f} deg", font=font, fill=(215, 222, 230))
                for _ in range(max(1, round(rec[6]*check_every))):
                    proc.stdin.write(np.asarray(img).tobytes())
    if rec:
        rec[0].close(); rec[4].stdin.close(); rec[4].wait()
    moving = [r for r in log if r["t"] > settle]
    # cyclic bending: range of the material-frame curvature vector per joint.  A bend
    # whose plane rotates through the material (rotating bending) gives 2*kappa.
    dk = 0.0
    for k, arr in kap.items():
        if arr:
            A = np.array(arr)                                  # time x joints x 2
            for j in range(A.shape[1]):
                P = A[:, j, :]
                c = P.mean(axis=0)
                far = P[np.argmax(np.linalg.norm(P - c, axis=1))]
                dk = max(dk, float(np.max(np.linalg.norm(P - far, axis=1))))
    per_turn = []
    for k in range(int(turns) + 1):
        rows = [r for r in log if abs(r["crank_turns"] - k) < 0.02]
        if rows:
            per_turn.append(round(float(np.mean([r["twist_deg"] for r in rows])), 2))
    res = {"layout": layout, "cable": info, "turns_done": log[-1]["crank_turns"],
           "twist_range_deg": [min(r["twist_deg"] for r in moving), max(r["twist_deg"] for r in moving)],
           "span_twist_range_deg": {k: [min(r[f"twist_{k}_deg"] for r in moving),
                                        max(r[f"twist_{k}_deg"] for r in moving)] for k in spans},
           "twist_at_each_full_turn_deg": per_turn,
           "final_twist_deg": log[-1]["twist_deg"],
           "min_bend_radius_mm": min(r["min_bend_radius_mm"] for r in moving),
           "settled_min_bend_radius_mm": moving[0]["min_bend_radius_mm"],
           "max_local_twist_rate_deg_per_m": max(r["max_twist_rate_deg_per_m"] for r in moving),
           "max_cyclic_curvature_range_per_m": dk,
           "cyclic_bend_radius_equiv_mm": 1000/dk if dk > 0 else float("inf"),
           "min_cable_clearance_mm": min_clear[0]*1000, "closest_part": min_clear[1],
           "clearance_by_part_mm": {k: round(v*1000, 2) for k, v in
                                    sorted(part_clear.items(), key=lambda kv: kv[1])[:6]},
           "peak_steer_torque_Nm": tau_peak,
           "warnings": {str(mujoco.mjtWarning(i)): int(x.number) for i, x in enumerate(d.warning) if x.number}}
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        import csv
        keys = list(dict.fromkeys(k for r in log for k in r))
        with (Path(out) / "trace.csv").open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader()
            wr.writerows([{k: r.get(k, "") for k in keys} for r in log])
        (Path(out) / "model.xml").write_text(xml, encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", choices=("hook", "gimbal", "clamp", "pin", "crank", "axis", "hang"), default="hook")
    ap.add_argument("--turns", type=float, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--video", default="", help="MP4 path (needs ffmpeg and an OpenGL context)")
    a = ap.parse_args()
    out = a.out or f"results/cable_{a.layout}"
    print(json.dumps(run(a.layout, a.turns, out=out, video=a.video), indent=2))
