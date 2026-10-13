"""Static interference check over the full steering cycle (mj_geomDistance).

Every pair of structure geoms on different bodies is checked except the pairs that
touch by design (a shaft in its own bearing).  Visual-only geoms (group 2) are skipped.
"""
from __future__ import annotations
import itertools
import json
import math
import numpy as np
import mujoco
from geometry import ModuleGeometry, VehicleGeometry, gimbal_state
from build_quad import build_bench, build_vehicle, MODULE_NAMES

# (suffix A, suffix B): bearing seats that are in contact on purpose
DESIGNED = {("trunx_p", "pillow_p"), ("trunx_n", "pillow_n"),
            ("truny_p", "ringx_p"), ("truny_n", "ringx_n"),
            ("stub", "pin_bearing"), ("crank_shaft", "topbearing")}


VEHICLE_DESIGNED = {("plate", "crank_shaft"), ("front_plate", "crank_shaft"),
                    ("plate", "rocker_pivot")}


def designed(n1, n2):
    for a, b in ((n1, n2), (n2, n1)):
        for p, q in VEHICLE_DESIGNED:
            if a == p and (b == q or b.endswith("_" + q)):
                return True
    s1, s2 = n1.split("_", 1)[-1], n2.split("_", 1)[-1]
    same_module = n1.split("_", 1)[0] == n2.split("_", 1)[0]
    return same_module and ((s1, s2) in DESIGNED or (s2, s1) in DESIGNED)


def candidate_pairs(m):
    geoms = [i for i in range(m.ngeom)
             if m.geom_group[i] != 2 and m.geom(i).name and m.geom(i).name != "floor"
             and not m.geom(i).name.startswith("bump")]
    pairs = []
    for a, b in itertools.combinations(geoms, 2):
        if m.geom_bodyid[a] == m.geom_bodyid[b]:
            continue
        na, nb = m.geom(a).name, m.geom(b).name
        if designed(na, nb):
            continue
        # weld-equivalent bodies (no joint between them) cannot move relative to each other
        if m.body_weldid[m.geom_bodyid[a]] == m.body_weldid[m.geom_bodyid[b]]:
            continue
        pairs.append((a, b))
    return pairs


def set_phase(m, d, g, p, phi):
    gx, gy, _, _ = gimbal_state(g, phi)
    d.qpos[m.joint(f"{p}gimbal_x").qposadr[0]] = gx
    d.qpos[m.joint(f"{p}gimbal_y").qposadr[0]] = gy
    d.qpos[m.joint(f"{p}crank_spin").qposadr[0]] = phi


# ---- signed-distance clearance (independent of MuJoCo's collision distance routines)
# MuJoCo 3.3.7's mj_geomDistance gave spurious values for some box pairs here (0 with
# native CCD at isolated angles, large negative depths with libccd), so clearance is
# computed from surface samples of one geom against the analytic SDF of the other.
TORUS = None  # (R, r) of the tire mesh, set by the caller


def _sdf(gtype, size, p):
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        q = np.abs(p) - size[:3]
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r, hh = size[0], size[1]
        dxy = np.linalg.norm(p[:, :2], axis=1) - r
        dz = np.abs(p[:, 2]) - hh
        q = np.stack([dxy, dz], axis=1)
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r, hh = size[0], size[1]
        z = np.clip(p[:, 2], -hh, hh)
        return np.linalg.norm(p - np.stack([0*z, 0*z, z], axis=1), axis=1) - r
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:     # tire torus
        R, r = TORUS
        q = np.stack([np.linalg.norm(p[:, :2], axis=1) - R, p[:, 2]], axis=1)
        return np.linalg.norm(q, axis=1) - r
    raise ValueError(gtype)


def _samples(gtype, size, h=0.0015):
    out = []
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        for ax in range(3):
            o = [i for i in range(3) if i != ax]
            n0 = max(2, int(2*size[o[0]]/h) + 1); n1 = max(2, int(2*size[o[1]]/h) + 1)
            u, v = np.meshgrid(np.linspace(-size[o[0]], size[o[0]], n0),
                               np.linspace(-size[o[1]], size[o[1]], n1))
            for sgn in (-1, 1):
                pts = np.zeros((u.size, 3)); pts[:, o[0]] = u.ravel(); pts[:, o[1]] = v.ravel()
                pts[:, ax] = sgn*size[ax]; out.append(pts)
    elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        r, hh = size[0], size[1]
        nt = max(8, int(2*np.pi*r/h)); nz = max(2, int(2*hh/h) + 1)
        t, z = np.meshgrid(np.linspace(0, 2*np.pi, nt, endpoint=False), np.linspace(-hh, hh, nz))
        out.append(np.stack([r*np.cos(t).ravel(), r*np.sin(t).ravel(), z.ravel()], axis=1))
        for sgn in (-1, 1):
            if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                for rr in np.linspace(0, r, max(2, int(r/h) + 1)):
                    tt = np.linspace(0, 2*np.pi, max(4, int(2*np.pi*rr/h)), endpoint=False)
                    out.append(np.stack([rr*np.cos(tt), rr*np.sin(tt), sgn*hh + 0*tt], axis=1))
            else:
                for ph in np.linspace(0, np.pi/2, 6):
                    rr = r*np.cos(ph)
                    tt = np.linspace(0, 2*np.pi, max(4, int(2*np.pi*max(rr, 1e-4)/h)), endpoint=False)
                    out.append(np.stack([rr*np.cos(tt), rr*np.sin(tt),
                                         sgn*(hh + r*np.sin(ph)) + 0*tt], axis=1))
    elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
        R, r = TORUS
        u, v = np.meshgrid(np.linspace(0, 2*np.pi, int(2*np.pi*R/h), endpoint=False),
                           np.linspace(0, 2*np.pi, 24, endpoint=False))
        rr = R + r*np.cos(v)
        out.append(np.stack([(rr*np.cos(u)).ravel(), (rr*np.sin(u)).ravel(),
                             (r*np.sin(v)).ravel()], axis=1))
    return np.concatenate(out)


class Clearance:
    def __init__(self, m, pairs):
        self.m, self.pairs = m, pairs
        ids = sorted({i for pr in pairs for i in pr})
        self.local = {i: _samples(m.geom_type[i], m.geom_size[i]) for i in ids}

    def frame(self, d, gid):
        # mesh geoms are re-framed by MuJoCo (principal axes); the tire is defined in
        # its wheel body frame, so use that frame for it
        if self.m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            b = self.m.geom_bodyid[gid]
            return d.xmat[b].reshape(3, 3), d.xpos[b]
        return d.geom_xmat[gid].reshape(3, 3), d.geom_xpos[gid]

    def dist(self, d, a, b, distmax=0.05):
        m = self.m
        ca, cb = d.geom_xpos[a], d.geom_xpos[b]
        if np.linalg.norm(ca - cb) - m.geom_rbound[a] - m.geom_rbound[b] > distmax:
            return distmax
        best = distmax
        for x, y in ((a, b), (b, a)):
            Rx, px = self.frame(d, x)
            Ry, py = self.frame(d, y)
            world = self.local[x] @ Rx.T + px
            in_y = (world - py) @ Ry
            best = min(best, float(_sdf(m.geom_type[y], m.geom_size[y], in_y).min()))
        return best


def sweep(m, d, pairs, set_fn, samples, distmax=0.05):
    cl = Clearance(m, pairs)
    best = {}
    for x in samples:
        set_fn(x)
        mujoco.mj_forward(m, d)
        for a, b in pairs:
            dist = cl.dist(d, a, b, distmax)
            key = (m.geom(a).name, m.geom(b).name)
            if key not in best or dist < best[key][0]:
                best[key] = (dist, x)
    return best


def module_check(g=None, n=181):
    global TORUS
    g = g or ModuleGeometry()
    TORUS = (g.wheel_R, g.tire_r)
    m = mujoco.MjModel.from_xml_string(build_bench(g)[0])
    d = mujoco.MjData(m)
    pairs = candidate_pairs(m)
    best = sweep(m, d, pairs, lambda phi: set_phase(m, d, g, "m_", phi),
                 np.linspace(0, 2*math.pi, n))
    ranked = sorted(best.items(), key=lambda kv: kv[1][0])
    return m, ranked, len(pairs)


def vehicle_check(n=30, seed=1):
    global TORUS
    v = VehicleGeometry()
    g = v.module
    TORUS = (g.wheel_R, g.tire_r)
    m = mujoco.MjModel.from_xml_string(build_vehicle(v))
    d = mujoco.MjData(m)
    # only pairs between different modules / module vs chassis
    pairs = [(a, b) for a, b in candidate_pairs(m)
             if m.geom(a).name.split("_")[0] != m.geom(b).name.split("_")[0]]
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2*math.pi, size=(n, 4))
    phases[0] = 0

    def set_all(row):
        for p, phi in zip(MODULE_NAMES, row):
            set_phase(m, d, g, f"{p}_", phi)
    best = sweep(m, d, pairs, set_all, phases)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0])
    return ranked, len(pairs)


if __name__ == "__main__":
    m, ranked, npairs = module_check()
    print(f"module: {npairs} pairs; 12 closest:")
    for (a, b), (dist, phi) in ranked[:12]:
        print(f"  {a:18s} {b:18s} {dist*1000:7.2f} mm at phi={math.degrees(phi):6.1f}")
    vr, vn = vehicle_check()
    print(f"vehicle (between modules/chassis): {vn} pairs; 6 closest:")
    for (a, b), (dist, _) in vr[:6]:
        print(f"  {a:18s} {b:18s} {dist*1000:7.2f} mm")
