"""MJCF generator for the 4-wheel nutating-caster vehicle and a single-module bench.

Each module: steering crank (actuated) -> igubal spherical bearing at the crank pin ->
motor housing on a passive 2-axis Cardan gimbal -> wheel on the motor shaft (actuated).
The crank/housing loop is closed with equality/connect (a ball joint, like the
spherical plain bearing it represents).  Only the tire geoms collide; structure geoms
are used for the separate clearance check (check_clearance.py).
"""
from __future__ import annotations

import argparse
from math import cos, pi, radians, sin
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from geometry import ModuleGeometry, VehicleGeometry

SUB = ET.SubElement


def f(x: float) -> str:
    return f"{x:.8g}"


def vec(*v) -> str:
    return " ".join(f(x) for x in v)


def torus_vertices(R: float, r: float, nu: int = 120, nv: int = 20) -> str:
    pts = []
    for i in range(nu):
        u = 2*pi*i/nu
        for j in range(nv):
            v = 2*pi*j/nv
            rr = R + r*cos(v)
            pts.append((rr*cos(u), rr*sin(u), r*sin(v)))
    return " ".join(vec(*p) for p in pts)


def bump_vertices(L_up=0.06, L_top=0.08, h=0.008, w=0.16) -> str:
    """Convex trapezoid prism (bump) along x, centred at x=0, resting on z=0."""
    xs = [(-L_up - L_top/2, 0), (-L_top/2, h), (L_top/2, h), (L_up + L_top/2, 0)]
    pts = [(x, y, z) for (x, z) in xs for y in (-w/2, w/2)]
    return " ".join(vec(*p) for p in pts)


def segment_geom(parent, name, p0, p1, radius, rgba, gtype="capsule", **kw):
    return SUB(parent, "geom", name=name, type=gtype, fromto=vec(*p0, *p1),
               size=f(radius), rgba=rgba, **kw)


def box(parent, name, center, half, rgba, **kw):
    return SUB(parent, "geom", name=name, type="box", pos=vec(*center),
               size=vec(*half), rgba=rgba, **kw)


FIXED_CABLE_RGBA = "1.0 0.5 0.1 1"


def _polyline(parent, pts, radius=0.0025):
    """Visual-only capsules for cable runs that are tied to one body (unnamed: not part of
    the clearance check; massless)."""
    for a, b in zip(pts[:-1], pts[1:]):
        SUB(parent, "geom", type="capsule", fromto=vec(*a, *b), size=f(radius),
            rgba=FIXED_CABLE_RGBA, mass="0", group="1")


# ------------------------------------------------------------------------ module
def add_module(parent, p: str, pos, g: ModuleGeometry, tire_collides=True, cable=None):
    """Add one module whose pivot P is at `pos` in `parent`'s frame.

    Bodies: {p}mount (fixed frame), {p}ring, {p}housing, {p}wheel, {p}crank.
    """
    th, s, c = g.theta, g.s, g.c
    h = g.crank_height
    plate_z = h + g.plate_dz
    mount = SUB(parent, "body", name=f"{p}mount", pos=vec(*pos))
    SUB(mount, "inertial", pos=vec(0, 0, 0.6*plate_z), mass=f(g.m_frame),
        diaginertia="0.0025 0.0025 0.0012")
    SUB(mount, "site", name=f"{p}pivot", pos="0 0 0", size="0.005", rgba="0.9 0.2 0.2 1")
    frame_rgba = "0.55 0.6 0.68 1"
    # Module frame: two posts from the top plate to pivot level, arms to pillow blocks.
    gl = g.cable_layout in ("gimbal", "hook")  # v2 loops (hollow trunnions or v4 brackets)
    gh = g.cable_layout == "hook"              # v4: external hook / bridge, solid trunnions
    gc = g.cable_layout == "clamp"             # v3: clamps next to solid trunnions
    ga = g.cable_layout == "axis"              # v5 candidate: one span up the hollow crank shaft
    gn = g.cable_layout == "hang"              # candidate: slack cable from the stub to the plate
    ay = g.arm_y if (gl or gc) else 0.0
    for sx, side in ((1, "p"), (-1, "n")):
        box(mount, f"{p}post_{side}", (sx*g.post_x, ay, (plate_z - 0.006)/2),
            (0.005, 0.015, (plate_z + 0.006)/2), frame_rgba)
        a_in, a_out = g.bearing_x + 0.008, g.post_x + 0.005
        # gimbal layout: arms run beside the x axis (y < 0) so the +x axis beyond the
        # pillow block is free for the outer cable coil
        box(mount, f"{p}arm_{side}", (sx*(a_in + a_out)/2, ay, g.arm_z),
            ((a_out - a_in)/2, 0.0125 if (gl or gc) else 0.015, 0.006), frame_rgba)
        # iglidur flange bushing in an aluminium pillow block (gimbal x bearing)
        box(mount, f"{p}pillow_{side}", (sx*g.bearing_x, 0, g.arm_z),
            (0.008, 0.014, g.pillow_hz), "0.85 0.75 0.2 1")
    if gh:
        # fixed cable run on the frame (visual only): end of loop B -> clamp arm -> post -> plate
        cx, cz = g.trunnion_x_end + g.loop_pitch, 2*g.loop_R
        _polyline(mount, [(cx, 0, cz), (cx, -0.006, cz + 0.0065),
                          (g.post_x - 0.0075, ay, cz + 0.0065), (g.post_x - 0.0075, ay, plate_z)])
    if gl:
        # frame-side cable clamp of loop B, carried from the +x post
        cx, cz = g.trunnion_x_end + g.loop_pitch, 2*g.loop_R
        box(mount, f"{p}clampF", (cx, -0.006, cz), (0.006, 0.006, 0.006), "0.3 0.3 0.35 1")
        box(mount, f"{p}clampF_arm", ((cx + g.post_x - 0.005)/2, -0.012, cz),
            ((g.post_x - 0.005 - cx)/2, 0.006, 0.004), frame_rgba)
    if gc:
        # frame-side clamp of span B on a lug rising from the +x arm, beyond the pillow
        fx, fy, fz = _clamp_B_end(g)
        box(mount, f"{p}clampF", (fx, fy - 0.0045, fz), (0.005, 0.0035, 0.005), "0.3 0.3 0.35 1")
        box(mount, f"{p}clampF_arm", (fx, -0.0105, (fz + 0.006)/2),
            (0.004, 0.0025, (fz - 0.006)/2 + 0.004), frame_rgba)
    if gn:
        cx, cy, cz = g.hang_clamp
        box(mount, f"{p}clampF", (cx, cy, cz + 0.006), (0.006, 0.006, 0.006), "0.3 0.3 0.35 1")
    if ga:
        # frame-side clamp above the hollow shaft, on an arm from the steering motor post
        _, _, z_top, _ = axis_cable_levels(g)
        ze = z_top + g.top_free_len
        box(mount, f"{p}clampF", (0, 0.009, ze), (0.005, 0.004, 0.006), "0.3 0.3 0.35 1")
        box(mount, f"{p}clampF_arm", (0.0325, 0.009, ze + 0.009), (0.0375, 0.004, 0.003), frame_rgba)
    # Top bearing block for the crank shaft, under the plate.
    SUB(mount, "geom", name=f"{p}topbearing", type="cylinder", pos=vec(0, 0, plate_z - 0.008),
        size="0.024 0.008", rgba="0.85 0.75 0.2 1")
    # Steering gear motor + GT2 belt above the plate (visual only).
    SUB(mount, "geom", name=f"{p}steer_motor", type="cylinder",
        pos=vec(0.065, 0, plate_z + 0.005 + 0.04), size="0.0185 0.035", rgba="0.25 0.25 0.28 1")
    SUB(mount, "geom", name=f"{p}belt", type="box", pos=vec(0.0325, 0, plate_z + 0.012),
        size="0.045 0.012 0.003", rgba="0.1 0.1 0.1 1")

    # ----- outer gimbal ring (Cardan cross, outer member), rotates about x
    ring = SUB(mount, "body", name=f"{p}ring")
    SUB(ring, "joint", name=f"{p}gimbal_x", type="hinge", axis="1 0 0",
        damping="0.0005", armature="1e-6", frictionloss="0.002")
    SUB(ring, "inertial", pos="0 0 0", mass=f(g.m_ring), diaginertia="2e-5 3e-5 4e-5")
    hx, hy, b = g.ring_half_x, (g.ring_half_y_clamp if gc else g.ring_half_y), g.ring_bar
    rr = "0.88 0.27 0.2 1"
    for sx, side in ((1, "p"), (-1, "n")):
        box(ring, f"{p}ringy_{side}", (sx*hx, 0, 0), (b, hy + b, b), rr)
        box(ring, f"{p}ringx_{side}", (0, sx*hy, 0), (hx + b, b, b), rr)
        hollow = gl and not gh and sx > 0
        xend = g.trunnion_x_end if hollow else g.bearing_x + 0.004
        segment_geom(ring, f"{p}trunx_{side}", (sx*(hx + b), 0, 0),
                     (sx*xend, 0, 0), 0.005 if hollow else 0.004, rr, gtype="cylinder")
    if gh:
        # bridge carrying the cable (tied to it) under the +x pillow block to the outer axis
        x0, xe = hx + b, g.trunnion_x_end - 0.004     # bridge hangs from the bar's outer face
        # top of the bridge 9 mm below the pillow: its corner stays clear for +-45 deg
        zt, zb = -g.pillow_hz - 0.011, -g.pillow_hz - 0.015
        brc = "0.2 0.72 0.35 1"
        box(ring, f"{p}bridge_d", (x0 + 0.0015, 0, (-b + zb)/2), (0.0015, 0.004, (zb + b)/-2), brc)
        box(ring, f"{p}bridge_h", ((x0 + xe)/2, 0, (zt + zb)/2), ((xe - x0)/2, 0.004, 0.002), brc)
        box(ring, f"{p}bridge_u", (xe - 0.003, 0, (zb + 0.004)/2), (0.003, 0.004, (0.004 - zb)/2), brc)
        # fixed cable run along the ring (visual only): clamp of loop A -> strut -> under the
        # +y and +x bars -> down the bridge -> along its top -> up to the start of loop B
        cxr, cyr = -2*g.loop_R, g.trunnion_y_end + g.loop_pitch
        _polyline(ring, [(cxr, cyr, -0.0175), (-hx - b - 0.004, hy + b - 0.002, -0.0175),
                         (-hx + b, hy, -b - 0.0025), (hx - b, hy, -b - 0.0025),
                         (hx - b + 0.0015, 0, -b - 0.0025), (x0 - 0.0025, 0, -b - 0.0025),
                         (x0 - 0.0025, 0, zt + 0.0025), (xe - 0.0085, 0, zt + 0.0025),
                         (xe - 0.0085, 0, 0.0065), (g.trunnion_x_end, 0, 0)])
        box(ring, f"{p}clampB0", (xe + 0.002, 0, 0), (0.002, 0.004, 0.004), "0.3 0.3 0.35 1")
    if gl:
        # ring-side cable clamp of loop A (at x = -2R, beside the +y trunnion)
        cxr, cyr = -2*g.loop_R, g.trunnion_y_end + g.loop_pitch
        box(ring, f"{p}clampR", (cxr, cyr, -0.009), (0.006, 0.006, 0.009), "0.3 0.3 0.35 1")
        box(ring, f"{p}ringstrut", ((cxr - hx - b)/2, (cyr + hy)/2, -0.024 if gh else -0.020),
            (abs(cxr + hx + b)/2 + 0.004, (cyr - hy)/2, 0.004), rr)

    if gc:
        # ring side: lug carrying the end of span A outside the -x bar; bridge under the
        # +x pillow block carrying the cable (fixed to the ring) to the start of span B
        ax_, ay_, az_ = _clamp_A_end(g)
        box(ring, f"{p}clampR", (ax_, ay_, az_ - 0.005), (0.005, 0.005, 0.005), "0.3 0.3 0.35 1")
        box(ring, f"{p}ringlug", ((ax_ - hx - b)/2, ay_ + 0.001, az_ - 0.012),
            (abs(ax_ + hx + b)/2 + 0.004, 0.004, 0.003), rr)
        bx_, by_, bz_ = clamp_B_start(g)
        x0 = hx + b
        box(ring, f"{p}bridge_d", (x0 + 0.004, 0, -0.016), (0.004, 0.005, 0.012), rr)
        box(ring, f"{p}bridge_h", ((x0 + bx_)/2 + 0.002, 0, -0.024), ((bx_ - x0)/2 + 0.002, 0.005, 0.004), rr)
        box(ring, f"{p}bridge_u", (bx_, 0, -0.017), (0.004, 0.005, 0.007), rr)
        box(ring, f"{p}clampRB", (bx_, by_ - 0.006, bz_), (0.005, 0.0035, 0.005), "0.3 0.3 0.35 1")

    # ----- motor housing: local z = motor axis a
    hs = SUB(ring, "body", name=f"{p}housing", euler=vec(0, th, 0))
    SUB(hs, "joint", name=f"{p}gimbal_y", type="hinge", axis="0 1 0", ref=f(th),
        damping="0.0005", armature="1e-6", frictionloss="0.002")
    SUB(hs, "inertial", pos="0 0 0.035", mass=f(g.m_housing), diaginertia="3.2e-4 3.2e-4 5e-5")
    SUB(hs, "geom", name=f"{p}motor", type="cylinder",
        pos=vec(0, 0, (g.motor_a0 + g.motor_a1)/2),
        size=vec(g.motor_radius, (g.motor_a1 - g.motor_a0)/2), rgba="0.96 0.6 0.15 1")
    segment_geom(hs, f"{p}tube", (0, 0, g.tube_a0), (0, 0, g.motor_a0), g.tube_radius,
                 "0.7 0.72 0.76 1", gtype="cylinder")
    segment_geom(hs, f"{p}stub", (0, 0, g.motor_a1), (0, 0, g.stub_a1), g.stub_radius,
                 "0.75 0.78 0.82 1", gtype="cylinder")
    for sy, side in ((1, "p"), (-1, "n")):
        hollow = gl and not gh and sy > 0
        yend = g.trunnion_y_end if hollow else hy - b
        segment_geom(hs, f"{p}truny_{side}", (0, sy*g.tube_radius, 0),
                     (0, sy*yend, 0), 0.005 if hollow else 0.004, "0.7 0.72 0.76 1",
                     gtype="cylinder")
    if gh:
        # hook on the housing: from the tube over the ring bar (16 mm above the inner axis,
        # clear of the bar for +-45 deg) and down onto the inner axis outside the bar
        ye = g.trunnion_y_end - 0.006
        box(hs, f"{p}hookH_h", (0, (g.tube_radius + ye)/2, 0.017),
            (0.004, (ye - g.tube_radius)/2, 0.003), "0.2 0.72 0.35 1")
        box(hs, f"{p}hookH_v", (0, ye - 0.003, 0.0075), (0.004, 0.003, 0.0125), "0.2 0.72 0.35 1")
        # fixed cable run on the housing (visual only): motor rear -> along the motor ->
        # over the hook -> down its outer face -> start of loop A on the inner axis
        _polyline(hs, [(0, g.motor_radius + 0.0025, g.motor_a1 - 0.003),
                       (0, g.motor_radius + 0.0025, 0.0225), (0, ye + 0.0025, 0.0225),
                       (0, ye + 0.0025, 0.003), (0, g.trunnion_y_end, 0)])
        box(hs, f"{p}clampA0", (0, ye + 0.003, -0.006), (0.004, 0.003, 0.004), "0.3 0.3 0.35 1")
    if gn and g.hang_exit in ("side_y", "side_y_ax"):
        # clamp on the +y (trunnion-side) face of the motor, cable leaving along +y
        yr = g.motor_radius + 0.0025
        _polyline(hs, [(0, 0.010, g.motor_a1 + 0.003), (0, yr, g.motor_a1 + 0.003), (0, yr, g.exit_a)])
        box(hs, f"{p}clampH", (0, g.motor_radius + 0.003, g.exit_a), (0.005, 0.003, 0.006),
            "0.3 0.3 0.35 1")
    if ga or (gn and g.hang_exit == "side"):
        # fixed run on the housing (visual only): motor rear -> along the inner side of the
        # motor -> clamp on the inner side at exit_a, cable leaving radially inwards
        xr = -(g.motor_radius + 0.0025)
        _polyline(hs, [(-0.010, 0, g.motor_a1 + 0.003), (xr, 0, g.motor_a1 + 0.003),
                       (xr, 0, g.exit_a)])
        box(hs, f"{p}clampH", (-(g.motor_radius + 0.003), 0, g.exit_a), (0.003, 0.005, 0.006),
            "0.3 0.3 0.35 1")
    if gc:
        # housing-side clamp of span A: on the tube beside the motor, above the +y trunnion
        cxa, cya, cza = clamp_A_start(g)
        box(hs, f"{p}clampH", (cxa, cya - 0.004, cza - 0.004), (0.005, 0.004, 0.004),
            "0.3 0.3 0.35 1")
    # stripe: makes the (non-)rotation of the stator visible
    segment_geom(hs, f"{p}stripe", (g.motor_radius, 0, g.motor_a0 + 0.01),
                 (g.motor_radius, 0, g.motor_a1 - 0.01), 0.003, "0.1 0.2 0.35 1", group="2")
    SUB(hs, "site", name=f"{p}rear_pin", pos=vec(0, 0, g.rear_length), size="0.004")
    SUB(hs, "site", name=f"{p}cable_exit", pos=vec(0, 0, g.stub_a1), size="0.003",
        rgba="0.6 0.3 0.9 1")

    # ----- wheel on the motor shaft
    wh = SUB(hs, "body", name=f"{p}wheel", pos=vec(0, 0, -g.wheel_offset))
    SUB(wh, "joint", name=f"{p}wheel_spin", type="hinge", axis="0 0 1",
        damping="0.002", armature="2e-4", frictionloss="0.01")
    SUB(wh, "inertial", pos="0 0 0", mass=f(g.m_wheel), diaginertia="1.0e-4 1.0e-4 1.9e-4")
    SUB(wh, "geom", name=f"{p}tire", type="mesh", mesh="tire",
        rgba="0.12 0.14 0.16 1",
        contype="1" if tire_collides else "0", conaffinity="1" if tire_collides else "0",
        condim="4", friction="0.9 0.002 0.0001", solref="0.015 1",
        solimp="0.95 0.99 0.0005", margin="0.001", gap="0.001")
    SUB(wh, "geom", name=f"{p}hub", type="cylinder", size=vec(g.wheel_R - 0.004, 0.0065),
        rgba="0.2 0.55 0.8 1")
    for k in range(3):
        u = 2*pi*k/3
        segment_geom(wh, f"{p}spoke{k}", (0, 0, 0.0075),
                     (0.045*cos(u), 0.045*sin(u), 0.0075), 0.003, "0.9 0.9 0.95 1",
                     group="2")

    # ----- steering crank
    ck = SUB(mount, "body", name=f"{p}crank", pos=vec(0, 0, h))
    SUB(ck, "joint", name=f"{p}crank_spin", type="hinge", axis="0 0 1",
        damping="0.002", armature="3e-4", frictionloss="0.02")
    SUB(ck, "inertial", pos=vec(0.5*g.crank_radius, 0, 0.02), mass=f(g.m_crank),
        diaginertia="2e-5 1.5e-4 1.5e-4")
    cr = "0.65 0.38 0.87 1"
    dz = g.crank_arm_dz
    x_in = 0.012 if ga else -0.005                 # v5: the arm starts outside the hub bore
    box(ck, f"{p}crank_arm", ((x_in + g.crank_radius + 0.010)/2, 0, dz),
        ((g.crank_radius + 0.010 - x_in)/2, 0.010, 0.004), cr)
    SUB(ck, "geom", name=f"{p}crank_hub", type="cylinder", pos=vec(0, 0, dz),
        size="0.016 0.008", rgba=cr)
    # hollow shaft (OD 20) from the hub up through the top bearing; side slot for the cable
    segment_geom(ck, f"{p}crank_shaft", (0, 0, dz + 0.008), (0, 0, g.plate_dz + 0.004),
                 0.010, cr, gtype="cylinder")
    # igubal flange bearing (EFOM-10 class) on the crank pin, bore along the motor axis
    pin = np.array([g.crank_radius, 0, 0])
    box(ck, f"{p}pin_bearing", pin, (0.013, 0.026, 0.006), "0.85 0.75 0.2 1",
        euler=vec(0, th, 0))
    # two bracket legs at the flange holes (y = +-0.020) leave the centre free for the cable
    top = pin + 0.013*np.array([-c, 0, s]) + 0.006*np.array([s, 0, c])
    for sy, side in ((1, "p"), (-1, "n")):
        segment_geom(ck, f"{p}bracket_{side}", (top[0], sy*0.020, top[2] - 0.004),
                     (top[0], sy*0.020, dz - 0.004), 0.004, cr)
    SUB(ck, "site", name=f"{p}crank_pin", pos=vec(*pin), size="0.004")
    return mount


# ------------------------------------------------------------------------ cable route
def _arc(C, R, a0, a1, n):
    return [C + R*np.array([cos(a), 0, sin(a)]) for a in np.linspace(a0, a1, n)[1:]]


def cable_route(g: ModuleGeometry, n_seg: int = 64):
    """Polyline (mount frame, phi = 0) from the stub exit to an anchor above the plate.

    route 'over' : out along the motor axis, 135 deg bend back over the crank arm,
                   horizontal to the shaft slot, 90 deg bend up the hollow shaft.
    route 'diag' : 90 deg bend from the axis direction to an up-and-in diagonal,
                   straight to the shaft axis, 45 deg bend into the hollow shaft
                   (less total bending; needs a taller module, see plate_dz).
    Returns points, index of the node at the shaft entry, index at the plate top, length.
    """
    s, c = g.s, g.c
    a = np.array([s, 0, c])
    E = g.stub_a1*a
    Rb = g.cable_bend_R
    nrm = np.array([-c, 0, s])                       # left normal of a in the x-z plane
    C1 = E + Rb*nrm
    h = g.crank_height
    pts = [E]
    start = np.arctan2(-nrm[2], -nrm[0])             # angle of E about C1
    if g.cable_route == "over":
        pts += _arc(C1, Rb, start, start + 3*pi/4, 13)
        z_run, x_end = pts[-1][2], pts[-1][0]
        pts += [np.array([x, 0, z_run]) for x in np.linspace(x_end, Rb, 8)[1:]]
        C2 = np.array([Rb, 0, z_run + Rb])
        pts += _arc(C2, Rb, -pi/2, -pi, 9)
    else:
        pts += _arc(C1, Rb, start, start + pi/2, 11)      # tangent now (-s, 0, c)
        P1 = pts[-1]
        d1 = np.array([-c, 0, s])
        # straight run until the 45 deg bend lands on the axis
        x_bend = Rb*(1 - cos(pi/4))
        L = (P1[0] - x_bend)/c
        pts += [P1 + d1*u for u in np.linspace(0, L, 10)[1:]]
        P2 = pts[-1]
        C2 = P2 + Rb*np.array([s, 0, c])                 # centre to the right of travel
        a0 = np.arctan2(P2[2] - C2[2], P2[0] - C2[0])
        pts += _arc(C2, Rb, a0, a0 - pi/4, 6)
    slot = len(pts) - 1
    top = max(h + g.plate_dz + 0.012, pts[-1][2] + 0.01)
    pts += [np.array([0.0, 0, zz]) for zz in np.linspace(pts[-1][2], top + g.cable_free_len, 30)[1:]]
    plate_node = min(range(slot, len(pts)), key=lambda i: abs(pts[i][2] - top))
    pts = np.array(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    sarc = np.concatenate([[0], np.cumsum(seg)])
    target = np.linspace(0, sarc[-1], n_seg + 1)
    res = np.column_stack([np.interp(target, sarc, pts[:, i]) for i in range(3)])
    slot_i = int(np.argmin(np.abs(target - sarc[slot])))
    plate_i = int(np.argmin(np.abs(target - sarc[plate_node])))
    return res, slot_i, plate_i, float(sarc[-1])


def add_cable(root, world_parent, p: str, g: ModuleGeometry, attach: str,
              armature=2e-6, damping=2e-4, twist="4e6", bend="1.2e7"):
    """Elastic cable (MuJoCo cable plugin) from the stub exit to the frame.

    attach = 'housing' (this design) or 'crank' (a conventional rotating-motor caster).
    """
    pts, slot_i, plate_i, length = cable_route(g)
    th = g.theta
    # composite vertices are given in the parent body frame
    if attach == "housing":
        parent = world_parent.find(f".//body[@name='{p}housing']")
        Rm = np.array([[cos(th), 0, sin(th)], [0, 1, 0], [-sin(th), 0, cos(th)]])  # housing->mount
        local = (pts) @ Rm                 # mount -> housing: R^T x  (row form: x R)
    else:
        parent = world_parent.find(f".//body[@name='{p}crank']")
        local = pts - np.array([0, 0, g.crank_height])
    comp = SUB(parent, "composite", prefix=f"{p}cab", type="cable", initial="none",
               vertex=" ".join(vec(*q) for q in local))
    plug = SUB(comp, "plugin", plugin="mujoco.elasticity.cable")
    # PUR jacketed 5 mm cable, order-of-magnitude moduli (Pa)
    SUB(plug, "config", key="twist", value=twist)
    SUB(plug, "config", key="bend", value=bend)
    SUB(plug, "config", key="vmax", value="0")
    # stress-free shape = straight (a real cable is not pre-curved to its route)
    SUB(plug, "config", key="flat", value="true")
    SUB(comp, "joint", kind="main", damping=f(damping), armature=f(armature))
    SUB(comp, "geom", type="capsule", size=f(g.cable_radius), rgba="0.15 0.15 0.9 1",
        density="1500", contype="0", conaffinity="0", group="3")
    eq = root.find("equality")
    n = len(pts) - 1
    # slot node: on the shaft axis (connect = position only: the shaft turns freely around it)
    SUB(eq, "connect", name=f"{p}cab_slot", body1=f"{p}cabB_{slot_i}", body2=f"{p}crank",
        anchor="0 0 0", solref="0.004 1")
    SUB(eq, "connect", name=f"{p}cab_plate", body1=f"{p}cabB_{plate_i}", body2=f"{p}mount",
        anchor="0 0 0", solref="0.004 1")
    SUB(eq, "weld", name=f"{p}cab_anchor", body1=f"{p}cabB_last", body2=f"{p}mount",
        solref="0.004 1")
    return {"n_seg": n, "length_m": length, "slot_index": slot_i, "plate_index": plate_i}


def axis_cable_levels(g: ModuleGeometry):
    """z (mount frame) of: hub bottom, bore entry node, shaft top node, frame clamp."""
    h = g.crank_height
    z_hub = h + g.crank_arm_dz - 0.008
    z_entry = z_hub - 0.004
    z_top = h + g.plate_dz + 0.004 + 0.004
    return z_hub, z_entry, z_top, z_top + g.top_free_len


def axis_cable_route(g: ModuleGeometry, n_seg: int = 64):
    """v5 candidate: polyline (mount frame, phi = 0) from the clamp on the inner side of
    the motor to the frame clamp above the hollow crank shaft.  The free span under the
    crank arm is a Hermite curve from the clamp (leaving radially inwards, i.e. 45 deg
    up-and-in) to the bore entry on the crank axis (vertical).  Returns points, index of the
    bore-entry node, index of the shaft-top node, length."""
    s, c = g.s, g.c
    a = np.array([s, 0, c])
    xl = np.array([c, 0, -s])                        # housing local x in the mount frame
    E = g.exit_a*a - (g.motor_radius + g.cable_radius + g.exit_clear)*xl
    d0 = -xl
    _, z_entry, z_top, z_end = axis_cable_levels(g)
    B0 = np.array([0.0, 0, z_entry])
    d1 = np.array([0.0, 0, 1])
    L = np.linalg.norm(B0 - E)
    pts = []
    for t in np.linspace(0, 1, 25):
        h00, h10, h01, h11 = 2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t, -2*t**3 + 3*t**2, t**3 - t**2
        pts.append(h00*E + h10*L*d0 + h01*B0 + h11*L*d1)
    entry = len(pts) - 1
    pts += [np.array([0.0, 0, z]) for z in np.linspace(z_entry, z_top, 12)[1:]]
    top = len(pts) - 1
    pts += [np.array([0.0, 0, z]) for z in np.linspace(z_top, z_end, 20)[1:]]
    pts = np.array(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    sarc = np.concatenate([[0], np.cumsum(seg)])
    target = np.linspace(0, sarc[-1], n_seg + 1)
    res = np.column_stack([np.interp(target, sarc, pts[:, i]) for i in range(3)])
    entry_i = int(np.argmin(np.abs(target - sarc[entry])))
    top_i = int(np.argmin(np.abs(target - sarc[top])))
    return res, entry_i, top_i, float(sarc[-1])


def add_cable_axis(root, world_parent, p: str, g: ModuleGeometry, armature=2e-6,
                   damping=2e-4, twist="4e6", bend="1.2e7"):
    """v5 candidate: one elastic span, housing clamp -> under the crank arm -> up the
    hollow crank shaft -> frame clamp.  Inside the bore the cable is held on the axis at
    the entry and at the shaft top (position only: free to turn and bend there)."""
    pts, entry_i, top_i, length = axis_cable_route(g)
    th = g.theta
    parent = world_parent.find(f".//body[@name='{p}housing']")
    Rm = np.array([[cos(th), 0, sin(th)], [0, 1, 0], [-sin(th), 0, cos(th)]])
    _composite(parent, f"{p}cab", pts @ Rm, g, armature, damping, twist, bend)
    eq = root.find("equality")
    for nm, i in (("entry", entry_i), ("top", top_i)):
        SUB(eq, "connect", name=f"{p}cab_{nm}", body1=f"{p}cabB_{i}", body2=f"{p}mount",
            anchor="0 0 0", solref="0.004 1")
    SUB(eq, "weld", name=f"{p}cab_anchor", body1=f"{p}cabB_last", body2=f"{p}mount",
        solref="0.004 1")
    return {"n_seg": len(pts) - 1, "length_m": length, "entry_index": entry_i,
            "top_index": top_i}


def hang_cable_route(g: ModuleGeometry, n_seg: int = 64):
    """Candidate 'hang': polyline (mount frame, phi = 0) from the stub end, leaving along
    the motor axis, sagging to the frame clamp; the sag depth is chosen so that the arc
    length equals hang_len."""
    s, c = g.s, g.c
    a = np.array([s, 0, c])
    C = np.array(g.hang_clamp, dtype=float)
    if g.hang_exit == "side":
        xl = np.array([c, 0, -s])
        E = g.exit_a*a - (g.motor_radius + g.cable_radius + g.exit_clear)*xl
        P1 = E - 0.01*xl
    elif g.hang_exit in ("side_y", "side_y_ax"):
        yv = np.array([0, 1.0, 0])
        E = g.exit_a*a + (g.motor_radius + g.cable_radius + g.exit_clear)*yv
        # 'side_y': cable leaves along +y (parallel to the inner gimbal axis: its swing is
        # torsion).  'side_y_ax': leaves along the motor axis, rearwards (the swing bends it).
        ang = radians(90.0 if g.hang_exit == "side_y_ax" else g.exit_angle_deg)
        P1 = E + 0.01*(cos(ang)*yv + sin(ang)*a)
    else:
        E = g.stub_a1*a
        P1 = E + 0.02*a
    def curve(depth):
        mid = (P1 + C)/2 - np.array([0, 0, depth])
        pts = [E] + [(1 - t)**2*P1 + 2*(1 - t)*t*mid + t*t*C for t in np.linspace(0, 1, 60)]
        pts = np.array(pts)
        return pts, float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    lo, hi = 0.0, 0.5
    for _ in range(40):
        d = (lo + hi)/2
        pts, L = curve(d)
        lo, hi = (d, hi) if L < g.hang_len else (lo, d)
    pts, L = curve((lo + hi)/2)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    sarc = np.concatenate([[0], np.cumsum(seg)])
    target = np.linspace(0, sarc[-1], n_seg + 1)
    res = np.column_stack([np.interp(target, sarc, pts[:, i]) for i in range(3)])
    return res, float(sarc[-1])


def add_cable_hang(root, world_parent, p: str, g: ModuleGeometry, armature=2e-6,
                   damping=2e-4, twist="4e6", bend="1.2e7"):
    """Candidate 'hang': one slack elastic span, stub end -> frame clamp under the plate."""
    pts, length = hang_cable_route(g)
    th = g.theta
    parent = world_parent.find(f".//body[@name='{p}housing']")
    Rm = np.array([[cos(th), 0, sin(th)], [0, 1, 0], [-sin(th), 0, cos(th)]])
    _composite(parent, f"{p}cab", pts @ Rm, g, armature, damping, twist, bend)
    eq = root.find("equality")
    SUB(eq, "weld", name=f"{p}cab_anchor", body1=f"{p}cabB_last", body2=f"{p}mount",
        solref="0.004 1")
    return {"n_seg": len(pts) - 1, "length_m": length}


def _ry(a):
    return np.array([[cos(a), 0, sin(a)], [0, 1, 0], [-sin(a), 0, cos(a)]])


def u_loop(O, e_t, e_n, axis, R, leg1, leg2, pitch, n):
    """U loop leaving O along e_t, turning 180 deg towards e_n (radius R), returning
    parallel to the first leg; it advances `pitch` along `axis` from start to end."""
    O, e_t, e_n, axis = map(np.asarray, (O, e_t, e_n, axis))
    L = leg1 + pi*R + leg2
    sarc = np.linspace(0, L, n + 1)
    pts = []
    for sv in sarc:
        if sv <= leg1:
            q = sv*e_t
        elif sv <= leg1 + pi*R:
            a = (sv - leg1)/R
            q = leg1*e_t + R*(np.sin(a)*e_t + (1 - np.cos(a))*e_n)
        else:
            q = leg1*e_t + 2*R*e_n - (sv - leg1 - pi*R)*e_t
        pts.append(O + q + pitch*(sv/L)*axis)
    return np.array(pts), sarc/L


def gimbal_cable_routes(g: ModuleGeometry, n=40):
    """The two free cable spans, in the frames of their parent bodies at qpos0 (phi = 0).

    A: housing (+y trunnion end, on the inner axis) -> ring clamp; U loop in the plane
       normal to the inner axis, up and over towards -x.  Designed symmetric at gy = 0;
       at qpos0 the housing is turned by gy = theta, so the housing end is pre-rotated by
       theta and the rotation is blended out along the loop (rotation about the axis,
       which passes through the clamp, so both clamp positions are unchanged).
    B: ring (+x trunnion end, on the outer axis) -> frame clamp; U loop in the plane
       normal to the outer axis, out along +y and back over at z = 2R.  Neutral at qpos0.
    Only the clamp on the rotating body has to sit on the axis.
    """
    th = g.theta
    A, u = u_loop((0, g.trunnion_y_end, 0), (0, 0, 1), (-1, 0, 0), (0, 1, 0),
                  g.loop_R, g.loop_leg, g.loop_leg, g.loop_pitch, n)
    w = u*u*(3 - 2*u)
    A_ring = np.array([_ry(th*(1 - wi)) @ p for p, wi in zip(A, w)])
    A_housing = A_ring @ _ry(th)                         # ring -> housing (R^T x, row form)
    B, _ = u_loop((g.trunnion_x_end, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0),
                  g.loop_R, g.loop_leg_B, g.loop_leg_B, g.loop_pitch, n)
    return {"A": A_housing, "B": B}


# ------------------------------------------------------------- v3 clamp layout routes
def clamp_A_start(g):
    # housing frame: in the slot between the motor (r 18.5) and the ring bar, above the
    # +y trunnion, 12 mm from the inner axis
    return (0.0, 0.024, 0.012)


def clamp_B_start(g):
    # ring frame: end of the bridge that runs under the pillow block, 6 mm below the outer axis
    return (g.bearing_x + 0.008 + 0.014, 0.0, -0.006)


def _turtle(p0, t0, ops, step=0.0005):
    """Polyline from straight segments ('s', L) and arcs ('a', R, deg, k): the tangent
    turns about the unit vector k (right-hand) at radius R."""
    p, t = np.array(p0, float), np.array(t0, float)
    pts = [p.copy()]
    for op in ops:
        if op[0] == "s":
            nstep = max(1, int(op[1]/step))
            for _ in range(nstep):
                p = p + t*op[1]/nstep
                pts.append(p.copy())
        else:
            _, R, deg, k = op
            k = np.asarray(k, float)
            L = R*np.radians(deg)
            nstep = max(1, int(L/step))
            dth = np.radians(deg)/nstep
            for _ in range(nstep):
                t_mid = t*np.cos(dth/2) + np.cross(k, t)*np.sin(dth/2)
                p = p + t_mid*L/nstep
                t = t*np.cos(dth) + np.cross(k, t)*np.sin(dth) + k*np.dot(k, t)*(1 - np.cos(dth))
                pts.append(p.copy())
    return np.array(pts)


def _resample(pts, n):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    tgt = np.linspace(0, s[-1], n + 1)
    return np.column_stack([np.interp(tgt, s, pts[:, i]) for i in range(3)]), tgt/s[-1]


def clamp_cable_routes(g: ModuleGeometry, n=40):
    """v3 free spans in the neutral pose; each lies in the plane normal to its hinge axis
    and leaves its rotating clamp perpendicular to the axis, so the +-45 deg swing is
    in-plane bending of the loop (no torsion), as in v2 but without hollow trunnions.

    A: housing clamp beside the motor (+z), loop in the x-z plane at y = 24 mm (inside the
       ring slot), U turn towards -x, down to a lug on the ring outside the -x bar.
       Pre-rotated to the qpos0 pose (gy = theta) as in v2.
    B: clamp at the end of the ring bridge beyond the pillow (+y), loop in the y-z plane,
       U turn upwards, back (-y) to a lug on the frame arm.
    """
    th, R = g.theta, g.arch_R
    A = _turtle(clamp_A_start(g), (0, 0, 1),
                [("s", 0.015), ("a", R, 180, (0, -1, 0)), ("s", 0.015)])
    A, u = _resample(A, n)
    w = u*u*(3 - 2*u)
    A_ring = np.array([_ry(th*(1 - wi)) @ q for q, wi in zip(A, w)])
    A_housing = A_ring @ _ry(th)
    B = _turtle(clamp_B_start(g), (0, 1, 0),
                [("s", 0.010), ("a", g.loop_R_B3, 180, (1, 0, 0)), ("s", 0.010)])
    B, _ = _resample(B, n)
    return {"A": A_housing, "B": B, "A_ring_neutral": A}


def _clamp_A_end(g):
    return tuple(clamp_cable_routes(g)["A_ring_neutral"][-1])


def _clamp_B_end(g):
    return tuple(clamp_cable_routes(g)["B"][-1])


def _composite(parent, prefix, pts, g, armature, damping, twist, bend):
    comp = SUB(parent, "composite", prefix=prefix, type="cable", initial="none",
               vertex=" ".join(vec(*q) for q in pts))
    plug = SUB(comp, "plugin", plugin="mujoco.elasticity.cable")
    SUB(plug, "config", key="twist", value=twist)
    SUB(plug, "config", key="bend", value=bend)
    SUB(plug, "config", key="vmax", value="0")
    SUB(plug, "config", key="flat", value="true")        # stress-free shape: straight
    SUB(comp, "joint", kind="main", damping=f(damping), armature=f(armature))
    SUB(comp, "geom", type="capsule", size=f(g.cable_radius), rgba="0.15 0.15 0.9 1",
        density="1500", contype="0", conaffinity="0", group="3")


def add_cable_gimbal(root, world_parent, p, g, armature=2e-6, damping=2e-4, twist="4e6",
                     bend="1.2e7"):
    """Free cable spans of the gimbal layout: loop A (housing -> ring) and loop B
    (ring -> frame).  Inside the hollow trunnions and along the ring channel the cable is
    fixed to its body and is not simulated."""
    r = clamp_cable_routes(g) if g.cable_layout == "clamp" else gimbal_cable_routes(g)  # v2 & v4
    hs = world_parent.find(f".//body[@name='{p}housing']")
    ring = world_parent.find(f".//body[@name='{p}ring']")
    _composite(hs, f"{p}cabA", r["A"], g, armature, damping, twist, bend)
    _composite(ring, f"{p}cabB", r["B"], g, armature, damping, twist, bend)
    eq = root.find("equality")
    SUB(eq, "weld", name=f"{p}cabA_clamp", body1=f"{p}cabAB_last", body2=f"{p}ring",
        solref="0.004 1")
    SUB(eq, "weld", name=f"{p}cabB_clamp", body1=f"{p}cabBB_last", body2=f"{p}mount",
        solref="0.004 1")
    L = lambda q: float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1)))
    return {"spans": {"A": {"n_seg": len(r["A"]) - 1, "length_m": L(r["A"])},
                      "B": {"n_seg": len(r["B"]) - 1, "length_m": L(r["B"])}},
            "length_m": L(r["A"]) + L(r["B"])}


# ------------------------------------------------------------------------ models
def base_root(name, dt, gravity=True):
    root = ET.Element("mujoco", model=name)
    SUB(root, "compiler", angle="radian", autolimits="true")
    SUB(root, "option", timestep=f(dt), gravity="0 0 -9.81" if gravity else "0 0 0",
        integrator="implicitfast", solver="Newton", iterations="100", tolerance="1e-10",
        cone="elliptic", impratio="3")
    vis = SUB(root, "visual")
    SUB(vis, "global", offwidth="1280", offheight="720")
    SUB(vis, "headlight", diffuse="0.75 0.75 0.75", ambient="0.35 0.35 0.35")
    SUB(vis, "quality", shadowsize="4096")
    df = SUB(root, "default")
    SUB(df, "geom", contype="0", conaffinity="0")
    SUB(df, "site", rgba="0.2 0.9 0.3 1")
    return root


def add_assets(root, g: ModuleGeometry, bumps=False):
    a = SUB(root, "asset")
    SUB(a, "texture", name="grid", type="2d", builtin="checker", rgb1="0.82 0.84 0.86",
        rgb2="0.72 0.75 0.78", width="512", height="512", mark="edge", markrgb="0.5 0.52 0.55")
    SUB(a, "material", name="floor", texture="grid", texrepeat="10 10", reflectance="0.05")
    SUB(a, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.95 0.96 0.98",
        rgb2="0.75 0.8 0.86", width="256", height="256")
    SUB(a, "mesh", name="tire", vertex=torus_vertices(g.wheel_R, g.tire_r))
    if bumps:
        SUB(a, "mesh", name="bump", vertex=bump_vertices())


def add_floor(world, size=4.0):
    SUB(world, "light", pos="0 -1.5 3", dir="0 0.4 -1", diffuse="0.6 0.6 0.6",
        castshadow="true")
    SUB(world, "geom", name="floor", type="plane", size=vec(size, size, 0.1), material="floor",
        contype="1", conaffinity="1", condim="4", friction="0.9 0.002 0.0001",
        solref="0.015 1", solimp="0.95 0.99 0.0005", margin="0.001", gap="0.001")


MODULE_NAMES = ("fl", "fr", "rl", "rr")


def build_vehicle(v: VehicleGeometry | None = None, dt=0.001, bumps=None, cable=False):
    """bumps: list of (x, y) centres of 8 mm trapezoid bumps on the floor."""
    v = v or VehicleGeometry()
    g = v.module
    root = base_root("nutating_swerve_quad", dt)
    if cable:
        ext = SUB(root, "extension")
        SUB(ext, "plugin", plugin="mujoco.elasticity.cable")
    add_assets(root, g, bumps=bool(bumps))
    world = SUB(root, "worldbody")
    add_floor(world)
    for k, (bx, by) in enumerate(bumps or []):
        SUB(world, "geom", name=f"bump{k}", type="mesh", mesh="bump", pos=vec(bx, by, 0),
            rgba="0.8 0.5 0.2 1", contype="1", conaffinity="1", condim="4",
            friction="0.9 0.002 0.0001", solref="0.015 1", solimp="0.95 0.99 0.0005",
            margin="0.001", gap="0.001")
    SUB(root, "equality")
    zp = g.pivot_height + 0.0005
    plate_z = g.crank_height + g.plate_dz
    ch = SUB(world, "body", name="chassis", pos=vec(0, 0, zp))
    SUB(ch, "freejoint", name="root")
    SUB(ch, "inertial", pos=vec(-0.02, 0, plate_z + 0.03), mass=f(v.m_chassis),
        diaginertia="0.045 0.065 0.10")
    px, py, pz = v.plate_half
    split = 0.075
    if v.rocker:
        box(ch, "plate", ((-px + split)/2 - 0.005, 0, plate_z + pz),
            ((px + split)/2 - 0.005, py, pz), "0.35 0.42 0.52 0.45")
    else:
        box(ch, "plate", (0, 0, plate_z + pz), (px, py, pz), "0.35 0.42 0.52 0.45")
    box(ch, "battery", (-0.04, 0, plate_z + 2*pz + 0.025), (0.07, 0.04, 0.025),
        "0.2 0.22 0.25 1")
    SUB(ch, "site", name="body_center", pos="0 0 0", size="0.006")
    SUB(ch, "camera", name="chase", pos="-1.1 -1.1 0.9", xyaxes="0.707 -0.707 0 0.35 0.35 0.87",
        mode="trackcom")
    pos = v.module_positions()
    if v.rocker:
        rk = SUB(ch, "body", name="rocker", pos=vec(v.half_length, 0, 0))
        SUB(rk, "joint", name="rocker", type="hinge", axis="1 0 0", pos=vec(0, 0, plate_z),
            range=vec(-np.radians(v.rocker_limit_deg), np.radians(v.rocker_limit_deg)),
            damping="0.5", armature="0.001")
        SUB(rk, "inertial", pos=vec(0, 0, plate_z), mass="0.35", diaginertia="0.006 0.001 0.006")
        box(rk, "front_plate", ((px - split)/2 + split - v.half_length + 0.005 - 0.0, 0, plate_z + pz),
            ((px - split)/2 - 0.005, py, pz), "0.42 0.48 0.58 0.45")
        box(rk, "rocker_pivot", (split - v.half_length, 0, plate_z + 2*pz + 0.01),
            (0.012, 0.012, 0.012), "0.85 0.75 0.2 1")
        parents = [rk, rk, ch, ch]
        rel = [(0, pos[0][1], 0), (0, pos[1][1], 0), (pos[2][0], pos[2][1], 0),
               (pos[3][0], pos[3][1], 0)]
    else:
        parents = [ch]*4
        rel = [(x, y, 0) for x, y in pos]
    info = {}
    for name, par, r in zip(MODULE_NAMES, parents, rel):
        add_module(par, f"{name}_", r, g)
        SUB(root.find("equality"), "connect", name=f"{name}_rear_ball", site1=f"{name}_rear_pin",
            site2=f"{name}_crank_pin", solref="0.002 1", solimp="0.99 0.999 0.0001")
    if cable:
        for name in MODULE_NAMES:
            info[name] = (add_cable_gimbal(root, world, f"{name}_", g)
                          if g.cable_layout in ("gimbal", "clamp", "hook")
                          else add_cable_axis(root, world, f"{name}_", g)
                          if g.cable_layout == "axis"
                          else add_cable(root, world, f"{name}_", g, "housing"))
    act = SUB(root, "actuator")
    for name in MODULE_NAMES:
        SUB(act, "motor", name=f"{name}_steer", joint=f"{name}_crank_spin", ctrlrange="-2 2")
        SUB(act, "motor", name=f"{name}_drive", joint=f"{name}_wheel_spin", ctrlrange="-1 1")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def build_bench(g: ModuleGeometry | None = None, dt=0.001, cable: str | None = None,
                gravity=False, **cable_kw):
    """One module, frame fixed in the air (no floor contact)."""
    g = g or ModuleGeometry()
    root = base_root("nutating_swerve_bench", dt, gravity=gravity)
    if cable:
        ext = SUB(root, "extension")
        SUB(ext, "plugin", plugin="mujoco.elasticity.cable")
    add_assets(root, g)
    world = SUB(root, "worldbody")
    add_floor(world)
    eq = SUB(root, "equality")
    add_module(world, "m_", (0, 0, 0.25), g, tire_collides=False)
    SUB(eq, "connect", name="m_rear_ball", site1="m_rear_pin", site2="m_crank_pin",
        solref="0.002 1", solimp="0.99 0.999 0.0001")
    if cable in ("gimbal", "clamp", "hook"):
        info = add_cable_gimbal(root, world, "m_", g, **cable_kw)
    elif cable == "axis":
        info = add_cable_axis(root, world, "m_", g, **cable_kw)
    elif cable == "hang":
        info = add_cable_hang(root, world, "m_", g, **cable_kw)
    elif cable:
        info = add_cable(root, world, "m_", g, cable, **cable_kw)
    else:
        info = {}
    act = SUB(root, "actuator")
    SUB(act, "motor", name="m_steer", joint="m_crank_spin", ctrlrange="-2 2")
    SUB(act, "motor", name="m_drive", joint="m_wheel_spin", ctrlrange="-1 1")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n", info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "models")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "quad.xml").write_text(build_vehicle(), encoding="utf-8")
    (args.out / "quad_rocker.xml").write_text(build_vehicle(VehicleGeometry(rocker=True)),
                                              encoding="utf-8")
    (args.out / "bench.xml").write_text(build_bench()[0], encoding="utf-8")
    print("written to", args.out)
