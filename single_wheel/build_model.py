"""Generate an MJCF closed-loop prototype. Lengths: m; angles: rad."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import cos, sin, radians
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass
class Geometry:
    tilt_deg: float = 45.0
    wheel_radius: float = 0.070
    wheel_half_width: float = 0.006
    rear_length: float = 0.120
    wheel_offset: float = 0.040
    timestep: float = 0.001


def f(value: float) -> str:
    return f"{value:.10g}"


def vec(*values: float) -> str:
    return " ".join(map(f, values))


def build_xml(mode: str = "air", g: Geometry | None = None) -> str:
    g = g or Geometry()
    if mode not in {"air", "ground"}:
        raise ValueError("mode must be air or ground")
    if not 5 <= g.tilt_deg <= 75:
        raise ValueError("This prototype supports 5 <= tilt_deg <= 75")
    if min(g.wheel_radius, g.wheel_half_width, g.rear_length, g.wheel_offset, g.timestep) <= 0:
        raise ValueError("All dimensions and timestep must be positive")
    theta = radians(g.tilt_deg)
    radius, height = g.rear_length * sin(theta), g.rear_length * cos(theta)
    pivot_z = (g.wheel_radius * sin(theta)
               + (g.wheel_offset + g.wheel_half_width) * cos(theta) + 0.0002)
    if mode == "air":
        pivot_z = 0.19
    root = ET.Element("mujoco", model=f"nutating_swerve_{mode}")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=f(g.timestep),
                  gravity="0 0 0" if mode == "air" else "0 0 -9.81",
                  integrator="implicitfast", solver="Newton", iterations="100",
                  tolerance="1e-10", cone="elliptic", impratio="5")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    ET.SubElement(visual, "headlight", diffuse="0.8 0.8 0.8", ambient="0.35 0.35 0.35")
    ET.SubElement(root, "statistic", center="0 0 0.15", extent="0.45")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0.001", armature="0.00001", limited="false")
    ET.SubElement(default, "geom", contype="0", conaffinity="0", rgba="0.6 0.65 0.7 1")
    ET.SubElement(default, "site", size="0.004", rgba="0.2 0.9 0.3 1")
    assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "texture", name="checker", type="2d", builtin="checker",
                  rgb1="0.15 0.18 0.22", rgb2="0.23 0.27 0.31", width="256", height="256")
    ET.SubElement(assets, "material", name="floor_mat", texture="checker", texrepeat="12 12")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 1.5", dir="0 0 -1", diffuse="0.9 0.9 0.9")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 0.1",
                  material="floor_mat", contype="1", conaffinity="1", condim="4",
                  friction="1.0 0.003 0.0001", solref="0.015 1", solimp="0.95 0.99 0.001")
    carrier = ET.SubElement(world, "body", name="carrier", pos=vec(0, 0, pivot_z))
    if mode == "ground":
        # Translation is free; all rotations are constrained by an ideal test fixture.
        for name, axis in (("base_x", "1 0 0"), ("base_y", "0 1 0"), ("base_z", "0 0 1")):
            ET.SubElement(carrier, "joint", name=name, type="slide", axis=axis,
                          damping="0", armature="0")
    ET.SubElement(carrier, "inertial", pos="0 0 0.10", mass="1.2", diaginertia="0.004 0.004 0.004")
    ET.SubElement(carrier, "geom", type="box", pos="0 0 0.14", size="0.055 0.055 0.008",
                  rgba="0.28 0.34 0.45 0.65")
    for x in (-0.046, 0.046):
        ET.SubElement(carrier, "geom", type="capsule", fromto=vec(x, 0, 0, x, 0, 0.14),
                      size="0.004", rgba="0.5 0.55 0.65 1")
    ET.SubElement(carrier, "site", name="pivot", pos="0 0 0", rgba="0.9 0.2 0.2 1")

    outer = ET.SubElement(carrier, "body", name="outer_gimbal")
    ET.SubElement(outer, "joint", name="gimbal_x", type="hinge", axis="1 0 0")
    ET.SubElement(outer, "inertial", pos="0 0 0", mass="0.06", diaginertia="0.00008 0.00005 0.00005")
    edges = [((-0.040, -0.034, 0), (0.040, -0.034, 0)),
             ((0.040, -0.034, 0), (0.040, 0.034, 0)),
             ((0.040, 0.034, 0), (-0.040, 0.034, 0)),
             ((-0.040, 0.034, 0), (-0.040, -0.034, 0))]
    for start, end in edges:
        ET.SubElement(outer, "geom", type="capsule", fromto=vec(*start, *end),
                      size="0.003", rgba="0.88 0.27 0.2 1")

    housing = ET.SubElement(outer, "body", name="motor_housing", euler=vec(0, theta, 0))
    ET.SubElement(housing, "joint", name="gimbal_y", type="hinge", axis="0 1 0", ref=f(theta))
    ET.SubElement(housing, "inertial", pos="0 0 0.05", mass="0.20",
                  diaginertia="0.00035 0.00035 0.00010")
    ET.SubElement(housing, "geom", type="cylinder", pos="0 0 0.055", size="0.018 0.045",
                  rgba="0.96 0.60 0.15 1")
    ET.SubElement(housing, "geom", type="capsule", fromto=vec(0, 0, -g.wheel_offset, 0, 0, g.rear_length),
                  size="0.004", rgba="0.72 0.75 0.8 1")
    # A visible asymmetric stripe makes rotation of the stator distinguishable from rotor spin.
    ET.SubElement(housing, "geom", type="capsule", fromto="0.019 0 0.02 0.019 0 0.09",
                  size="0.0025", rgba="0.1 0.2 0.3 1")
    ET.SubElement(housing, "site", name="rear_pin", pos=vec(0, 0, g.rear_length))
    ET.SubElement(housing, "site", name="cable_anchor", pos="0.021 0 0.07", rgba="0.6 0.3 0.9 1")

    wheel = ET.SubElement(housing, "body", name="wheel", pos=vec(0, 0, -g.wheel_offset))
    ET.SubElement(wheel, "joint", name="wheel_spin", type="hinge", axis="0 0 1",
                  damping="0.0002", armature="0.00002")
    ET.SubElement(wheel, "geom", name="tire", type="cylinder",
                  size=vec(g.wheel_radius, g.wheel_half_width), mass="0.15",
                  rgba="0.12 0.16 0.21 1", contype="1" if mode == "ground" else "0",
                  conaffinity="1" if mode == "ground" else "0", condim="4",
                  friction="1.0 0.003 0.0001", solref="0.015 1", solimp="0.95 0.99 0.001")
    ET.SubElement(wheel, "geom", type="cylinder", size=vec(g.wheel_radius * 0.65, g.wheel_half_width + 0.0002),
                  mass="0", rgba="0.20 0.55 0.78 1")
    for axis in ("x", "y"):
        start = (-g.wheel_radius*0.85, 0, g.wheel_half_width+0.001) if axis == "x" else (0, -g.wheel_radius*0.85, g.wheel_half_width+0.001)
        end = (g.wheel_radius*0.85, 0, g.wheel_half_width+0.001) if axis == "x" else (0, g.wheel_radius*0.85, g.wheel_half_width+0.001)
        ET.SubElement(wheel, "geom", type="capsule", fromto=vec(*start, *end),
                      size="0.0015", mass="0", rgba="0.8 0.85 0.9 1")

    crank = ET.SubElement(carrier, "body", name="crank", pos=vec(0, 0, height))
    ET.SubElement(crank, "joint", name="crank_spin", type="hinge", axis="0 0 1",
                  damping="0.001", armature="0.0001")
    ET.SubElement(crank, "inertial", pos=vec(radius/2, 0, 0), mass="0.10",
                  diaginertia="0.00008 0.00020 0.00020")
    ET.SubElement(crank, "geom", type="capsule", fromto=vec(0, 0, 0, radius, 0, 0),
                  size="0.005", rgba="0.65 0.38 0.87 1")
    ET.SubElement(crank, "geom", type="cylinder", size="0.014 0.008", rgba="0.65 0.38 0.87 1")
    ET.SubElement(crank, "site", name="crank_pin", pos=vec(radius, 0, 0))
    equality = ET.SubElement(root, "equality")
    # Ball-joint loop closure, not a weld: orientations are allowed to differ.
    ET.SubElement(equality, "connect", name="rear_ball_joint", site1="rear_pin", site2="crank_pin",
                  solref="0.003 1", solimp="0.99 0.999 0.0001")
    actuators = ET.SubElement(root, "actuator")
    ET.SubElement(actuators, "motor", name="steering_motor", joint="crank_spin",
                  ctrllimited="true", ctrlrange="-3 3", gear="1")
    ET.SubElement(actuators, "motor", name="drive_motor", joint="wheel_spin",
                  ctrllimited="true", ctrlrange="-0.8 0.8", gear="1")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tilt", type=float, default=45)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "models")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for mode in ("air", "ground"):
        path = args.out / f"{mode}.xml"
        path.write_text(build_xml(mode, Geometry(tilt_deg=args.tilt)), encoding="utf-8")
        print(path)
