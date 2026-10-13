"""Kinematic checks on the bench model (no dynamics):
loop closure with ideal gimbal angles, housing pose periodicity, and the closed-form
contact coefficients J_phi, J_psi versus MuJoCo's point Jacobian."""
from __future__ import annotations
import math
import numpy as np
import mujoco
from geometry import ModuleGeometry, gimbal_state, contact_coefficients
from build_quad import build_bench


def set_ideal(m, d, g, phi, p="m_"):
    gx, gy, _, _ = gimbal_state(g, phi)
    d.qpos[m.joint(f"{p}gimbal_x").qposadr[0]] = gx
    d.qpos[m.joint(f"{p}gimbal_y").qposadr[0]] = gy
    d.qpos[m.joint(f"{p}crank_spin").qposadr[0]] = phi


def ideal_contact(m, d, g, p="m_"):
    """Lowest point of the torus tire (world frame)."""
    R = d.body(f"{p}housing").xmat.reshape(3, 3)
    a = R[:, 2]
    W = d.body(f"{p}wheel").xpos
    e = a.copy(); e[2] = 0; e /= np.linalg.norm(e)
    n = g.c*e - g.s*np.array([0, 0, 1.0])
    return W + g.wheel_R*n - g.tire_r*np.array([0, 0, 1.0])


def check(g: ModuleGeometry | None = None, n=181):
    g = g or ModuleGeometry()
    m = mujoco.MjModel.from_xml_string(build_bench(g)[0])
    d = mujoco.MjData(m)
    ids = {k: (m.joint(f"m_{k}").qposadr[0], m.joint(f"m_{k}").dofadr[0])
           for k in ("gimbal_x", "gimbal_y", "crank_spin", "wheel_spin")}
    closure, pose_err, jphi_err, jpsi_err, perp = [], [], [], [], []
    first = None
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    for phi in np.linspace(0, 2*math.pi, n):
        set_ideal(m, d, g, phi)
        mujoco.mj_forward(m, d)
        closure.append(np.linalg.norm(d.site("m_rear_pin").xpos - d.site("m_crank_pin").xpos))
        pose = d.body("m_housing").xmat.reshape(3, 3).copy()
        first = pose if first is None else first
        pose_err.append(np.linalg.norm(pose - first))
        C = ideal_contact(m, d, g)
        mujoco.mj_jac(m, d, jacp, jacr, C, m.body("m_wheel").id)
        _, _, dgx, dgy = gimbal_state(g, phi)
        qd = np.zeros(m.nv)
        qd[ids["gimbal_x"][1]] = dgx; qd[ids["gimbal_y"][1]] = dgy; qd[ids["crank_spin"][1]] = 1
        v_phi = jacp @ qd
        v_psi = jacp[:, ids["wheel_spin"][1]]
        t = np.array([-math.sin(phi), math.cos(phi), 0])
        jp, js = contact_coefficients(g, phi)
        jphi_err.append(np.linalg.norm(v_phi - jp*t))
        jpsi_err.append(np.linalg.norm(v_psi - js*t))
        perp.append(abs(v_phi @ np.cross([0, 0, 1], t)) + abs(v_psi @ np.cross([0, 0, 1], t)))
    return {"max_closure_m": float(max(closure)), "one_turn_pose_error": float(pose_err[-1]),
            "max_Jphi_error_m": float(max(jphi_err)), "max_Jpsi_error_m": float(max(jpsi_err)),
            "max_radial_component_m": float(max(perp))}


if __name__ == "__main__":
    import json
    print(json.dumps(check(), indent=2))
