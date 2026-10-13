"""Model control for the 4-module vehicle.

Control quantities come from the vehicle/module kinematic model and the known
reference; feedback only trims the error (pose PI-less P trim, joint PD/PI loops).

Per module (body frame, pivot p_i, crank angle phi, rolling direction t(phi)):
  u_i   = v + w x p_i                         velocity the module must produce
  phi*  : t(phi*) parallel to u_i  (choose phi* nearest the current reference;
          optional wheel reversal = pi branches; otherwise 2 pi branches = endless steering)
  dpsi* = -( (u_i . t + w rho) + J_phi(phi) dphi ) / J_psi        no slip along t
  tau_s += (J_phi / J_psi) tau_d                                  coupling feed-forward
"""
from __future__ import annotations

import math
import numpy as np

from geometry import VehicleGeometry, contact_coefficients
from build_quad import MODULE_NAMES


def wrap(a):
    return (a + math.pi) % (2*math.pi) - math.pi


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def motor_limit(tau, omega, stall, omega0):
    """Linear DC motor torque-speed envelope (4 quadrant, clipped at stall)."""
    hi = stall*(1 - omega/omega0)
    lo = -stall*(1 + omega/omega0)
    return float(np.clip(tau, max(lo, -stall), min(hi, stall)))


class QuadController:
    # gains / limits
    kp_pos, kp_yaw = 2.0, 2.0
    kp_s, kd_s = 20.0, 0.35
    kv_d, ki_d = 0.25, 2.0
    steer_rate_max, steer_acc_max = 8.0, 30.0      # crank reference limits
    drive_stall, drive_w0 = 1.0, 35.0              # wheel side of the gear motor
    steer_stall, steer_w0 = 2.0, 15.0              # crank side incl. belt
    eps_speed = 0.01

    def __init__(self, model, ref, vg: VehicleGeometry | None = None, flip=None,
                 feedback=True, coupling_ff=True):
        self.m, self.ref = model, ref
        self.vg = vg or VehicleGeometry()
        self.g = self.vg.module
        self.flip = ref.flip if flip is None else flip
        self.feedback, self.coupling_ff = feedback, coupling_ff
        self.p = np.array(self.vg.module_positions(), float)
        self.j = []
        for n in MODULE_NAMES:
            self.j.append(dict(
                cq=model.joint(f"{n}_crank_spin").qposadr[0], cv=model.joint(f"{n}_crank_spin").dofadr[0],
                wv=model.joint(f"{n}_wheel_spin").dofadr[0],
                a_s=model.actuator(f"{n}_steer").id, a_d=model.actuator(f"{n}_drive").id))
        self.phi_ref = np.zeros(4)
        self.phid_ref = np.zeros(4)
        self.phi_tgt = np.zeros(4)
        self.prev_tgt = None
        self.integ = np.zeros(4)
        self.tau_d = np.zeros(4)
        self.tau_s = np.zeros(4)
        self.dpsi_ref = np.zeros(4)
        self.scale = np.ones(4)

    # -- reference velocities in body frame
    def body_twist(self, t, yaw):
        v = self.ref.vel(t)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c*v[0] + s*v[1], -s*v[0] + c*v[1]]), v[2]

    def module_u(self, vb, w):
        return vb[None, :] + w*np.column_stack([-self.p[:, 1], self.p[:, 0]])

    def heading_hint(self, t, i):
        """Direction for module i when it is (nearly) stopped: the next motion in the plan."""
        for dt in np.arange(0.05, 3.0, 0.05):
            pose = self.ref.pose(t + dt)
            vb, w = self.body_twist(t + dt, pose[2])
            u = self.module_u(vb, w)[i]
            if np.linalg.norm(u) > self.eps_speed:
                return math.atan2(u[1], u[0])
        return None

    def choose_branch(self, alpha, current, i=0):
        """Crank angle for heading alpha nearest to `current`.

        Right-hand modules prefer running with the wheel reversed (t = -u): their axis
        tilt then mirrors the left side and the contact offsets rho*e stay symmetric.
        Without wheel reversal (flip=False) that preference is the only branch family,
        i.e. 2 pi spacing = endless steering in one direction when the heading keeps turning.
        """
        base = alpha - math.pi/2 + (math.pi if self.p[i, 1] < 0 else 0.0)
        step = math.pi if self.flip else 2*math.pi
        k0 = round((current - base)/step)
        best, cost = None, None
        for k in (k0 - 1, k0, k0 + 1):
            c = base + k*step
            pen = 0.35 if (self.flip and k % 2) else 0.0      # non-preferred (reversed) branch
            if cost is None or abs(c - current) + pen < cost:
                best, cost = c, abs(c - current) + pen
        return best

    t_offset = 0.0     # reference time = data.time - t_offset

    def hold(self, data):
        """Before the run: hold the cranks, wheels free-damped (vehicle settles on its tires)."""
        for jj in self.j:
            data.ctrl[jj['a_s']] = np.clip(-20*data.qpos[jj['cq']] - 0.35*data.qvel[jj['cv']], -2, 2)
            data.ctrl[jj['a_d']] = -0.2*data.qvel[jj['wv']]

    def update(self, data, dt):
        t = data.time - self.t_offset
        m = self.m
        pos = data.qpos[0:2]
        yaw = yaw_of(data.qpos[3:7])
        rp = self.ref.pose(t)
        rv = self.ref.vel(t)
        v_w = rv[:2].copy()
        w = rv[2]
        if self.feedback:
            v_w += self.kp_pos*(rp[:2] - pos)
            w += self.kp_yaw*wrap(rp[2] - yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        vb = np.array([c*v_w[0] + s*v_w[1], -s*v_w[0] + c*v_w[1]])
        U = self.module_u(vb, w)
        tgt = self.phi_tgt.copy()
        for i in range(4):
            sp = np.linalg.norm(U[i])
            if sp > self.eps_speed:
                alpha = math.atan2(U[i][1], U[i][0])
            else:
                alpha = self.heading_hint(t, i) if (int(t*1000) % 20 == 0 or self.prev_tgt is None) else None
            if alpha is not None:
                tgt[i] = self.choose_branch(alpha, self.phi_tgt[i] if self.prev_tgt is not None
                                            else data.qpos[self.j[i]['cq']], i)
        tgt_rate = np.zeros(4) if self.prev_tgt is None else (tgt - self.prev_tgt)/dt
        tgt_rate = np.clip(tgt_rate, -self.steer_rate_max, self.steer_rate_max)
        self.prev_tgt = tgt.copy()
        self.phi_tgt = tgt
        # rate/accel limited crank reference that follows the target (model trajectory)
        err = tgt - self.phi_ref
        # time-optimal approach (decelerate in time) plus the target's own rate
        des = tgt_rate + np.sign(err)*np.minimum(np.sqrt(2*0.8*self.steer_acc_max*np.abs(err)),
                                                 25.0*np.abs(err) + 0.0)
        des = np.clip(des, -self.steer_rate_max, self.steer_rate_max)
        acc = np.clip((des - self.phid_ref)/dt, -self.steer_acc_max, self.steer_acc_max)
        self.phid_ref += acc*dt
        self.phi_ref += self.phid_ref*dt
        ctrl = data.ctrl
        for i, jj in enumerate(self.j):
            phi = data.qpos[jj['cq']]
            phid = data.qvel[jj['cv']]
            psid = data.qvel[jj['wv']]
            jphi, jpsi = contact_coefficients(self.g, phi)
            tvec = np.array([-math.sin(phi), math.cos(phi)])
            # only drive along t to the extent the wheel is aligned (no lateral scrubbing)
            align = math.cos(phi - self.phi_tgt[i])
            self.scale[i] = max(0.0, align)**4 if np.linalg.norm(U[i]) > self.eps_speed else 0.0
            ut = U[i] @ tvec + w*self.g.rho
            self.dpsi_ref[i] = -(self.scale[i]*ut + jphi*phid)/jpsi
            err = self.dpsi_ref[i] - psid
            self.integ[i] = np.clip(self.integ[i] + err*dt, -0.2, 0.2)
            td = self.kv_d*err + self.ki_d*self.integ[i]
            td = motor_limit(td, psid, self.drive_stall, self.drive_w0)
            ts = self.kp_s*(self.phi_ref[i] - phi) + self.kd_s*(self.phid_ref[i] - phid)
            if self.coupling_ff:
                ts += (jphi/jpsi)*td
            ts = motor_limit(ts, phid, self.steer_stall, self.steer_w0)
            self.tau_d[i], self.tau_s[i] = td, ts
            ctrl[jj['a_s']] = ts
            ctrl[jj['a_d']] = td
