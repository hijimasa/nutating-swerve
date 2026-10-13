"""Independent geometry and dynamics checks; no prescribed gimbal motion in dynamics."""
from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from build_model import Geometry, build_xml
from run import simulate


def ideal_state(theta, phase):
    s, c = math.sin(theta), math.cos(theta)
    gx = -math.atan2(s*math.sin(phase), c)
    gy = math.asin(s*math.cos(phase))
    gx_rate = -s*c*math.cos(phase)/(c*c+s*s*math.sin(phase)**2)
    gy_rate = -s*math.sin(phase)/math.sqrt(1-s*s*math.cos(phase)**2)
    return gx, gy, gx_rate, gy_rate


def geometry_check():
    records = []
    for tilt in (30.0, 45.0, 60.0):
        g = Geometry(tilt_deg=tilt)
        theta = math.radians(tilt)
        m = mujoco.MjModel.from_xml_string(build_xml("air", g))
        d = mujoco.MjData(m)
        assert m.nu == 2
        actuated_joints = {int(x) for x in m.actuator_trnid[:, 0]}
        assert actuated_joints == {m.joint('crank_spin').id, m.joint('wheel_spin').id}
        joints = {k: (int(m.joint(k).qposadr[0]), int(m.joint(k).dofadr[0]))
                  for k in ('gimbal_x','gimbal_y','crank_spin','wheel_spin')}
        closures, pose_errors, singular_ratios = [], [], []
        initial = None
        for phase in np.linspace(0, 2*math.pi, 181):
            gx, gy, gx_rate, gy_rate = ideal_state(theta, phase)
            # This is an independent kinematic check, distinct from torque-driven simulation.
            d.qpos[joints['gimbal_x'][0]] = gx
            d.qpos[joints['gimbal_y'][0]] = gy
            d.qpos[joints['crank_spin'][0]] = phase
            mujoco.mj_forward(m, d)
            closures.append(float(np.linalg.norm(d.site('rear_pin').xpos-d.site('crank_pin').xpos)))
            pose = d.body('motor_housing').xmat.reshape(3,3).copy()
            if initial is None:
                initial = pose
            pose_errors.append(float(np.linalg.norm(pose-initial)))
            axis = pose[:, 2]
            e = axis.copy(); e[2] = 0; e /= np.linalg.norm(e)
            ideal_contact = (d.body('wheel').xpos
                             + g.wheel_radius*(math.cos(theta)*e-math.sin(theta)*np.array([0,0,1]))
                             - g.wheel_half_width*axis)
            jacp, jacr = np.zeros((3,m.nv)), np.zeros((3,m.nv))
            mujoco.mj_jac(m,d,jacp,jacr,ideal_contact,m.body('wheel').id)
            passive_rate = np.zeros(m.nv)
            passive_rate[joints['gimbal_x'][1]] = gx_rate
            passive_rate[joints['gimbal_y'][1]] = gy_rate
            passive_rate[joints['crank_spin'][1]] = 1
            # Required carrier translation = minus wheel material velocity at ideal contact.
            mapping = -np.column_stack((jacp[:2, joints['wheel_spin'][1]], jacp[:2]@passive_rate))
            singular = np.linalg.svd(mapping, compute_uv=False)
            singular_ratios.append(float(singular[1]/singular[0]))
        assert max(closures) < 1e-8
        assert pose_errors[-1] < 1e-8
        # The centered constant-tilt arrangement is a steerable wheel, not a full-rank active caster.
        assert max(singular_ratios) < 1e-8
        records.append({'tilt_deg':tilt, 'max_ideal_closure_m':max(closures),
                        'one_turn_pose_matrix_error':pose_errors[-1],
                        'max_motion_jacobian_sigma2_over_sigma1':max(singular_ratios),
                        'instantaneous_planar_rank':1})
    return records


def run_check(mode, out, **kwargs):
    settings = dict(mode=mode, tilt=45.0, cycles=2, period=4.0, wheel_speed=3.0,
                    torsion=0.003, dt=0.001, out=str(out), viewer=False, video='', hold_steering=False)
    settings.update(kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        result = simulate(SimpleNamespace(**settings))
    assert not result['warnings'], result['warnings']
    assert result['closure_max_mm'] < 0.05
    assert max(abs(x-45.0) for x in result['tilt_range_deg']) < 0.03
    assert result['final_housing_return_error_deg'] < 0.02
    if not settings['hold_steering']:
        assert abs(result['completed_crank_turns']-settings['cycles']) < 0.001
    return result


if __name__ == '__main__':
    root = Path(__file__).parent / 'results'
    geometric = geometry_check()
    air = run_check('air', root/'air_10turns', cycles=10, period=2.0)
    ground = run_check('ground', root/'ground')
    half = run_check('ground', root/'ground_half_dt', dt=0.0005)
    straight = run_check('ground', root/'straight', cycles=1, hold_steering=True)
    displacement_difference = float(np.linalg.norm(np.array(ground['final_base_xy_m'])-half['final_base_xy_m']))
    assert ground['contact_fraction'] > 0.99
    assert half['contact_fraction'] > 0.99
    assert displacement_difference < 0.003
    assert abs(straight['final_base_xy_m'][0]) < 0.002
    assert straight['final_base_xy_m'][1] < -0.5
    output = {'passed':True, 'geometry':geometric,
              'air_10turns':air, 'ground':ground, 'ground_half_dt':half,
              'straight':straight, 'ground_dt_endpoint_difference_mm':displacement_difference*1000}
    (root/'verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'air_turns':air['completed_crank_turns'],
                      'air_closure_max_mm':air['closure_max_mm'],
                      'ground_dt_endpoint_difference_mm':displacement_difference*1000,
                      'instantaneous_planar_rank':1},indent=2))
