import sys, json, numpy as np
from math import radians
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # vehicle/
from geometry import ModuleGeometry, gimbal_state
from build_quad import _ry
import cable_test as ct
name = sys.argv[1]; a_exit, cx, cy, cz = map(float, sys.argv[2:6]); out = sys.argv[6]; period = float(sys.argv[7]); margin = float(sys.argv[8]) if len(sys.argv) > 8 else 0.010; side = sys.argv[9] if len(sys.argv) > 9 else "side"; exit_angle = float(sys.argv[10]) if len(sys.argv) > 10 else 0.0
g0 = ModuleGeometry()
def Rx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
off = g0.motor_radius + g0.cable_radius + g0.exit_clear
E_local = np.array([-off, 0, a_exit]) if side == "side" else np.array([0, off, a_exit])  # side_y / side_y_ax share the exit point
chords = [np.linalg.norm(Rx(gx) @ _ry(gy) @ E_local - np.array([cx, cy, cz])) for gx, gy, _, _ in (gimbal_state(g0, radians(d)) for d in range(0, 360, 5))]
L = max(chords) + 0.01 + margin          # 10 mm straight stub at the clamp + slack margin
g = ModuleGeometry(cable_layout="hang", hang_exit=side, exit_angle_deg=exit_angle, exit_a=a_exit, hang_clamp=(cx, cy, cz), hang_len=L,
                   ring_bar=0.005, pillow_hz=0.014, arm_y=-0.0225)
r = ct.run("hang", 5, period=period, out=f"{out}/{name}", g=g)
keys = ["twist_range_deg","twist_at_each_full_turn_deg","max_local_twist_rate_deg_per_m","min_bend_radius_mm",
        "settled_min_bend_radius_mm","max_cyclic_curvature_range_per_m","cyclic_bend_radius_equiv_mm",
        "min_cable_clearance_mm","closest_part","clearance_by_part_mm","peak_steer_torque_Nm","turns_done","warnings"]
print(name, f"L={L*1000:.0f}mm chord {1000*min(chords):.0f}..{1000*max(chords):.0f}", json.dumps({k: r.get(k) for k in keys}))
