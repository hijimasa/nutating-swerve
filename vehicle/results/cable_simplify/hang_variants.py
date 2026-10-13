import sys, json
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # vehicle/
from geometry import ModuleGeometry
import cable_test as ct
name = sys.argv[1]; cx, cy, cz, L = map(float, sys.argv[2:6]); out = sys.argv[6]
g = ModuleGeometry(cable_layout="hang", hang_clamp=(cx, cy, cz), hang_len=L, ring_bar=0.005, pillow_hz=0.014, arm_y=-0.0225)
r = ct.run("hang", 5, period=float(sys.argv[7]) if len(sys.argv) > 7 else 2.0, out=f"{out}/{name}", g=g)
keys = ["twist_range_deg","twist_at_each_full_turn_deg","max_local_twist_rate_deg_per_m","min_bend_radius_mm",
        "settled_min_bend_radius_mm","max_cyclic_curvature_range_per_m","cyclic_bend_radius_equiv_mm",
        "min_cable_clearance_mm","closest_part","clearance_by_part_mm","peak_steer_torque_Nm","turns_done","warnings"]
print(name, json.dumps({k: r.get(k) for k in keys}))
