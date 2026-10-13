import sys, json
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # vehicle/
from geometry import ModuleGeometry
import cable_test as ct
name, route, R, dz, free = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
g = ModuleGeometry(cable_layout="pin", cable_route=route, cable_bend_R=R, plate_dz=dz,
                   cable_free_len=free, ring_bar=0.005, pillow_hz=0.014, arm_y=-0.0225)
r = ct.run("pin", 5, out=f"{sys.argv[6]}/{name}", g=g)
keys = ["twist_range_deg","twist_at_each_full_turn_deg","max_local_twist_rate_deg_per_m","min_bend_radius_mm",
        "settled_min_bend_radius_mm","max_cyclic_curvature_range_per_m","cyclic_bend_radius_equiv_mm",
        "min_cable_clearance_mm","closest_part","peak_steer_torque_Nm","turns_done"]
print(name, json.dumps({k: r.get(k) for k in keys}))
