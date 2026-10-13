"""Run every check for the 4-wheel model and write results/quad/verification.json.

  python verify_quad.py                 # everything (about 3-4 min on a laptop core)
  python verify_quad.py --only square circle --reuse
`--reuse` keeps result folders that already have a summary.json.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent / "results"


def run_scenario(name, scenario, extra=(), reuse=False):
    import run_quad
    out = ROOT / name
    if reuse and (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text(encoding="utf-8"))
    args = run_quad.parse(["--scenario", scenario, "--out", str(out), *extra])
    with contextlib.redirect_stdout(io.StringIO()):
        return run_quad.simulate(args)


def part_kinematics(reuse):
    import kinematics_check
    return kinematics_check.check()


def part_clearance(reuse):
    import check_clearance as cc
    m, ranked, n = cc.module_check()
    vr, vn = cc.vehicle_check()
    top = lambda r: [{"pair": list(k), "clearance_mm": round(v[0]*1000, 2)} for k, v in r[:8]]
    return {"module_pairs": n, "module_min_mm": ranked[0][1][0]*1000, "module_closest": top(ranked),
            "vehicle_pairs": vn, "vehicle_min_mm": vr[0][1][0]*1000, "vehicle_closest": top(vr)}


def part_cable(reuse):
    import cable_test
    res = {}
    for layout, turns in (("hook", 5), ("gimbal", 5), ("clamp", 5), ("pin", 5), ("crank", 2)):
        out = ROOT / f"cable_{layout}"
        if reuse and (out / "summary.json").exists():
            res[layout] = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        else:
            res[layout] = cable_test.run(layout, turns, out=out)
    return res


SCEN = {
    "square": ("square", ()), "circle": ("circle", ()), "circle_spin": ("circle_spin", ()),
    "star": ("star", ()), "spin": ("spin", ()),
    "bump": ("straight", ("--bumps",)), "bump_rocker": ("straight", ("--bumps", "--rocker")),
    "circle_half_dt": ("circle", ("--dt", "0.0005")),
    "square_model_only": ("square", ("--no-feedback",)),
    "square_no_coupling_ff": ("square", ("--no-coupling-ff",)),
}


def checks(R):
    c = {}
    k = R.get("kinematics")
    if k:
        c["loop closure and housing periodicity"] = k["max_closure_m"] < 1e-8 and k["one_turn_pose_error"] < 1e-8
        c["closed-form J_phi/J_psi match MuJoCo Jacobian"] = max(k["max_Jphi_error_m"], k["max_Jpsi_error_m"]) < 1e-9
        c["module instantaneous rank 1 (no radial component)"] = k["max_radial_component_m"] < 1e-9
    cl = R.get("clearance")
    if cl:
        c["module parts >= 2 mm apart over a full turn"] = cl["module_min_mm"] >= 2.0
        c["no contact between modules / chassis"] = cl["vehicle_min_mm"] >= 2.0
    for name in ("square", "circle", "circle_spin", "star", "spin", "circle_half_dt"):
        s = R.get(name)
        if not s:
            continue
        mods = s["modules"].values()
        c[f"{name}: tracking < 10 mm / 1.5 deg, no warnings"] = (
            s["max_pos_err_mm"] < 10 and s["max_yaw_err_deg"] < 1.5 and not s["warnings"])
        c[f"{name}: 4 wheels within 0.2 mm of floor >= 99 %"] = min(m["contact_fraction"] for m in mods) >= 0.99
        c[f"{name}: housing pose = f(crank angle) within 1 deg"] = max(
            m["max_housing_dev_from_phi_model_deg"] for m in mods) < 1.0
    if R.get("square"):
        t = [m["crank_turns"] for m in R["square"]["modules"].values()]
        c["square: endless steering (>= 1.5 turns, one direction)"] = min(t) > 1.45
    if R.get("circle_spin"):
        t = [m["crank_turns"] for m in R["circle_spin"]["modules"].values()]
        c["circle_spin: >= 3.5 crank turns"] = min(t) > 3.5
    s = R.get("bump")
    if s:
        c["bump (rigid chassis): tracking < 10 mm / 2 deg, no warnings"] = (
            s["max_pos_err_mm"] < 10 and s["max_yaw_err_deg"] < 2 and not s["warnings"])
    if R.get("circle") and R.get("circle_half_dt"):
        a, b = R["circle"], R["circle_half_dt"]
        c["circle: dt 1 ms vs 0.5 ms max error within 2 mm"] = abs(a["max_pos_err_mm"] - b["max_pos_err_mm"]) < 2.0
    cb = R.get("cable")
    if cb and "hook" in cb and "crank" in cb:
        gc = cb["hook"]                       # adopted layout; the others are reported only
        c["cable (v4 brackets): twist returns within 10 deg every turn"] = max(
            abs(x) for x in gc["twist_at_each_full_turn_deg"][1:]) < 10
        c["cable (v4 brackets): local twist rate <= 180 deg/m (chainflex torsion rating)"] = (
            gc["max_local_twist_rate_deg_per_m"] <= 180)
        c["cable (v4 brackets): >= 2 mm from every part (beyond the clamp sleeves)"] = (
            gc["min_cable_clearance_mm"] >= 2)
        c["conventional (cable end steered): twist accumulates > 250 deg/turn"] = (
            cb["crank"]["final_twist_deg"] < -250*cb["crank"]["turns_done"] + 1)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "verification.json"
    R = (json.loads(path.read_text(encoding="utf-8"))["results"]
         if ((a.reuse or a.only) and path.exists()) else {})
    parts = ["kinematics", "clearance", *SCEN.keys(), "cable"]
    for p in parts:
        if a.only and p not in a.only:
            continue
        if a.reuse and p in R and p not in (a.only or []):
            continue
        print("running", p, flush=True)
        fresh = a.reuse                       # --reuse: keep result folders that exist
        if p == "kinematics":
            R[p] = part_kinematics(fresh)
        elif p == "clearance":
            R[p] = part_clearance(fresh)
        elif p == "cable":
            R[p] = part_cable(fresh)
        else:
            sc, extra = SCEN[p]
            R[p] = run_scenario(p, sc, extra, a.reuse and not a.only)
        C = {k: bool(v) for k, v in checks(R).items()}
        path.write_text(json.dumps({"passed": all(C.values()), "checks": C, "results": R},
                                   indent=2), encoding="utf-8")
    C = checks(R)
    for k, v in C.items():
        print(("PASS " if v else "FAIL ") + k)
    print("ALL PASSED" if all(C.values()) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
