"""Figures for REPORT.md from results/*/trace.csv (run verify_quad.py first)."""
from __future__ import annotations
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).parent / "results"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # categorical slots 1-4
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.6,
                     "legend.frameon": False, "axes.titlesize": 10, "axes.titleweight": "bold"})


def load(name):
    with open(R / name / "trace.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in rows[0]}


def main():
    fig, ax = plt.subplots(2, 3, figsize=(14, 8.2), constrained_layout=True)
    # (a) paths
    a = ax[0, 0]
    for name, col in (("square", S[0]), ("star", S[1]), ("circle_spin", S[2])):
        d = load(name)
        a.plot(d["x_ref"], d["y_ref"], color=GRID, lw=4, solid_capstyle="round")
        a.plot(d["x"], d["y"], color=col, label=name)
    a.set_aspect("equal"); a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    a.set_title("(a) Body path: reference (grey) vs simulated")
    a.legend(loc="upper left", fontsize=8)
    # (b) position error, model only vs model + trim
    a = ax[0, 1]
    for name, col, lab in (("square_model_only", S[1], "model control only"),
                           ("square", S[0], "model control + pose trim")):
        d = load(name)
        a.plot(d["t"], d["pos_err"]*1000, color=col, label=lab)
    a.set_xlabel("time [s]"); a.set_ylabel("position error [mm]")
    a.set_title("(b) Square, 2 laps (4.8 m): tracking error"); a.legend(fontsize=8)
    # (c) crank turns (endless) per module
    a = ax[0, 2]
    d = load("circle_spin")
    for k, n in enumerate(("fl", "fr", "rl", "rr")):
        a.plot(d["t"], (d[f"{n}_phi"] - d[f"{n}_phi"][0])/(2*math.pi), color=S[k], label=n.upper())
    a.set_xlabel("time [s]"); a.set_ylabel("steering crank [turns]")
    a.set_title("(c) Circle + body yaw: crank keeps turning"); a.legend(fontsize=8, ncol=2)
    # (d) housing orientation stays bounded
    a = ax[1, 0]
    a.plot(d["t"], d["fl_housing_rot_deg"], color=S[0], label="FL motor housing vs start")
    a.plot(d["t"], d["fl_housing_dev_deg"]*50, color=S[1], lw=1.0,
           label="deviation from f(crank angle) x50")
    a.set_xlabel("time [s]"); a.set_ylabel("rotation [deg]")
    a.set_ylim(0, 100)
    a.set_title("(d) Motor housing only nutates (bounded)"); a.legend(fontsize=8, loc="upper right")
    # (e) cable twist
    a = ax[1, 1]
    for name, col, lab in (("cable_hook", S[0], "U loop per gimbal axis, brackets (v4)"),
                           ("cable_pin", S[2], "through the crank-pin stub (v1)"),
                           ("cable_crank", S[1], "cable end turning with steering")):
        c = load(name)
        m = c["t"] > 1.0
        a.plot(c["crank_turns"][m], c["twist_deg"][m], color=col, label=lab)
    a.set_xlabel("steering crank [turns]"); a.set_ylabel("total cable twist [deg]")
    a.set_ylim(-120, 60)
    a.annotate("conventional: -320 deg/turn (off scale)", xy=(0.6, -75), color=S[1], fontsize=8)
    a.set_title("(e) Elastic cable model: twist"); a.legend(fontsize=8, loc="lower right")
    # (f) bump: wheel loads
    a = ax[1, 2]
    for name, ls, lab in (("bump", "-", "rigid (current)"), ("bump_rocker", ":", "rocker")):
        b = load(name)
        for n, col in (("fl", S[0]), ("rl", S[2])):
            a.plot(b["t"], b[f"{n}_Fn"], color=col, ls=ls, lw=1.4 if ls == "-" else 1.8,
                   label=f"{n.upper()} {lab}")
    a.set_xlim(2.0, 7.0); a.set_xlabel("time [s]"); a.set_ylabel("normal load [N]")
    a.set_title("(f) 8 mm bump under left wheels"); a.legend(fontsize=8, ncol=2)
    out = R / "quad_results.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
