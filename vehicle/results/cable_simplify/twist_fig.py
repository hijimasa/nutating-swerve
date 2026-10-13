"""Where and when does the twist appear?  Re-simulates one hang configuration, logs total
twist / local twist rate per joint / bend radius, renders the cable at several phases with
segments coloured by local twist rate, and plots the metrics over one steering turn."""
import math, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # vehicle/
import mujoco, numpy as np
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
from build_quad import build_bench
from geometry import ModuleGeometry
import cable_test as ct, route_figure as rf

side, a_exit, cx, cy, cz, L, period, out = sys.argv[1], *map(float, sys.argv[2:8]), sys.argv[8]
g = ModuleGeometry(cable_layout="hang", hang_exit=side, exit_a=a_exit, hang_clamp=(cx, cy, cz), hang_len=L,
                   ring_bar=0.005, pillow_hz=0.014, arm_y=-0.0225)
xml, _ = build_bench(g, dt=0.0005, cable="hang", gravity=True)
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
ids = ct.cable_bodies(m, "m_cab"); geom_of = {b: [i for i in range(m.ngeom) if m.geom_bodyid[i] == b][0] for b in ids}
for i in range(m.ngeom):
    if m.geom_bodyid[i] not in ids and m.geom(i).name != "floor": m.geom_rgba[i, 3] = 0.28
qc, vc = m.joint("m_crank_spin").qposadr[0], m.joint("m_crank_spin").dofadr[0]
vw = m.joint("m_wheel_spin").dofadr[0]
ramp, w, settle = 0.5, 2*math.pi/period, 1.0
turns = 2.0
mujoco.mj_forward(m, d)
ref = None; log = []; states = {}
want = {"0": 1.0, "90": 1.25, "180": 1.5, "270": 1.75}          # crank turns at which to snapshot (2nd turn)
rend = mujoco.Renderer(m, rf.H, rf.W); opt = mujoco.MjvOption(); opt.geomgroup[1] = opt.geomgroup[3] = 1
def snapshot():
    return (d.qpos.copy(), d.qvel.copy())
next_check = 0.0
while d.time < settle + turns*period + ramp + 0.3:
    t = d.time - settle
    if t >= 0 and ref is None:
        ref = ct.twist_and_bend(m, d, ids)
    if t < 0: ang, rate = 0.0, 0.0
    elif t < ramp: ang, rate = w*t*t/(2*ramp), w*t/ramp
    else: ang, rate = min(w*(t - ramp/2), 2*math.pi*turns), (w if t < turns*period + ramp/2 else 0.0)
    d.ctrl[m.actuator("m_steer").id] = np.clip(20*(ang - d.qpos[qc]) + 0.4*(rate - d.qvel[vc]), -2, 2)
    d.ctrl[m.actuator("m_drive").id] = -0.05*d.qvel[vw]
    mujoco.mj_step(m, d)
    if d.time >= next_check and ref is not None:
        next_check += 0.02
        tw, rb, rates = ct.twist_and_bend(m, d, ids)
        loc = np.degrees(rates - ref[2])
        turnsn = d.qpos[qc]/(2*math.pi)
        log.append((turnsn, math.degrees(tw - ref[0]), rb*1000, float(np.max(np.abs(loc))), int(np.argmax(np.abs(loc))), loc.copy()))
        for k, v in want.items():
            if k not in states and turnsn >= v: states[k] = (snapshot(), len(log) - 1)
# moment of largest total twist in the second turn
second = [i for i, r in enumerate(log) if 1.0 <= r[0] <= 2.0]
imax = max(second, key=lambda i: abs(log[i][1]))
# re-simulate to that moment for a snapshot: simpler to store all states -> redo with stored qpos
# (we stored only 4 states; re-run quickly to imax's time)
d2 = mujoco.MjData(m); next_check = 0.0; ref2 = None; k = -1
while True:
    t = d2.time - settle
    if t >= 0 and ref2 is None: ref2 = True
    if t < 0: ang, rate = 0.0, 0.0
    elif t < ramp: ang, rate = w*t*t/(2*ramp), w*t/ramp
    else: ang, rate = min(w*(t - ramp/2), 2*math.pi*turns), (w if t < turns*period + ramp/2 else 0.0)
    d2.ctrl[m.actuator("m_steer").id] = np.clip(20*(ang - d2.qpos[qc]) + 0.4*(rate - d2.qvel[vc]), -2, 2)
    d2.ctrl[m.actuator("m_drive").id] = -0.05*d2.qvel[vw]
    mujoco.mj_step(m, d2)
    if d2.time >= next_check and ref2:
        next_check += 0.02; k += 1
        if k == imax: states["max"] = ((d2.qpos.copy(), d2.qvel.copy()), imax); break

def render(state, title, loc, imark):
    d.qpos[:], d.qvel[:] = state; mujoco.mj_forward(m, d)
    lim = 400.0
    for j, b in enumerate(ids):
        r = min(1.0, (abs(loc[min(j, len(loc) - 1)])/lim))
        m.geom_rgba[geom_of[b]] = (0.1 + 0.9*r, 0.25*(1 - r), 0.95*(1 - r), 1)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:] = d.xpos[m.body("m_mount").id] + np.array([0.0, 0.06, 0.06]); cam.distance, cam.azimuth, cam.elevation = 0.5, 35, -18
    rend.update_scene(d, cam, opt); img = Image.fromarray(rend.render()); dr = ImageDraw.Draw(img)
    x, y = rf.project(m, cam, d.xpos[ids[imark]])
    dr.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(220, 30, 30), width=3)
    xc, yc = rf.project(m, cam, d.xpos[ids[0]]); dr.text((xc + 8, yc), "モーター側留め具", font=rf.font(15), fill=(40, 40, 40))
    xf, yf = rf.project(m, cam, d.xpos[ids[-1]]); dr.text((xf - 140, yf - 22), "フレーム側留め具", font=rf.font(15), fill=(40, 40, 40))
    dr.rectangle((0, 0, rf.W, 34), fill=(22, 27, 34)); dr.text((12, 5), title, font=rf.font(18), fill=(240, 243, 246))
    return img
panels = []
for k, lab in (("0", "操舵角0°"), ("90", "90°"), ("180", "180°"), ("270", "270°"), ("max", "ねじれ最大の瞬間")):
    st, i = states[k]; r = log[i]
    panels.append(render(st, f"{lab}（{r[0]*360 % 360:.0f}°）  全体ねじれ {r[1]:+.0f}°  局所最大 {r[3]:.0f} °/m（節 {r[4]}/{len(ids)-1}）  最小曲げ半径 {r[2]:.0f} mm", r[5], r[4]))
W, H = rf.W, rf.H
plt.rcParams["font.family"] = font_manager.FontProperties(fname=rf.FONT).get_name() if os.path.exists(rf.FONT) else "sans-serif"
fig, ax = plt.subplots(3, 1, figsize=(W*5/100/2.2, 5.2), sharex=True, dpi=100)
ph = [r[0]*360 for r in log if 1.0 <= r[0] <= 2.0]; sel = [r for r in log if 1.0 <= r[0] <= 2.0]
ax[0].plot(ph, [r[1] for r in sel], color="#1a40f2"); ax[0].set_ylabel("全体ねじれ [°]"); ax[0].axhline(0, color="gray", lw=0.5)
ax[1].plot(ph, [r[3] for r in sel], color="#d02020"); ax[1].set_ylabel("局所ねじり率\n最大 [°/m]"); ax[1].set_yscale("log")
ax[2].plot(ph, [r[2] for r in sel], color="#208040"); ax[2].set_ylabel("最小曲げ半径 [mm]"); ax[2].set_xlabel("操舵角（2周目）[°]"); ax[2].set_ylim(0, 80)
for a in ax: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(out + "_plot.png"); plot = Image.open(out + "_plot.png")
canvas = Image.new("RGB", (W*3, H*2 + 70), (255, 255, 255))
for k, im in enumerate(panels[:3]): canvas.paste(im, (k*W, 0))
for k, im in enumerate(panels[3:]): canvas.paste(im, (k*W, H))
canvas.paste(plot.resize((W, H)), (2*W, H))
dr = ImageDraw.Draw(canvas); dr.text((16, 2*H + 20), "ケーブルの色：青＝局所ねじり率ほぼ0、赤＝400 °/m以上。赤丸＝局所ねじり率が最大の節。半透明＝部品（接触は計算していない）", font=rf.font(19), fill=(30, 30, 30))
canvas.save(out + ".png")
# where does the local maximum sit along the cable?
import collections
c = collections.Counter(r[4] for r in sel); print("local-max joint histogram (joint index: count):", sorted(c.items())[:12], "... n_joints", len(ids) - 1)
print("max total twist in 2nd turn:", max(abs(r[1]) for r in sel), "deg at phase", log[imax][0]*360 % 360)
print(out + ".png")
