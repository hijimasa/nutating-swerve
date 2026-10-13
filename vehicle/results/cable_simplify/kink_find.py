"""Find when/where the largest local twist rate occurs over the full 5-turn test and render it."""
import math, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # vehicle/
import mujoco, numpy as np
from PIL import Image, ImageDraw
from build_quad import build_bench
from geometry import ModuleGeometry
import cable_test as ct, route_figure as rf
side, a_exit, cx, cy, cz, L, period, out = sys.argv[1], *map(float, sys.argv[2:8]), sys.argv[8]
g = ModuleGeometry(cable_layout="hang", hang_exit=side, exit_a=a_exit, hang_clamp=(cx, cy, cz), hang_len=L,
                   ring_bar=0.005, pillow_hz=0.014, arm_y=-0.0225)
xml, _ = build_bench(g, dt=0.0005, cable="hang", gravity=True)
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
ids = ct.cable_bodies(m, "m_cab"); geom_of = {b: [i for i in range(m.ngeom) if m.geom_bodyid[i] == b][0] for b in ids}
qc, vc = m.joint("m_crank_spin").qposadr[0], m.joint("m_crank_spin").dofadr[0]; vw = m.joint("m_wheel_spin").dofadr[0]
turns, ramp, w, settle = 5.0, 0.5, 2*math.pi/period, 1.0
T_move = turns*period + ramp
mujoco.mj_forward(m, d); ref = None; log = []; next_check = 0.0; states = []
while d.time < settle + T_move + 0.5:
    t = d.time - settle
    if t >= 0 and ref is None: ref = ct.twist_and_bend(m, d, ids)
    if t < 0: ang, rate = 0.0, 0.0
    elif t < ramp: ang, rate = w*t*t/(2*ramp), w*t/ramp
    else: ang, rate = min(w*(t - ramp/2), 2*math.pi*turns), (w if t < T_move - ramp/2 else 0.0)
    d.ctrl[m.actuator("m_steer").id] = np.clip(20*(ang - d.qpos[qc]) + 0.4*(rate - d.qvel[vc]), -2, 2)
    d.ctrl[m.actuator("m_drive").id] = -0.05*d.qvel[vw]
    mujoco.mj_step(m, d)
    if d.time >= next_check:
        next_check += 0.05
        if ref is None: continue
        tw, rb, rates = ct.twist_and_bend(m, d, ids)
        loc = np.degrees(rates - ref[2]); j = int(np.argmax(np.abs(loc)))
        log.append(dict(t=d.time, turns=d.qpos[qc]/(2*math.pi), twist=math.degrees(tw - ref[0]), rb=rb*1000, locmax=float(abs(loc[j])), j=j, loc=loc.copy(),
                        qpos=d.qpos.copy()))
top = sorted(log, key=lambda r: -r["locmax"])[:8]
print("top local twist-rate moments:")
for r in top: print(f"  t={r['t']:6.2f}s turns={r['turns']:5.2f} (phase {r['turns']*360%360:4.0f}°) local {r['locmax']:7.0f} °/m at joint {r['j']}/{len(ids)-1}  total twist {r['twist']:+6.1f}°  min bend R {r['rb']:5.1f} mm  neighbours {np.round(r['loc'][max(0,r['j']-2):r['j']+3]).astype(int)}")
over = [r for r in log if r["locmax"] > 2000]
print(f"checks with local rate > 2000 °/m: {len(over)} of {len(log)} ({100*len(over)/len(log):.1f} % of time); joints involved: {sorted(set(r['j'] for r in over))}")
print("per-turn max local rate:", [round(max(r['locmax'] for r in log if k <= r['turns'] < k+1)) for k in range(5)])
# render the worst moment and a typical moment
rend = mujoco.Renderer(m, rf.H, rf.W); opt = mujoco.MjvOption(); opt.geomgroup[1] = opt.geomgroup[3] = 1
for i in range(m.ngeom):
    if m.geom_bodyid[i] not in ids and m.geom(i).name != "floor": m.geom_rgba[i, 3] = 0.28
def render(r, title, az=35, el=-18, dist=0.5, zoom=None):
    d.qpos[:] = r["qpos"]; mujoco.mj_forward(m, d)
    for jj, b in enumerate(ids):
        v = min(1.0, abs(r["loc"][min(jj, len(r["loc"]) - 1)])/400.0)
        m.geom_rgba[geom_of[b]] = (0.1 + 0.9*v, 0.25*(1 - v), 0.95*(1 - v), 1)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:] = (d.xpos[ids[r["j"]]] if zoom else d.xpos[m.body("m_mount").id] + np.array([0.0, 0.06, 0.06])); cam.distance, cam.azimuth, cam.elevation = (zoom or dist), az, el
    rend.update_scene(d, cam, opt); img = Image.fromarray(rend.render()); dr = ImageDraw.Draw(img)
    x, y = rf.project(m, cam, d.xpos[ids[r["j"]]]); dr.ellipse([x - 10, y - 10, x + 10, y + 10], outline=(220, 30, 30), width=3)
    dr.rectangle((0, 0, rf.W, 34), fill=(22, 27, 34)); dr.text((12, 5), title, font=rf.font(18), fill=(240, 243, 246)); return img
worst = top[0]
p1 = render(worst, f"局所ねじり率が最大の瞬間：{worst['turns']*360%360:.0f}°（{worst['turns']:.2f}周目） {worst['locmax']:.0f} °/m 節{worst['j']}")
p2 = render(worst, f"同じ瞬間の拡大（赤丸＝最大の節）", zoom=0.12)
typical = max((r for r in log if 1.0 <= r["turns"] <= 2.0), key=lambda r: abs(r["twist"]))
p3 = render(typical, f"比較：2周目のねじれ最大 {typical['turns']*360%360:.0f}°  局所最大 {typical['locmax']:.0f} °/m 節{typical['j']}")
canvas = Image.new("RGB", (rf.W*3, rf.H), (255, 255, 255))
for k, im in enumerate((p1, p2, p3)): canvas.paste(im, (k*rf.W, 0))
canvas.save(out); print(out)
