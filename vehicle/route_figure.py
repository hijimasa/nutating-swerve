"""Figure of the v4 cable route: free loops (blue), runs tied to a part (orange), brackets
(green); everything else translucent.  Writes results/cable_route.png."""
from __future__ import annotations
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_quad import build_bench, FIXED_CABLE_RGBA
from geometry import ModuleGeometry
import cable_test as ct

W, H = 760, 620
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def font(size):
    try:
        return ImageFont.truetype(FONT, size, index=0)
    except OSError:
        return ImageFont.load_default()


def project(m, cam, p):
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    f = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
    pos = np.array(cam.lookat) - cam.distance*f
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    v = np.asarray(p) - pos
    foc = (H/2)/math.tan(math.radians(m.vis.global_.fovy)/2)
    return W/2 + foc*(v @ r)/(v @ f), H/2 - foc*(v @ u)/(v @ f)


def main(out="results/cable_route.png"):
    g = ModuleGeometry()
    xml, _ = build_bench(g, dt=0.0005, cable="hook", gravity=True)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    cab = set(ct.cable_bodies(m, "m_cabA") + ct.cable_bodies(m, "m_cabB"))
    orange = np.array([float(x) for x in FIXED_CABLE_RGBA.split()])
    for i in range(m.ngeom):
        nm = m.geom(i).name
        if m.geom_bodyid[i] in cab:
            m.geom_rgba[i] = (0.1, 0.25, 0.95, 1)
        elif np.allclose(m.geom_rgba[i], orange) or "hook" in nm or "bridge" in nm or "clamp" in nm:
            continue
        elif nm == "floor":
            continue
        else:
            m.geom_rgba[i, 3] = 0.22
    qc, vc = m.joint("m_crank_spin").qposadr[0], m.joint("m_crank_spin").dofadr[0]
    rend = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = opt.geomgroup[3] = 1
    panels = []

    def step_to(target_phi, T):
        while d.time < T:
            tau = np.clip(20*(target_phi - d.qpos[qc]) - 0.4*d.qvel[vc], -2, 2)
            d.ctrl[m.actuator("m_steer").id] = tau
            mujoco.mj_step(m, d)

    def shot(az, el, title):
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
        cam.lookat[:] = d.xpos[m.body("m_mount").id] + np.array([0.01, 0.01, 0.025])
        cam.distance, cam.azimuth, cam.elevation = 0.36, az, el
        rend.update_scene(d, cam, opt)
        img = Image.fromarray(rend.render())
        dr = ImageDraw.Draw(img)
        mid = lambda name: d.xpos[ct.cable_bodies(m, name)[len(ct.cable_bodies(m, name))//2]]
        labels = [(mid("m_cabA"), "ループA（内側軸 ±45°）", (0.1, 0.25, 0.95)),
                  (mid("m_cabB"), "ループB（外側軸 ±45°）", (0.1, 0.25, 0.95)),
                  (d.geom("m_hookH_h").xpos, "フック金具（ハウジング）", (0.15, 0.55, 0.25)),
                  (d.geom("m_bridge_h").xpos, "ブリッジ金具（リング）", (0.15, 0.55, 0.25)),
                  (d.geom("m_clampR").xpos, "リング側の留め具", (0.3, 0.3, 0.3)),
                  (d.geom("m_clampF").xpos, "フレーム側の留め具", (0.3, 0.3, 0.3))]
        f = font(17)
        placed = []
        for p, txt, col in labels:
            x, y = project(m, cam, p)
            if not (0 <= x < W and 0 <= y < H):
                continue
            tx, ty = x + 40, y - 40
            for (px, py) in placed:
                if abs(ty - py) < 26 and abs(tx - px) < 220:
                    ty = py + 28
            tx = min(tx, W - 230)
            placed.append((tx, ty))
            c = tuple(int(255*v) for v in col)
            dr.line([(x, y), (tx, ty + 11)], fill=c, width=2)
            dr.ellipse([x - 4, y - 4, x + 4, y + 4], outline=c, width=2)
            bb = dr.textbbox((tx, ty), txt, font=f)
            dr.rectangle((bb[0] - 4, bb[1] - 2, bb[2] + 4, bb[3] + 2), fill=(255, 255, 255))
            dr.text((tx, ty), txt, font=f, fill=c)
        dr.rectangle((0, 0, W, 34), fill=(22, 27, 34))
        dr.text((12, 5), title, font=font(18), fill=(240, 243, 246))
        return img

    step_to(0.0, 1.0)                                    # settle (cable rest shape = straight)
    panels.append(shot(55, -18, "操舵角 0°：手前（+y 側）から"))
    panels.append(shot(150, -22, "操舵角 0°：+x 側から"))
    step_to(math.pi, 2.5)
    panels.append(shot(55, -18, "操舵角 180°：内側ジンバルが反対側に振れた状態"))
    canvas = Image.new("RGB", (W*3, H + 64), (255, 255, 255))
    for k, im in enumerate(panels):
        canvas.paste(im, (k*W, 0))
    dr = ImageDraw.Draw(canvas)
    f = font(19)
    x = 16
    for col, txt in [((26, 64, 242), "青：自由に動く区間（U字ループ2つ）"),
                     ((255, 128, 26), "橙：部品に結束して動かない区間"),
                     ((51, 184, 89), "緑：外付け曲げ金具（フック・ブリッジ）"),
                     ((120, 120, 120), "半透明：その他の部品")]:
        dr.rectangle((x, H + 22, x + 22, H + 42), fill=col)
        dr.text((x + 30, H + 18), txt, font=f, fill=(30, 30, 30))
        x += 30 + dr.textlength(txt, font=f) + 36
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    rend.close()
    print(out)


if __name__ == "__main__":
    main()
