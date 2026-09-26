"""生成した .blend から確認用の画像を描く（Workbench）。

    python tools/preview.py output/C1_loop.blend output/preview
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector


def look_at(cam, eye, target):
    cam.location = Vector(eye)
    d = Vector(target) - Vector(eye)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def main(blend, outdir, lanes_path=None):
    bpy.ops.wm.open_mainfile(filepath=blend)
    os.makedirs(outdir, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    cam_data = bpy.data.cameras.new("cam")
    cam_data.clip_end = 20000
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    lanes_path = lanes_path or os.path.splitext(blend)[0] + "_lanes.json"
    with open(lanes_path) as f:
        routes = json.load(f)["routes"]
    # 俯瞰は周回全体が入るよう、全ルートの範囲の中心を狙う
    allp = np.array([p for r in routes for lane in r["lanes"] for p in lane if p])
    c = (allp.min(0) + allp.max(0)) / 2
    span = float((allp.max(0) - allp.min(0))[:2].max())
    views = [("overview", (c[0], c[1] - span * 0.6, span * 2.0), (c[0], c[1], 0), 50)]
    for r in routes:
        lane = [p for p in r["lanes"][0] if p]
        n = len(lane)
        for k, frac in enumerate((0.05, 0.3, 0.55, 0.8) if r["closed"] else (0.5,)):
            i = int(n * frac)
            p = np.array(lane[i])
            q = np.array(lane[min(i + 15, n - 1)])
            eye = p + [0, 0, 1.2]
            views.append((f"{r['name']}_{k}", tuple(eye), tuple(q + [0, 0, 1.0]), 70))
            views.append((f"{r['name']}_{k}_aerial", tuple(p + [0, 0, 60] - (q - p) * 4), tuple(q), 50))
    for name, eye, target, lens in views:
        cam_data.lens = 35 * 50 / lens / 1.0 if lens != 50 else 35
        cam_data.angle = math.radians(lens)
        look_at(cam, eye, target)
        scene.render.filepath = os.path.join(outdir, f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print("wrote", scene.render.filepath)


if __name__ == "__main__":
    main(*sys.argv[1:])
