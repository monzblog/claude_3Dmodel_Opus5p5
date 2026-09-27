"""背景（ビル・空・地面）を足して、運転席から見たドライブ動画を書き出す。

シミュレーター用の output/C1_loop.blend は変えず、背景を足したものを output/C1_showcase.blend に保存する。
先に python -m c1gen.build で C1_loop.blend を作っておくこと。

    python -m c1gen.video                                  # 縦長 1080x1920、30fps、約40秒（EEVEE）
    python -m c1gen.video --engine workbench --scale 0.4   # 軽い確認用
    python -m c1gen.video --stills 0,400,800               # 決めたコマだけ静止画で確認
"""
import argparse
import math
import os

import bpy
import numpy as np
from mathutils import Quaternion, Vector

from . import blend
from . import config as C
from . import dem as dem_mod
from . import geom as G
from . import osm
from . import plateau as PL
from . import scenery
from .build import DATA, OUT, make_routes


def routes():
    dem = dem_mod.DEM(os.path.join(DATA, "dem"))
    pl = PL.Plateau.load(os.path.join(DATA, "plateau_c1.npz"))
    loops, ramps = make_routes(*osm.load(os.path.join(DATA, "osm_c1.json")), dem, pl)
    return loops, ramps, dem


def add_scenery(loops, ramps, dem):
    terrain = bpy.data.objects.get("Terrain")
    if terrain is not None:
        terrain.data.materials.clear()
        terrain.data.materials.append(blend.material("CityGround"))
    bldgs = scenery.load_buildings()
    if not bldgs:
        print("警告: 建物のデータ（data/buildings_c1.npz）が無いので、ビル無しで描きます")
        return
    walls, roofs, skipped = scenery.building_meshes(bldgs, dem, scenery.corridor(loops + ramps))
    for name, mesh in walls.items():
        blend.add_object("Scenery", f"Buildings_{name}", mesh, name)
    blend.add_object("Scenery", "Buildings_Roof", roofs, "Roof")
    print(f"ビル {len(bldgs) - skipped} 棟（道路に掛かる {skipped} 棟は除外）")


def sky(scene, engine):
    world = bpy.data.worlds.new("Sky")
    scene.world = world
    world.color = (0.55, 0.68, 0.85)
    if engine == "workbench":
        return
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    try:
        tex = nt.nodes.new("ShaderNodeTexSky")
        for t in ("NISHITA", "MULTIPLE_SCATTERING", "SINGLE_SCATTERING"):
            try:
                tex.sky_type = t
                break
            except TypeError:
                continue
        if hasattr(tex, "sun_elevation"):
            tex.sun_elevation = math.radians(35)
            tex.sun_rotation = math.radians(150)
        nt.links.new(tex.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 0.35
    except Exception as e:  # 空のノードが無い版でも単色の空で描ける
        print("空のテクスチャが使えないので単色にします:", e)
        bg.inputs["Color"].default_value = (0.55, 0.68, 0.85, 1)
    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = 4.0
    sun.angle = math.radians(1.0)
    ob = bpy.data.objects.new("Sun", sun)
    ob.rotation_euler = (math.radians(55), 0, math.radians(150))
    scene.collection.objects.link(ob)


def camera_path(r, start_m, seconds, speed_kmh, fps, lane, height=1.2, ahead=25.0):
    """車線の中心を一定の速さで走るカメラの位置と向き（コマごと）。"""
    centers = G.lane_centers(r)[min(lane, len(G.lane_centers(r))) - 1]
    # 車線が無い所（車線数が減る所）は第1車線に寄せる
    first = G.lane_centers(r)[0]
    centers = np.where(np.isnan(centers), first, centers)
    L = r.length
    n = int(seconds * fps)
    s = start_m + np.arange(n) * speed_kmh / 3.6 / fps

    def at(sv):
        sv = np.mod(sv, L) if r.closed else np.clip(sv, 0, r.s[-1])
        idx = sv / r.step
        i0 = np.floor(idx).astype(int) % len(r.s)
        i1 = (i0 + 1) % len(r.s)
        f = (idx - np.floor(idx))[:, None]
        return centers[i0] * (1 - f) + centers[i1] * f, r.bank[i0] * (1 - f[:, 0]) + r.bank[i1] * f[:, 0]

    eye, bank = at(s)
    tgt, _ = at(s + ahead)
    eye[:, 2] += height
    tgt[:, 2] += height * 0.8
    return eye, tgt, bank


def animate(cam, eye, tgt, bank, roll_gain=0.6):
    cam.rotation_mode = "QUATERNION"
    prev = None
    for f, (p, q, b) in enumerate(zip(eye, tgt, bank)):
        d = Vector(q) - Vector(p)
        rot = d.to_track_quat("-Z", "Y")
        # 片勾配で車が傾いた分だけ、少し視界も傾ける（左が上がる向きが正）
        rot = rot @ Quaternion((0, 0, 1), -roll_gain * math.atan(b))
        if prev is not None and rot.dot(prev) < 0:
            rot.negate()
        prev = rot
        cam.location = Vector(p)
        cam.rotation_quaternion = rot
        cam.keyframe_insert("location", frame=f + 1)
        cam.keyframe_insert("rotation_quaternion", frame=f + 1)


def render_settings(scene, engine, res, scale, fps, frames, out, samples=8):
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = int(scale * 100)
    scene.render.fps = fps
    scene.frame_start, scene.frame_end = 1, frames
    if engine == "workbench":
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "TEXTURE"
        scene.display.shading.show_shadows = True
    elif engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = max(samples, 16)
        scene.cycles.use_denoising = True
    else:
        scene.render.engine = "BLENDER_EEVEE"
        if hasattr(scene, "eevee"):
            # 動くカメラでは少ないサンプルでも目立たない。1 コマの時間はほぼサンプル数に比例する
            scene.eevee.taa_render_samples = samples
            for attr in ("use_raytracing", "use_gtao"):
                if hasattr(scene.eevee, attr):
                    setattr(scene.eevee, attr, False)
    scene.render.filepath = out
    ims = scene.render.image_settings
    if hasattr(ims, "media_type"):
        ims.media_type = "VIDEO"
    ims.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", default="Inner", help="Inner（内回り）か Outer（外回り）")
    ap.add_argument("--start", type=float, default=8.85, help="走り始める位置（周回の起点からの km）")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--speed", type=float, default=70.0, help="km/h")
    ap.add_argument("--lane", type=int, default=2, help="1 = 一番左の車線")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", default="1080x1920")
    ap.add_argument("--scale", type=float, default=1.0, help="解像度の倍率（確認用に小さくする）")
    ap.add_argument("--engine", default="eevee", choices=("eevee", "cycles", "workbench"))
    ap.add_argument("--samples", type=int, default=8, help="EEVEE のサンプル数（多いほどきれいで遅い）")
    ap.add_argument("--stills", default="", help="この番号のコマだけ PNG で描く（例: 0,400,800）")
    ap.add_argument("--blend", default=os.path.join(OUT, "C1_loop.blend"))
    ap.add_argument("--out", default=os.path.join(OUT, "C1_drive.mp4"))
    args = ap.parse_args()

    loops, ramps, dem = routes()
    r = next(x for x in loops if x.name == args.loop)
    bpy.ops.wm.open_mainfile(filepath=args.blend)
    scene = bpy.context.scene
    add_scenery(loops, ramps, dem)
    sky(scene, args.engine)

    cd = bpy.data.cameras.new("DriveCam")
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = 24.0
    cd.lens = 14.0
    cd.clip_start, cd.clip_end = 0.1, 6000
    cam = bpy.data.objects.new("DriveCam", cd)
    scene.collection.objects.link(cam)
    scene.camera = cam
    eye, tgt, bank = camera_path(r, args.start * 1000, args.seconds, args.speed, args.fps, args.lane)
    animate(cam, eye, tgt, bank)

    res = tuple(int(v) for v in args.res.split("x"))
    render_settings(scene, args.engine, res, args.scale, args.fps, len(eye), args.out, args.samples)
    showcase = os.path.join(os.path.dirname(args.out), "C1_showcase.blend")
    bpy.ops.wm.save_as_mainfile(filepath=showcase, compress=True)
    print(f"保存しました: {showcase}")

    if args.stills:
        base = os.path.splitext(args.out)[0]
        if hasattr(scene.render.image_settings, "media_type"):
            scene.render.image_settings.media_type = "IMAGE"
        scene.render.image_settings.file_format = "PNG"
        for f in (int(v) for v in args.stills.split(",")):
            scene.frame_set(f + 1)
            scene.render.filepath = f"{base}_{f:04d}.png"
            bpy.ops.render.render(write_still=True)
        print(f"静止画を書き出しました: {base}_*.png")
        return
    bpy.ops.render.render(animation=True)
    print(f"動画を書き出しました: {args.out}")


if __name__ == "__main__":
    main()
