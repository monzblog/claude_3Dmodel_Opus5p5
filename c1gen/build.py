"""首都高C1モデルの生成。

    python -m c1gen.build --fetch          # OSM と標高を取得してから生成
    python -m c1gen.build                  # 取得済みデータから生成
    python -m c1gen.build --synthetic      # 仮データ（楕円の周回）で動作確認
"""
import argparse
import json
import os

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from . import config as C
from . import dem as dem_mod
from . import geom as G
from . import osm
from .proj import to_latlon
from .route import Route, loop_profile, ramp_profile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
FONT = os.path.join(DATA, "fonts", "ipaexg.ttf")

DIR_JA = {"Outer": "外回り", "Inner": "内回り"}


class Layer(dict):
    """オブジェクト名 -> (Mesh, 材質名)。"""

    def mesh(self, name, mat):
        if name not in self:
            self[name] = (G.Mesh(), mat)
        return self[name][0]


def _union_buffer(lines):
    polys = [LineString(p).buffer(w) for p, w in lines if len(p) >= 2]
    return shapely.union_all(polys) if polys else None


def _inside(poly, xy):
    if poly is None:
        return np.zeros(len(xy), bool)
    return shapely.contains_xy(poly, xy[:, 0], xy[:, 1])


def _deck_polygon(route, extra=C.BARRIER_BASE):
    """中心線を車線数の同じ区間ごとにバッファして道路の平面形を作る。"""
    pieces = []
    P = np.vstack([route.P, route.P[:1]]) if route.closed else route.P
    w = np.append(route.width, route.width[0]) if route.closed else route.width
    bounds = np.flatnonzero(np.diff(np.round(w, 1)) != 0)
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds + 1, [len(P) - 1]])
    for a, b in zip(starts, ends):
        pieces.append((P[a : b + 1], float(w[a:b + 1].max() / 2 + extra)))
    return _union_buffer(pieces)


def make_routes(loops_raw, ramps_raw, dem):
    loops = []
    for lp in loops_raw:
        r = Route(lp["name"], lp["pts"], lp["attrs"], closed=True)
        loop_profile(r, dem)
        loops.append(r)
    ramps = []
    counts = {}
    for rr in ramps_raw:
        if len(rr["pts"]) < 2:
            continue
        main = loops[rr["loop"]]
        key = (main.name, rr["kind"])
        counts[key] = counts.get(key, 0) + 1
        r = Route(f"{main.name}_{rr['kind']}_{counts[key]:02d}", rr["pts"], rr["attrs"], closed=False)
        if r.length < 10:
            continue
        j = int(np.argmin(np.linalg.norm(main.P - r.P[0], axis=1)))
        ramp_profile(r, dem, main.z[j])
        k = min(len(r.P) - 1, int(60 / r.step))
        r.kind = rr["kind"]
        r.main = main
        r.junction = j
        r.side = 1 if np.dot(r.P[k] - main.P[j], main.N[j]) >= 0 else -1
        r.destination = rr["destination"]
        ramps.append(r)
    return loops, ramps


def build_scene(loops, ramps):
    layers = {}
    texts = []
    main_poly = {id(r): _deck_polygon(r) for r in loops}

    for r in loops:
        L = layers.setdefault(f"C1_{r.name}", Layer())
        mine = [q for q in ramps if q.main is r]
        ramp_poly = _union_buffer([(q.P[: int(150 / q.step)], float(q.width.max() / 2)) for q in mine])
        allseg = np.ones(len(r.s), bool)
        G.deck(L.mesh(f"C1_{r.name}_Road", "Asphalt"), r, allseg)
        G.markings(L.mesh(f"C1_{r.name}_Markings", "Marking"), r, allseg)
        removed = {}
        for side in (1, -1):
            bp = r.P + r.N * (side * (G.half_width(r) + C.BARRIER_BASE / 2))[:, None]
            gap = _inside(ramp_poly, bp)
            removed[side] = gap
            G.barrier(L.mesh(f"C1_{r.name}_Barriers", "Concrete"), r, side, G.seg(~gap, True))
            G.retaining_wall(L.mesh(f"C1_{r.name}_RetainingWalls", "Concrete"), r, side,
                             G.seg(r.cutting & ~gap, True))
        G.deck_body(L.mesh(f"C1_{r.name}_Viaduct", "StructureConcrete"), r,
                    G.seg(r.z - r.ground > 1.5, True))
        G.piers(L.mesh(f"C1_{r.name}_Piers", "StructureConcrete"), r)
        G.tunnel_shell(L.mesh(f"C1_{r.name}_Tunnel", "TunnelWall"), r, G.seg(r.tunnel, True))
        G.portals(L.mesh(f"C1_{r.name}_Tunnel_Portals", "StructureConcrete"), r, r.tunnel)
        G.tunnel_lights(L.mesh(f"C1_{r.name}_TunnelLights", "TunnelLight"), r)
        G.light_poles(L.mesh(f"C1_{r.name}_LightPoles", "Metal"), L.mesh(f"C1_{r.name}_Lamps", "LampLight"),
                      r, skip=removed[1])
        _signs(L, texts, r, mine)

    R = layers.setdefault("C1_Ramps", Layer())
    for q in ramps:
        near = q.s < 150
        in_main = _inside(main_poly[id(q.main)], q.P) & near
        zoff = np.where(in_main, -0.03, 0.0)
        allseg = np.ones(len(q.s) - 1, bool)
        G.deck(R.mesh("C1_Ramps_Road", "Asphalt"), q, allseg, zoff=zoff)
        G.markings(R.mesh("C1_Ramps_Markings", "Marking"), q, G.seg(~in_main, False), zoff=zoff)
        for side in (1, -1):
            bp = q.P + q.N * (side * (G.half_width(q) + C.BARRIER_BASE / 2))[:, None]
            ok = ~(_inside(main_poly[id(q.main)], bp) & near)
            G.barrier(R.mesh("C1_Ramps_Barriers", "Concrete"), q, side, G.seg(ok, False), zoff=zoff)
            G.retaining_wall(R.mesh("C1_Ramps_RetainingWalls", "Concrete"), q, side, G.seg(q.cutting & ok, False))
        G.deck_body(R.mesh("C1_Ramps_Viaduct", "StructureConcrete"), q,
                    G.seg((q.z - q.ground > 1.5) & ~in_main, False), zoff=zoff)
        G.piers(R.mesh("C1_Ramps_Piers", "StructureConcrete"), q)
        G.tunnel_shell(R.mesh("C1_Ramps_Tunnel", "TunnelWall"), q, G.seg(q.tunnel & ~in_main, False))
        G.portals(R.mesh("C1_Ramps_Tunnel_Portals", "StructureConcrete"), q, q.tunnel & ~in_main)
        G.tunnel_lights(R.mesh("C1_Ramps_TunnelLights", "TunnelLight"), q)
    return layers, texts


def _signs(L, texts, r, ramps):
    frames = L.mesh(f"C1_{r.name}_SignFrames", "Metal")
    panels = L.mesh(f"C1_{r.name}_SignPanels", "SignGreen")
    n = len(r.s)
    for q in ramps:
        dest = q.destination or "出口"
        if q.kind == "diverge":
            for dist, extra in ((C.SIGN_ADVANCE, f"{int(C.SIGN_ADVANCE)}m"), (40.0, "出口" if q.side > 0 else "右 出口")):
                i = (q.junction - int(dist / r.step)) % n
                if not r.tunnel[i]:
                    G.gantry(frames, panels, texts, r, i, [dest, extra], q.side)
        else:
            i = (q.junction - int(120 / r.step)) % n
            G.warning_sign(frames, L.mesh(f"C1_{r.name}_SignYellow", "SignYellow"), texts, r, i, q.side, "合流")
    every = int(C.SPEED_SIGN_SPACING / r.step)
    for i in range(every // 2, n, every):
        G.speed_sign(frames, L.mesh(f"C1_{r.name}_SignRed", "SignRed"), L.mesh(f"C1_{r.name}_SignWhite", "SignWhite"),
                     texts, r, i, C.SPEED_LIMIT)


def terrain(routes, dem):
    """周回の範囲の地形。路面が地面近く・地下にある所は、地面を路面より下へ押し下げる。"""
    allp = np.vstack([r.P for r in routes])
    x0, y0 = allp.min(0) - C.TERRAIN_MARGIN
    x1, y1 = allp.max(0) + C.TERRAIN_MARGIN
    xs = np.arange(x0, x1 + C.TERRAIN_STEP, C.TERRAIN_STEP)
    ys = np.arange(y0, y1 + C.TERRAIN_STEP, C.TERRAIN_STEP)
    X, Y = np.meshgrid(xs, ys)
    grid = np.column_stack([X.ravel(), Y.ravel()])
    lat, lon = to_latlon(grid[:, 0], grid[:, 1])
    Z = dem.sample(lat, lon) - 0.05
    cuts, pts, zs = [], [], []
    for r in routes:
        low = (r.z < r.ground + 1.5) & ~r.tunnel
        for rows in G.runs(G.seg(low, r.closed), r.closed):
            # 格子1.5マス分広く下げ、斜めの面が路面に被らないようにする
            cuts.append((r.P[rows], float(r.width.max() / 2 + C.BARRIER_BASE + 1.5 * C.TERRAIN_STEP)))
        pts.append(r.P[low])
        zs.append(r.z[low])
    hole = _union_buffer(cuts)
    if hole is not None:
        inside = _inside(hole, grid)
        tree = cKDTree(np.vstack(pts))
        _, k = tree.query(grid[inside])
        Z[inside] = np.minimum(Z[inside], np.concatenate(zs)[k] - 1.0)
    ny, nx = X.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    faces = np.stack([a, b, c, d], -1).reshape(-1, 4).tolist()
    m = G.Mesh()
    m.add(np.column_stack([grid, Z]), faces)
    return m


def lanes_json(routes, path):
    out = {"crs": C.CRS, "origin_latlon": C.ORIGIN_LATLON, "note": "lane 1 = 一番左の車線。座標は原点からのメートル (X=東, Y=北, Z=標高)", "routes": []}
    for r in routes:
        lanes = []
        for pts in G.lane_centers(r):
            if getattr(r, "kind", None) == "merge":
                pts = pts[::-1]
            lanes.append([None if np.isnan(p[0]) else [round(float(v), 2) for v in p] for p in pts])
        out["routes"].append({"name": r.name, "closed": r.closed, "length_m": round(r.length, 1), "lanes": lanes})
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="OSM と標高タイルを取得する")
    ap.add_argument("--synthetic", action="store_true", help="仮データで生成する")
    ap.add_argument("--out", default=os.path.join(OUT, "C1_loop.blend"))
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.synthetic:
        from . import synthetic
        osm_path = os.path.join(DATA, "synthetic_osm.json")
        synthetic.write(osm_path)
        dem = dem_mod.DEM(os.path.join(DATA, "no_dem"))
    else:
        osm_path = os.path.join(DATA, "osm_c1.json")
        dem_root = os.path.join(DATA, "dem")
        if args.fetch:
            print("OSM を取得中…")
            osm.fetch(osm_path)
            print("標高タイルを取得中…")
            dem_mod.fetch(dem_root)
        dem = dem_mod.DEM(dem_root)
        if not dem.available:
            print("警告: 標高データが無いので地面を一定の高さとして作ります")

    loops_raw, ramps_raw = osm.load(osm_path)
    if len(loops_raw) != 2:
        print(f"警告: 周回が {len(loops_raw)} 本見つかりました（内回り・外回りの 2 本を想定）")
    loops, ramps = make_routes(loops_raw, ramps_raw, dem)
    for r in loops:
        print(f"{DIR_JA.get(r.name, r.name)}: {r.length / 1000:.2f} km, 高架 {r.elevated.mean():.0%}, "
              f"トンネル {r.tunnel.mean():.0%}, 掘割 {r.cutting.mean():.0%}")
    print(f"分岐 {sum(q.kind == 'diverge' for q in ramps)} 本, 合流 {sum(q.kind == 'merge' for q in ramps)} 本")

    layers, texts = build_scene(loops, ramps)
    ground = terrain(loops + ramps, dem)
    lanes_json(loops + ramps, os.path.splitext(args.out)[0] + "_lanes.json")

    from . import blend
    blend.reset()
    for coll, objs in layers.items():
        for name, (mesh, mat) in objs.items():
            blend.add_object(coll, name, mesh, mat)
    for mat, mesh in blend.text_meshes(texts, FONT).items():
        blend.add_object("C1_Signs_Text", f"C1_SignText_{mat}", mesh, mat)
    blend.add_object("Terrain", "Terrain", ground, "Ground")
    blend.save(args.out, {"crs": C.CRS, "origin_lat": C.ORIGIN_LATLON[0], "origin_lon": C.ORIGIN_LATLON[1]})
    print(f"保存しました: {args.out}")


if __name__ == "__main__":
    main()
