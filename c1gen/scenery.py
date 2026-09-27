"""紹介動画用の背景（周りのビル・地面・空）。シミュレーター用の .blend には入れない。

ビルは OSM の建物の形と高さ（height / building:levels）から、C1 の近くだけを押し出して作る。
取得した建物は data/buildings_c1.npz に小さくまとめる（リポジトリに入れられる大きさ）。

    python -m c1gen.scenery --fetch      # OSM から建物を取得して data/buildings_c1.npz を作る
"""
import argparse
import json
import os
import re
import zlib

import numpy as np
import shapely
from shapely.geometry import Polygon

from . import config as C
from . import geom as G
from . import osm
from .proj import to_xy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
BUILDINGS = os.path.join(DATA, "buildings_c1.npz")

FACADES = ("FacadeGlass", "FacadeBeige", "FacadeGray", "FacadeDark")


def _num(text):
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", str(text or ""))
    return float(m.group(1)) if m else None


def _height(tags, area, key):
    """建物の高さ。タグが無ければ、面積から都心のビルらしい高さを決める（id で毎回同じ値）。"""
    h = _num(tags.get("height"))
    if h:
        return h
    lv = _num(tags.get("building:levels"))
    if lv:
        return lv * C.FLOOR_HEIGHT + 1.0
    r = (zlib.crc32(str(key).encode()) % 1000) / 1000.0
    if area < 150:
        return 6.0 + 6.0 * r
    if area < 600:
        return 12.0 + 18.0 * r
    return 20.0 + 45.0 * r


def fetch_buildings(out=BUILDINGS, osm_path=os.path.join(DATA, "osm_c1.json"), radius=C.BUILDING_RADIUS):
    """OSM の建物を取得し、C1（周回とランプ）から radius 以内のものだけを npz にまとめる。"""
    tmp = out + ".json"
    osm._query(tmp, 'way["building"]({bbox});', C.BBOX)
    with open(tmp) as f:
        data = json.load(f)
    loops, ramps = osm.load(osm_path)
    line = shapely.union_all([shapely.LineString(p["pts"]) for p in loops + ramps if len(p["pts"]) >= 2])
    near = line.buffer(radius)
    xy, start, height = [], [0], []
    for el in data["elements"]:
        g = el.get("geometry")
        if el.get("type") != "way" or not g or len(g) < 4:
            continue
        x, y = to_xy([p["lat"] for p in g], [p["lon"] for p in g])
        pts = np.column_stack([x, y])[:-1]
        poly = Polygon(pts)
        if not poly.is_valid or poly.area < 20 or not near.intersects(poly):
            continue
        xy.append(pts)
        start.append(start[-1] + len(pts))
        height.append(_height(el.get("tags", {}), poly.area, el["id"]))
    np.savez_compressed(out, xy=np.vstack(xy).astype(np.float32), start=np.array(start, np.int32),
                        height=np.array(height, np.float32))
    os.remove(tmp)
    return len(height)


def load_buildings(path=BUILDINGS):
    if not os.path.exists(path):
        return []
    d = np.load(path)
    xy, st, h = d["xy"].astype(float), d["start"], d["height"].astype(float)
    return [(xy[a:b], float(hh)) for a, b, hh in zip(st[:-1], st[1:], h)]


def corridor(routes, margin=2.0):
    """地上に出ている路面（高架・地平・掘割）の範囲。ここに掛かる建物は置かない。"""
    pieces = []
    for r in routes:
        open_ = ~r.tunnel
        for rows in G.runs(G.seg(open_, r.closed), r.closed):
            if len(rows) >= 2:
                pieces.append(shapely.LineString(r.P[rows]).buffer(float(r.width.max() / 2 + C.BARRIER_BASE + margin)))
    return shapely.union_all(pieces) if pieces else None


def building_meshes(bldgs, dem, keep_out=None):
    """建物を押し出したメッシュ。外壁は外観の種類ごと、屋上は 1 つにまとめて返す。"""
    from .proj import to_latlon

    walls = {name: G.Mesh() for name in FACADES}
    roofs = G.Mesh()
    skipped = 0
    for n, (pts, h) in enumerate(bldgs):
        poly = Polygon(pts)
        if keep_out is not None and keep_out.intersects(poly):
            skipped += 1
            continue
        if poly.exterior.is_ccw is False:
            pts = pts[::-1]
        lat, lon = to_latlon(pts[:, 0], pts[:, 1])
        ground = dem.sample(lat, lon)
        base = float(ground.min()) - 0.5
        top = float(ground.max()) + h
        k = len(pts)
        seglen = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
        u = np.concatenate([[0], np.cumsum(seglen)[:-1]])
        # 周の最後の辺の UV が折り返さないよう、壁は辺ごとに頂点を分ける
        wv, wf, wuv = [], [], []
        for i in range(k):
            j = (i + 1) % k
            a, b = pts[i], pts[j]
            o = len(wv)
            wv += [(*a, base), (*b, base), (*b, top), (*a, top)]
            wuv += [(u[i], 0.0), (u[i] + seglen[i], 0.0), (u[i] + seglen[i], top - base), (u[i], top - base)]
            wf.append([o, o + 1, o + 2, o + 3])
        walls[FACADES[zlib.crc32(pts.tobytes()) % len(FACADES)]].add(np.array(wv), wf, np.array(wuv))
        roofs.add(np.column_stack([pts, np.full(k, top)]), [list(range(k))])
    return walls, roofs, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="OSM から建物を取得する")
    args = ap.parse_args()
    if args.fetch:
        n = fetch_buildings()
        print(f"建物 {n} 棟を {BUILDINGS} に保存しました（{os.path.getsize(BUILDINGS) / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
