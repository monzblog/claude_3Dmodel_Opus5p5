"""PLATEAU 首都高速道路（2023年度）の CityGML から、C1 の路面・橋・トンネルの面を軽い npz に抜き出す。

    python tools/plateau_extract.py [data/plateau] [data/plateau_c1.npz]

出力（座標は平面直角座標系 IX 系、原点は config.ORIGIN_LATLON、Z は T.P. 標高）:
    verts      float32 (N,3)  cm 単位で共有した頂点
    tris       int32   (M,3)
    kind       uint8   (M)    KINDS の番号
    obj        int32   (M)    元の地物（または部品）の番号
    obj_gmlid  str     (K)    obj 番号に対応する gml:id
"""
import glob
import os
import sys
from xml.etree.ElementTree import iterparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from c1gen.proj import to_xy  # noqa: E402

KINDS = {
    1: "road", 2: "island", 3: "bridge_floor", 4: "bridge_ceiling", 5: "bridge_side", 6: "bridge_ground",
    7: "bridge_installation", 8: "pier", 9: "tunnel_ground", 10: "tunnel_wall", 11: "tunnel_roof",
    12: "tunnel_closure", 13: "tunnel_installation",
}
# (モジュール, 要素名) -> 種類。TrafficArea / AuxiliaryTrafficArea は function で決める
PARTS = {
    ("brid", "OuterFloorSurface"): 3, ("brid", "OuterCeilingSurface"): 4, ("brid", "WallSurface"): 5,
    ("brid", "GroundSurface"): 6, ("brid", "BridgeInstallation"): 7, ("brid", "BridgeConstructionElement"): 8,
    ("tun", "GroundSurface"): 9, ("tun", "WallSurface"): 10, ("tun", "RoofSurface"): 11,
    ("tun", "ClosureSurface"): 12, ("tun", "TunnelInstallation"): 13,
    ("tran", "TrafficArea"): None, ("tran", "AuxiliaryTrafficArea"): None,
}
GML_ID = "{http://www.opengis.net/gml}id"


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def read_polygons(path, module):
    """(種類, gml:id, 緯度経度高さの配列) を順に返す。"""
    stack = []  # (要素名, gml:id)
    func = {}   # gml:id -> function の値
    for ev, el in iterparse(path, events=("start", "end")):
        t = _local(el.tag)
        if ev == "start":
            stack.append((t, el.attrib.get(GML_ID, "")))
            continue
        stack.pop()
        if t == "function" and stack and stack[-1][0] in ("TrafficArea", "AuxiliaryTrafficArea"):
            func[stack[-1][1]] = (el.text or "").strip()
        elif t == "posList":
            part = next(((s, i) for s, i in reversed(stack) if (module, s) in PARTS), None)
            if part is not None:
                kind = PARTS[(module, part[0])]
                if kind is None:
                    kind = 2 if func.get(part[1]) == "3000" else 1
                yield kind, part[1], np.array(el.text.split(), float).reshape(-1, 3)
        if t in ("Polygon", "cityObjectMember"):
            el.clear()


def extract(root):
    rings, kinds, objs, ids = [], [], [], {}
    for module in ("tran", "brid", "tun"):
        for path in sorted(glob.glob(os.path.join(root, "**", "udx", module, "*.gml"), recursive=True)):
            for kind, gid, ll in read_polygons(path, module):
                rings.append(ll)
                kinds.append(kind)
                objs.append(ids.setdefault(gid, len(ids)))
    lat = np.concatenate([r[:, 0] for r in rings])
    lon = np.concatenate([r[:, 1] for r in rings])
    x, y = to_xy(lat, lon)
    xyz = np.column_stack([x, y, np.concatenate([r[:, 2] for r in rings])])
    # cm 単位で頂点を共有する
    key = np.round(xyz * 100).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    verts = (uniq / 100.0).astype(np.float32)
    tris, tkind, tobj = [], [], []
    k = 0
    for ring, kind, obj in zip(rings, kinds, objs):
        idx = inv[k:k + len(ring)]
        k += len(ring)
        if len(idx) > 1 and idx[0] == idx[-1]:
            idx = idx[:-1]  # 閉じた輪の最後の点
        for i in range(1, len(idx) - 1):  # 扇形に三角形分割
            t = (idx[0], idx[i], idx[i + 1])
            if len(set(t)) == 3:
                tris.append(t)
                tkind.append(kind)
                tobj.append(obj)
    gmlids = np.array(sorted(ids, key=ids.get), dtype=str)
    return verts, np.array(tris, np.int32), np.array(tkind, np.uint8), np.array(tobj, np.int32), gmlids


def main(root="data/plateau", out="data/plateau_c1.npz"):
    verts, tris, kind, obj, gmlids = extract(root)
    np.savez_compressed(out, verts=verts, tris=tris, kind=kind, obj=obj, obj_gmlid=gmlids)
    print(f"{out}: {os.path.getsize(out) / 1e6:.2f} MB, 頂点 {len(verts)}, 三角形 {len(tris)}, 地物 {len(gmlids)}")
    for k, name in KINDS.items():
        m = kind == k
        print(f"  {k:2d} {name:20s} 三角形 {m.sum():6d}  地物 {len(np.unique(obj[m])):5d}")


if __name__ == "__main__":
    main(*sys.argv[1:])
