"""動作確認用の仮データ。楕円の周回（内回り・外回り）と分岐・合流ランプを Overpass 形式で書き出す。"""
import json

import numpy as np

from .proj import to_latlon


def _section(deg):
    if 160 <= deg < 220:
        return {"tunnel": "yes", "layer": "-1"}
    if 130 <= deg < 160:
        return {"cutting": "yes"}
    return {"bridge": "yes", "layer": "1"}


def write(path):
    elements, nid = [], [1]

    def node(x, y):
        nid[0] += 1
        return nid[0], (x, y)

    def way(wid, nodes, tags):
        xy = np.array([p for _, p in nodes])
        lat, lon = to_latlon(xy[:, 0], xy[:, 1])
        elements.append({
            "type": "way", "id": wid, "nodes": [i for i, _ in nodes],
            "geometry": [{"lat": float(a), "lon": float(b)} for a, b in zip(lat, lon)],
            "tags": tags,
        })

    wid = 1000
    rings = {}
    for name, off, cw in (("outer", 6.0, True), ("inner", -6.0, False)):
        t = np.radians(np.arange(0, 360, 1.2))
        if cw:
            t = -t
        pts = [node((1800 + off) * np.cos(a), (1300 + off) * np.sin(a)) for a in t]
        rings[name] = pts
        deg = np.degrees(np.abs(t)) % 360
        chunk = 15
        for k in range(0, len(pts), chunk):
            seg = pts[k:k + chunk + 1]
            if len(seg) < chunk + 1:
                seg = seg + pts[:1]
            tags = {"highway": "motorway", "ref": "C1", "name": "首都高速都心環状線", "oneway": "yes",
                    "lanes": "3" if 60 <= deg[k] < 90 else "2", **_section(deg[k])}
            wid += 1
            way(wid, seg, tags)

    # 外回り 45° 付近から外側へ出る分岐
    j = rings["outer"][300 - 38]
    p0 = np.array(j[1])
    d = p0 / np.linalg.norm(p0)
    t = np.array([d[1], -d[0]])
    ramp = [j] + [node(*(p0 + t * s + d * (s * s / 400.0))) for s in np.arange(20, 320, 20)]
    wid += 1
    way(wid, ramp, {"highway": "motorway_link", "oneway": "yes", "lanes": "1", "bridge": "yes",
                    "layer": "1", "destination": "霞が関"})

    # 内回り 250° 付近へ内側から入る合流
    j = rings["inner"][int(250 / 1.2)]
    p0 = np.array(j[1])
    d = -p0 / np.linalg.norm(p0)
    t = np.array([-d[1], d[0]])
    merge = [node(*(p0 - t * s + d * (s * s / 400.0))) for s in np.arange(300, 0, -20)] + [j]
    wid += 1
    way(wid, merge, {"highway": "motorway_link", "oneway": "yes", "lanes": "1", "bridge": "yes",
                     "layer": "1", "destination": "銀座"})

    with open(path, "w") as f:
        json.dump({"elements": elements}, f, ensure_ascii=False)
