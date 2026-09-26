"""OpenStreetMap（Overpass API）から C1 と周辺ランプを取得し、周回ごとの中心線にまとめる。"""
import json
import math
import re

import numpy as np
import requests

from . import config
from .proj import to_xy

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

C1_REF = re.compile(r"(^|;)\s*C1\s*(;|$)")
C1_NAME = "都心環状線"


def fetch(path, bbox=config.BBOX):
    s, w, n, e = bbox
    query = f"""
    [out:json][timeout:180];
    way["highway"~"^(motorway|motorway_link)$"]({s},{w},{n},{e});
    out body geom;
    """
    r = requests.post(OVERPASS_URL, data={"data": query}, timeout=240)
    r.raise_for_status()
    with open(path, "w") as f:
        f.write(r.text)


def _is_c1(tags):
    return tags.get("highway") == "motorway" and (
        bool(C1_REF.search(tags.get("ref", ""))) or C1_NAME in tags.get("name", "")
    )


def _seg_attrs(tags):
    try:
        lanes = int(str(tags.get("lanes", "")).split(";")[0])
    except ValueError:
        lanes = config.RAMP_LANES if tags.get("highway") == "motorway_link" else config.DEFAULT_LANES
    try:
        layer = int(tags.get("layer", "0"))
    except ValueError:
        layer = 0
    return {
        "lanes": lanes,
        "bridge": tags.get("bridge", "no") not in ("no", ""),
        "tunnel": tags.get("tunnel", "no") not in ("no", "") or tags.get("covered") == "yes",
        "cutting": tags.get("cutting", "no") not in ("no", ""),
        "layer": layer,
        "destination": tags.get("destination", "") or tags.get("exit_to", "") or tags.get("name", ""),
        "ref": tags.get("ref", ""),
    }


class Network:
    """有向グラフ。辺 = OSMのウェイの隣り合うノード間（一方通行の向き）。"""

    def __init__(self, elements):
        self.pos = {}
        self.succ = {}
        self.pred = {}
        self.edge = {}  # (a, b) -> {"c1": bool, **attrs}
        for el in elements:
            if el.get("type") != "way" or "geometry" not in el:
                continue
            tags = el.get("tags", {})
            nodes = el["nodes"]
            geom = el["geometry"]
            for nid, g in zip(nodes, geom):
                if g is not None:
                    self.pos[nid] = (g["lat"], g["lon"])
            order = list(range(len(nodes)))
            if tags.get("oneway") == "-1":
                order.reverse()
            attrs = dict(_seg_attrs(tags), c1=_is_c1(tags))
            for i, j in zip(order[:-1], order[1:]):
                a, b = nodes[i], nodes[j]
                self.edge[(a, b)] = attrs
                self.succ.setdefault(a, []).append(b)
                self.pred.setdefault(b, []).append(a)
                if tags.get("oneway") == "no":
                    self.edge[(b, a)] = attrs
                    self.succ.setdefault(b, []).append(a)
                    self.pred.setdefault(a, []).append(b)
        ids = list(self.pos)
        lat = np.array([self.pos[i][0] for i in ids])
        lon = np.array([self.pos[i][1] for i in ids])
        x, y = to_xy(lat, lon)
        self.xy = {i: (float(a), float(b)) for i, a, b in zip(ids, x, y)}

    def _heading(self, a, b):
        (x0, y0), (x1, y1) = self.xy[a], self.xy[b]
        return math.atan2(y1 - y0, x1 - x0)

    def _straightest(self, prev, cur, cands):
        if prev is None or len(cands) == 1:
            return cands[0]
        h = self._heading(prev, cur)

        def turn(n):
            d = self._heading(cur, n) - h
            return abs((d + math.pi) % (2 * math.pi) - math.pi)

        return min(cands, key=turn)

    def c1_loops(self):
        """C1 の辺だけを辿り、閉じた周回（内回り・外回り）を見つける。"""
        c1_succ = {}
        for (a, b), at in self.edge.items():
            if at["c1"]:
                c1_succ.setdefault(a, []).append(b)
        loops, used = [], set()
        for start in list(c1_succ):
            if start in used:
                continue
            path, seen, prev, cur = [], {}, None, start
            while cur in c1_succ and cur not in seen:
                seen[cur] = len(path)
                path.append(cur)
                nxt = self._straightest(prev, cur, c1_succ[cur])
                prev, cur = cur, nxt
            if cur in seen:
                cyc = path[seen[cur]:]
                if not used.intersection(cyc) and self._length(cyc + [cyc[0]]) > 5000:
                    loops.append(cyc)
                    used.update(cyc)
            used.update(path)
        return loops

    def _length(self, nodes):
        p = np.array([self.xy[n] for n in nodes])
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    def follow(self, junction, first, forward, loop_nodes, max_len):
        """分岐点から本線以外の辺を max_len まで辿る（forward=False なら逆向き）。"""
        nodes = [junction, first]
        length = self._length(nodes)
        prev, cur = junction, first
        while length < max_len and cur not in loop_nodes:
            nbrs = self.succ.get(cur, []) if forward else self.pred.get(cur, [])
            nbrs = [n for n in nbrs if n != prev]
            if not nbrs:
                break
            nxt = self._straightest(prev, cur, nbrs)
            nodes.append(nxt)
            length += self._length([cur, nxt])
            prev, cur = cur, nxt
        return nodes


def signed_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def load(path):
    """周回と分岐・合流ランプを返す。

    loops: [{"name", "pts"(N,2), "attrs"[N] （i番目の点から次の点までの区間の属性）, "nodes"}]
    ramps: [{"kind": "diverge"|"merge", "loop", "index"（本線ノードの番号）, "pts", "attrs"}]
    """
    with open(path) as f:
        data = json.load(f)
    net = Network(data["elements"])
    loops = []
    for cyc in net.c1_loops():
        pts = np.array([net.xy[n] for n in cyc])
        attrs = [net.edge[(a, b)] for a, b in zip(cyc, cyc[1:] + cyc[:1])]
        # 左側通行: 時計回り＝外回り
        name = "Outer" if signed_area(pts) < 0 else "Inner"
        loops.append({"name": name, "pts": pts, "attrs": attrs, "nodes": cyc})
    ramps = []
    all_loop_nodes = set(n for lp in loops for n in lp["nodes"])
    for li, lp in enumerate(loops):
        on_loop = set(lp["nodes"])
        for idx, n in enumerate(lp["nodes"]):
            nxt = lp["nodes"][(idx + 1) % len(lp["nodes"])]
            prv = lp["nodes"][idx - 1]
            for b in net.succ.get(n, []):
                if b != nxt and b not in on_loop:
                    ramps.append(_ramp(net, "diverge", li, idx, n, b, True, all_loop_nodes))
            for a in net.pred.get(n, []):
                if a != prv and a not in on_loop:
                    ramps.append(_ramp(net, "merge", li, idx, n, a, False, all_loop_nodes))
    return loops, ramps


def _ramp(net, kind, li, idx, junction, first, forward, loop_nodes):
    chain = net.follow(junction, first, forward, loop_nodes - {junction}, config.RAMP_LENGTH)
    pairs = zip(chain, chain[1:])
    attrs = [net.edge[(a, b) if forward else (b, a)] for a, b in pairs]
    # 点は分岐点から離れる向きに並べる。attrs[i] は i→i+1 の区間
    return {
        "kind": kind,
        "loop": li,
        "index": idx,
        "pts": np.array([net.xy[n] for n in chain]),
        "attrs": attrs + attrs[-1:],
        "destination": next((a["destination"] for a in attrs if a["destination"]), ""),
    }
