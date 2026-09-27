"""首都高C1モデルの生成。

    python -m c1gen.build --fetch          # OSM と標高を取得してから生成
    python -m c1gen.build                  # 取得済みデータから生成
    python -m c1gen.build --synthetic      # 仮データ（楕円の周回）で動作確認
"""
import argparse
import re
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
from . import plateau as PL
from . import sections as S
from .proj import to_latlon
from .route import Route, _gauss, _limit_grade, loop_profile, ramp_profile

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


def make_routes(loops_raw, ramps_raw, dem, pl=None):
    """pl（PLATEAU）があれば、路面の高さを実測に合わせる。"""
    loops = []
    for li, lp in enumerate(loops_raw):
        r = Route(lp["name"], lp["pts"], lp["attrs"], closed=True)
        r.P0 = r.P.copy()
        loop_profile(r, dem)
        if pl is not None:
            prior = r.z.copy()
            PL.fit_profile(r, pl)
            # 平面位置を実測の路面の端に合わせてから、高さをもう一度拾い直す
            near = np.zeros(len(r.s), bool)
            for rr in ramps_raw:
                if rr["loop"] == li and len(rr["pts"]):
                    near[np.argmin(np.linalg.norm(r.P - rr["pts"][0], axis=1))] = True
            r.plan_shift = PL.fit_plan(r, pl, S.dilate(near, 150.0, r.step, True))
            r.set_profile(r.ground, prior)
            r.coverage = PL.fit_profile(r, pl)
        loops.append(r)
    _separate_twins(loops)
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
        j = int(np.argmin(np.linalg.norm(main.P - main.shift - r.P[0], axis=1)))
        # 本線を押し広げた分だけ、分岐点の近くのランプも一緒に動かす
        fade = np.clip(1 - r.s / 60.0, 0, 1)[:, None]
        r.move(r.P + main.shift[j] * fade)
        ramp_profile(r, dem, main.z[j])
        if pl is not None:
            PL.fit_profile(r, pl, z_start=main.z[j])
        k = min(len(r.P) - 1, int(60 / r.step))
        r.kind = rr["kind"]
        r.main = main
        r.junction = j
        r.side = 1 if np.dot(r.P[k] - main.P[j], main.N[j]) >= 0 else -1
        r.destination = rr["destination"]
        ramps.append(r)
    ramps = _dedupe_ramps(ramps)
    for r in loops + ramps:
        r.bank = S.superelevation(r)
    return loops, ramps


def _separate_twins(loops, iters=3):
    """内回り・外回りの中心線が近すぎて壁高欄が重なる所を、左右へ滑らかに押し広げる。

    OSM の上下線は実際より近く描かれていることがあり、そのままだと相手の壁高欄が車線に入る。
    """
    orig = [getattr(r, "P0", r.P).copy() for r in loops]
    for r in loops:
        r.shift = r.P - getattr(r, "P0", r.P)
    if len(loops) != 2:
        return
    _stack_tunnels(loops)
    for _ in range(iters):
        shifts = []
        for a, b in (loops, loops[::-1]):
            d, k = cKDTree(b.P).query(a.P)
            need = G.half_width(a) + G.half_width(b)[k] + 2 * C.BARRIER_BASE + 0.1
            short = np.where((d < need) & (np.abs(a.z - b.z[k]) < C.TWIN_DZ), need - d, 0.0)
            away = (a.P - b.P[k]) / np.maximum(d, 1e-6)[:, None]
            shifts.append(away * (short / 2)[:, None])
        if not any(np.abs(v).max() > 0.02 for v in shifts):
            break
        for r, v in zip(loops, shifts):
            # ずらし量を滑らかにしてから足す（最大値を保つため少し多めに）
            sm = _gauss(v, 15.0 / r.step, True)
            gain = np.linalg.norm(v, axis=1).max() / max(np.linalg.norm(sm, axis=1).max(), 1e-6)
            r.move(r.P + sm * min(gain, 3.0))
    for r, p in zip(loops, orig):
        r.shift = r.P - p


def _stack_tunnels(loops):
    """上下2段に重なるトンネルで、下の段の天井が上の段の路面に届く所は、下の段を深くする。"""
    a, b = loops
    need_dz = C.TUNNEL_HEIGHT + 1.5
    for lo, hi in ((a, b), (b, a)):
        d, k = cKDTree(hi.P).query(lo.P)
        wide = G.half_width(lo) + G.half_width(hi)[k] + 2 * C.BARRIER_BASE + 0.1
        dz = hi.z[k] - lo.z
        tun = lo.tunnel | hi.tunnel[k]
        deficit = np.where((d < wide) & tun & (dz >= C.TWIN_DZ) & (dz < need_dz), need_dz - dz, 0.0)
        if not deficit.any():
            continue
        # 前後にも広げて滑らかに下げ、勾配の上限を守る
        sm = _gauss(deficit, 30.0 / lo.step, lo.closed)
        sm *= deficit.max() / max(sm.max(), 1e-6)
        z = _limit_grade(lo.z - np.maximum(sm, deficit), lo.step, lo.closed)
        lo.set_profile(lo.ground, z)


def _dedupe_ramps(ramps, tol=1.5):
    """同じ区間を二重に作っているランプ（並行する本線の別れ道を分岐・合流の両方から辿った時など）を除く。

    先に登録したランプと重なる点が半分以上なら捨て、途中から重なるなら重なる手前で切る。
    """
    kept = []
    for q in ramps:
        dup = np.zeros(len(q.P), bool)
        for o in kept:
            d, k = cKDTree(o.P).query(q.P)
            dup |= (d < tol) & (np.abs(q.z - o.z[k]) < 1.0)
        if dup.mean() > 0.5:
            continue
        if dup.any():
            end = int(np.argmax(dup))
            if q.s[end] < 30:
                continue
            q.truncate(end + 1)
        kept.append(q)
    return kept


def _blocker(routes):
    """橋脚・照明柱・標識が、別の路面（壁高欄を含む）を突き抜けないかを調べる関数を返す。"""
    P = np.vstack([r.P for r in routes])
    z = np.concatenate([r.z for r in routes])
    half = np.concatenate([G.half_width(r) + C.BARRIER_BASE for r in routes])
    owner = np.concatenate([np.full(len(r.P), k) for k, r in enumerate(routes)])
    ids = {id(r): k for k, r in enumerate(routes)}
    tree = cKDTree(P)
    reach = float(half.max() + C.PIER_SIZE)

    def blocked(route, xy, lo, hi):
        """xy の位置で、高さ lo〜hi の間を別の路面が通っていれば True。"""
        near = np.asarray(tree.query_ball_point(xy, reach), int)
        if not len(near):
            return False
        d = np.linalg.norm(P[near] - xy, axis=1)
        hit = (owner[near] != ids[id(route)]) & (d < half[near] + C.PIER_SIZE / 2 + 0.3) & (z[near] > lo) & (z[near] < hi)
        return bool(hit.any())

    return blocked


def _dot_progress(r, sharp):
    """急カーブの手前 DOT_ZONE の区間で 0→1 に増える値（区間外は NaN）。"""
    n = len(r.s)
    prog = np.full(n, np.nan)
    prev = np.roll(sharp, 1) if r.closed else np.concatenate([[False], sharp[:-1]])
    L = int(C.DOT_ZONE / r.step)
    for i0 in np.flatnonzero(sharp & ~prev):
        for k in range(L):
            j = i0 - L + k
            if not r.closed and j < 0:
                continue
            j %= n
            if not sharp[j] and not r.tunnel[j]:
                prog[j] = k / L
    return prog


def _twin_sides(r, others, margin=6.0):
    """左右それぞれ、すぐ隣を同じ高さで別の本線が並走している点を返す（掘割の中央側など）。"""
    out = {1: np.zeros(len(r.s), bool), -1: np.zeros(len(r.s), bool)}
    for o in others:
        d, k = cKDTree(o.P).query(r.P)
        lat = np.einsum("ij,ij->i", o.P[k] - r.P, r.N)
        close = (d < G.half_width(r) + G.half_width(o)[k] + 2 * C.BARRIER_BASE + margin) & (np.abs(o.z[k] - r.z) < 3.0)
        out[1] |= close & (lat > 0)
        out[-1] |= close & (lat < 0)
    return out


def build_scene(loops, ramps, water_path=None):
    layers = {}
    texts = []
    main_poly = {id(r): _deck_polygon(r) for r in loops}
    blocked = _blocker(loops + ramps)

    for r in loops:
        L = layers.setdefault(f"C1_{r.name}", Layer())
        n = f"C1_{r.name}"
        mine = [q for q in ramps if q.main is r]
        ramp_poly = _union_buffer([(q.P[: int(150 / q.step)], float(q.width.max() / 2)) for q in mine])
        allseg = np.ones(len(r.s), bool)
        kind = S.wall_types(r, water_path)
        curv = S.curvature(r)
        sharp = (np.abs(curv) > 1.0 / C.SHARP_CURVE_RADIUS) & ~r.tunnel
        ban = S.no_lane_change(r, [q.junction for q in mine])

        G.deck(L.mesh(f"{n}_Road", "Asphalt"), r, allseg)
        G.markings(L.mesh(f"{n}_Markings", "Marking"), L.mesh(f"{n}_MarkingsYellow", "MarkingYellow"),
                   r, allseg, no_change=ban)
        G.color_pavement(L.mesh(f"{n}_RedPavement", "RedPavement"), r, S.dilate(sharp, 10, r.step, True))
        G.slow_dots(L.mesh(f"{n}_Markings", "Marking"), r, _dot_progress(r, sharp))
        G.joints(L.mesh(f"{n}_ExpansionJoints", "SteelJoint"), r, r.elevated)

        removed = {}
        twin = _twin_sides(r, [o for o in loops if o is not r])
        for side in (1, -1):
            bp = r.P + r.N * (side * (G.half_width(r) + C.BARRIER_BASE / 2))[:, None]
            gap = _inside(ramp_poly, bp)
            removed[side] = gap
            G.barrier(L.mesh(f"{n}_Barriers", "Concrete"), r, side, G.seg(~gap & ~r.tunnel, True))
            G.walkway(L.mesh(f"{n}_TunnelWalkway", "Concrete"), r, side, G.seg(r.tunnel, True))
            # 掘割の擁壁は外側だけ。上下線が並ぶ中央側には立てない
            G.retaining_wall(L.mesh(f"{n}_RetainingWalls", "RetainingWall"), r, side,
                             G.seg(r.cutting & ~gap & ~twin[side], True))
        left_ok = ~removed[1]
        G.sound_wall(L.mesh(f"{n}_SoundWallPanels", "SoundPanel"), L.mesh(f"{n}_SoundWallClear", "ClearPanel"),
                     L.mesh(f"{n}_SoundWallPosts", "Metal"), r, G.seg((kind == S.SOUND) & left_ok, True))
        G.rail_fence(L.mesh(f"{n}_GuardRail", "Railing"), r, G.seg((kind == S.RAIL) & left_ok, True))
        G.delineators(L.mesh(f"{n}_Delineators", "DelineatorWhite"), L.mesh(f"{n}_DelineatorsOrange", "DelineatorOrange"),
                      r, left_ok & ~r.tunnel, ~removed[-1] & ~r.tunnel)
        G.chevrons(L.mesh(f"{n}_CurveChevrons", "Chevron"), r, curv, sharp)

        G.deck_body(L.mesh(f"{n}_Viaduct", "StructureConcrete"), r, G.seg(r.z - r.ground > 1.5, True))
        G.piers(L.mesh(f"{n}_Piers", "StructureConcrete"), r, blocked)
        G.tunnel_shell(L.mesh(f"{n}_TunnelTile", "TunnelTile"), L.mesh(f"{n}_TunnelTileLower", "TunnelTileDirty"),
                       L.mesh(f"{n}_TunnelUpper", "TunnelUpper"), r, G.seg(r.tunnel, True))
        G.portals(L.mesh(f"{n}_Tunnel_Portals", "StructureConcrete"), r, r.tunnel)
        G.tunnel_lights(L.mesh(f"{n}_TunnelLights", "TunnelLight"), r)
        G.tunnel_equipment(L.mesh(f"{n}_TunnelEquipment", "EquipBox"), L.mesh(f"{n}_TunnelEquipLamp", "EquipRed"),
                           L.mesh(f"{n}_EvacuationGuide", "GuideGreen"), L.mesh(f"{n}_JetFans", "JetFan"), r)
        G.light_poles(L.mesh(f"{n}_LightPoles", "Metal"), L.mesh(f"{n}_Lamps", "LampLight"),
                      r, skip=removed[1], blocked=blocked)
        _signs(L, texts, r, mine, blocked)

    R = layers.setdefault("C1_Ramps", Layer())
    for q in ramps:
        near = q.s < 150
        in_main = _inside(main_poly[id(q.main)], q.P) & near
        zoff = np.where(in_main, -0.03, 0.0)
        allseg = np.ones(len(q.s) - 1, bool)
        G.deck(R.mesh("C1_Ramps_Road", "Asphalt"), q, allseg, zoff=zoff)
        G.markings(R.mesh("C1_Ramps_Markings", "Marking"), R.mesh("C1_Ramps_MarkingsYellow", "MarkingYellow"),
                   q, G.seg(~in_main, False), zoff=zoff)
        ok = {}
        for side in (1, -1):
            bp = q.P + q.N * (side * (G.half_width(q) + C.BARRIER_BASE / 2))[:, None]
            ok[side] = ~(_inside(main_poly[id(q.main)], bp) & near)
            G.barrier(R.mesh("C1_Ramps_Barriers", "Concrete"), q, side, G.seg(ok[side] & ~q.tunnel, False), zoff=zoff)
            G.walkway(R.mesh("C1_Ramps_TunnelWalkway", "Concrete"), q, side, G.seg(q.tunnel & ~in_main, False))
            G.retaining_wall(R.mesh("C1_Ramps_RetainingWalls", "RetainingWall"), q, side,
                             G.seg(q.cutting & ok[side], False))
        G.delineators(R.mesh("C1_Ramps_Delineators", "DelineatorWhite"), R.mesh("C1_Ramps_DelineatorsOrange", "DelineatorOrange"),
                      q, ok[1] & ~q.tunnel, ok[-1] & ~q.tunnel)
        G.deck_body(R.mesh("C1_Ramps_Viaduct", "StructureConcrete"), q,
                    G.seg((q.z - q.ground > 1.5) & ~in_main, False), zoff=zoff)
        G.piers(R.mesh("C1_Ramps_Piers", "StructureConcrete"), q, blocked)
        G.tunnel_shell(R.mesh("C1_Ramps_TunnelTile", "TunnelTile"), R.mesh("C1_Ramps_TunnelTileLower", "TunnelTileDirty"),
                       R.mesh("C1_Ramps_TunnelUpper", "TunnelUpper"), q, G.seg(q.tunnel & ~in_main, False))
        G.portals(R.mesh("C1_Ramps_Tunnel_Portals", "StructureConcrete"), q, q.tunnel & ~in_main)
        G.tunnel_lights(R.mesh("C1_Ramps_TunnelLights", "TunnelLight"), q)
        _gore(R, q, ok)
    return layers, texts


def _gore(R, q, ok):
    """分岐・合流部の導流帯（ゼブラ）と、分岐の先端のクッションドラム。"""
    main = q.main
    j = q.junction
    k = min(len(q.P) - 1, int(60 / q.step))
    toward = 1 if np.dot(main.P[j] - q.P[k], q.N[k]) >= 0 else -1  # ランプから見た本線の側
    nose = np.flatnonzero(ok[toward] & (q.s > 5))
    if not len(nose):
        return
    i_nose = int(nose[0])
    hq = G.half_width(q)
    edge_q = q.P + q.N * (toward * (hq - C.SHOULDER_LEFT))[:, None]
    _, kk = cKDTree(main.P).query(edge_q[: i_nose + 1])
    hm = G.half_width(main)[kk]
    # 本線の、ランプ側の車道外側線
    side_m = np.sign(np.einsum("ij,ij->i", edge_q[: i_nose + 1] - main.P[kk], main.N[kk]))
    edge_m = main.P[kk] + main.N[kk] * (side_m * (hm - C.SHOULDER_LEFT))[:, None]
    lat_q = np.abs(np.einsum("ij,ij->i", edge_q[: i_nose + 1] - main.P[kk], main.N[kk]))
    outside = lat_q > hm - C.SHOULDER_LEFT + 0.5
    every = max(1, int(round(C.GORE_ZEBRA_SPACING / q.step)))
    shift = max(1, int(round(5.0 / q.step)))
    a, b = [], []
    for i in range(0, i_nose + 1 - shift, every):
        if outside[i]:
            a.append(edge_m[i])
            b.append(edge_q[min(i + shift, i_nose)])
    G.gore_zebra(R.mesh("C1_Ramps_GoreZebra", "Marking"), np.array(a), np.array(b), float(q.z[0]) + C.MARK_LIFT * 0.2)
    if q.kind == "diverge" and not q.tunnel[i_nose]:
        c = q.P[i_nose] + q.N[i_nose] * toward * (hq[i_nose] + C.BARRIER_BASE / 2) - q.T[i_nose] * 1.2
        G.cushion_drums(R.mesh("C1_Ramps_CushionDrums", "CushionDrum"), c, q.T[i_nose], float(q.z[i_nose]))


def sign_label(dest):
    """OSM の行き先を案内標識の文字にする（例: 首都高速4号新宿線 → 4 新宿、霞が関出口 → 霞が関）。

    C1 どうしの分かれ道（車線が分かれるだけの所）は None。
    """
    if not dest:
        return ""
    if "都心環状線" in dest:
        return None
    m = re.match(r"首都高速(\d+)号(.+?)線", dest)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    parts = [p for p in dest.split(";") if p][:2]
    return " ".join(p[:-2] if p.endswith(("出口", "入口")) else p for p in parts)


def _signs(L, texts, r, ramps, blocked):
    frames = L.mesh(f"C1_{r.name}_SignFrames", "Metal")
    panels = L.mesh(f"C1_{r.name}_SignPanels", "SignGreen")
    n = len(r.s)
    for q in ramps:
        dest = sign_label(q.destination)
        if dest is None:
            continue
        dest = dest or "出口"
        if q.kind == "diverge":
            exit_word = "出口" if q.side > 0 else "右 出口"
            for dist, extra in ((C.SIGN_ADVANCE, f"{exit_word} {int(C.SIGN_ADVANCE)}m"), (40.0, exit_word)):
                i = (q.junction - int(dist / r.step)) % n
                over = blocked(r, r.P[i], r.z[i] + 0.5, r.z[i] + C.SIGN_CLEARANCE + 3.0 + C.DECK_THICKNESS)
                if not r.tunnel[i] and not over:
                    lines = [dest, extra] if dest != "出口" else [extra]
                    G.gantry(frames, panels, texts, r, i, lines, q.side)
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
    step = C.TERRAIN_STEP
    # 路面が地面近く・地下にある所は、ルートごとにその路面より下へ下げる。
    # 一番近いルートの高さで決めると、掘割の本線の横を通る浅いランプの高さで本線が埋まる
    opens = []
    for r in routes:
        low = (r.z < r.ground + 1.5) & ~r.tunnel
        if not low.any():
            continue
        half = float(r.width.max() / 2 + C.BARRIER_BASE)
        runs = [r.P[rows] for rows in G.runs(G.seg(low, r.closed), r.closed)]
        # 格子1.5マス分広く下げ、斜めの面が路面に被らないようにする
        hole = _union_buffer([(p, half + 1.5 * step) for p in runs])
        opens.append((runs, half + 0.5 * step))
        if hole is None:
            continue
        inside = _inside(hole, grid)
        _, k = cKDTree(r.P[low]).query(grid[inside])
        Z[inside] = np.minimum(Z[inside], r.z[low][k] - 1.0)
    # トンネルの上は、10m格子の補間で地面が天井より下に来ないよう天井＋土かぶりまで上げる。
    # ただし別の路面（掘割・地平）の真上は開けておく
    keep_open = _union_buffer([(p, w) for runs, w in opens for p in runs])
    for r in routes:
        if not r.tunnel.any():
            continue
        half = float(r.width.max() / 2 + C.BARRIER_BASE)
        cover = _union_buffer([(r.P[rows], half + 1.5 * step)
                               for rows in G.runs(G.seg(r.tunnel, r.closed), r.closed)])
        if cover is None:
            continue
        inside = _inside(cover, grid) & ~_inside(keep_open, grid)
        _, k = cKDTree(r.P[r.tunnel]).query(grid[inside])
        top = r.z[r.tunnel] + C.TUNNEL_HEIGHT + C.TUNNEL_COVER_MIN
        Z[inside] = np.maximum(Z[inside], top[k])
    ny, nx = X.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    faces = np.stack([a, b, c, d], -1).reshape(-1, 4)
    # それでもトンネルの内側（路面〜天井の高さ）を横切る面は消す。
    # 坑口のすぐ横で別のランプが地上に出る所などで起きる
    fz = Z[faces]
    center = grid[faces].mean(1)
    drop = np.zeros(len(faces), bool)
    for r in routes:
        if not r.tunnel.any():
            continue
        o = float(r.width.max() / 2 + C.BARRIER_BASE + 0.3)
        band = _union_buffer([(r.P[rows], o + step * 0.75)
                              for rows in G.runs(G.seg(r.tunnel, r.closed), r.closed)])
        near = _inside(band, center)
        if not near.any():
            continue
        _, k = cKDTree(r.P[r.tunnel]).query(center[near])
        zt = r.z[r.tunnel][k]
        hit = (fz[near].min(1) < zt + C.TUNNEL_HEIGHT + 0.3) & (fz[near].max(1) > zt - 0.5)
        drop[np.flatnonzero(near)[hit]] = True
    faces = faces[~drop].tolist()
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
            osm.fetch_water(os.path.join(DATA, "osm_water.json"))
            print("標高タイルを取得中…")
            dem_mod.fetch(dem_root)
        dem = dem_mod.DEM(dem_root)
        if not dem.available:
            print("警告: 標高データが無いので地面を一定の高さとして作ります")

    loops_raw, ramps_raw = osm.load(osm_path)
    if len(loops_raw) != 2:
        print(f"警告: 周回が {len(loops_raw)} 本見つかりました（内回り・外回りの 2 本を想定）")
    pl = PL.Plateau.load(os.path.join(DATA, "plateau_c1.npz"))
    if pl is None:
        print("PLATEAU のデータ（data/plateau_c1.npz）が無いので、高さは推定値で作ります")
    loops, ramps = make_routes(loops_raw, ramps_raw, dem, pl)
    for r in loops:
        print(f"{DIR_JA.get(r.name, r.name)}: {r.length / 1000:.2f} km, 高架 {r.elevated.mean():.0%}, "
              f"トンネル {r.tunnel.mean():.0%}, 掘割 {r.cutting.mean():.0%}"
              + (f", 実測の高さ {r.coverage:.0%}" if hasattr(r, "coverage") else ""))
    print(f"分岐 {sum(q.kind == 'diverge' for q in ramps)} 本, 合流 {sum(q.kind == 'merge' for q in ramps)} 本")

    layers, texts = build_scene(loops, ramps, os.path.join(DATA, "osm_water.json"))
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
