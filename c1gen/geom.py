"""道路・構造物のメッシュを numpy で組み立てる（Blender に依存しない部分）。

断面は「進行方向の左 = 正」の横位置 lat と、路面からの高さ dz で表す。
"""
import numpy as np

from . import config as C


class Mesh:
    def __init__(self):
        self.v, self.f, self.uv = [], [], []
        self.n = 0

    def add(self, verts, faces, uv=None):
        verts = np.asarray(verts, float).reshape(-1, 3)
        self.v.append(verts)
        self.uv.append(np.zeros((len(verts), 2)) if uv is None else np.asarray(uv, float).reshape(-1, 2))
        self.f.extend([[i + self.n for i in face] for face in faces])
        self.n += len(verts)

    def empty(self):
        return self.n == 0

    def arrays(self):
        if not self.v:
            return np.zeros((0, 3)), [], np.zeros((0, 2))
        return np.concatenate(self.v), self.f, np.concatenate(self.uv)


def runs(segmask, closed):
    """segmask[i] = 区間 i→i+1 を作るか。連続区間ごとに行番号の配列を返す。"""
    m = np.asarray(segmask, bool)
    n = len(m)
    if not m.any():
        return []
    if closed:
        if m.all():
            return [np.append(np.arange(n), 0)]
        start = int(np.argmin(m))  # False の位置から数え始めれば回り込みを気にしなくてよい
        order = (np.arange(n) + start) % n
        out, cur = [], []
        for i in order:
            if m[i]:
                cur.append(i)
            elif cur:
                out.append(np.append(cur, (cur[-1] + 1) % n))
                cur = []
        if cur:
            out.append(np.append(cur, (cur[-1] + 1) % n))
        return out
    out, cur = [], []
    for i in range(len(m)):
        if m[i]:
            cur.append(i)
        elif cur:
            out.append(np.append(cur, cur[-1] + 1))
            cur = []
    if cur:
        out.append(np.append(cur, cur[-1] + 1))
    return out


def seg(mask, closed):
    """点ごとのマスクを区間ごと（両端が True）に変換する。"""
    mask = np.asarray(mask, bool)
    nxt = np.roll(mask, -1) if closed else mask[1:]
    return (mask & nxt) if closed else (mask[:-1] & nxt)


def sweep(mesh, route, lat, dz, segmask, zoff=None, uv=False):
    """断面（N×K の lat, dz）を中心線に沿って押し出す。"""
    lat = np.broadcast_to(np.asarray(lat, float), (len(route.s), np.shape(lat)[-1]))
    dz = np.broadcast_to(np.asarray(dz, float), lat.shape)
    K = lat.shape[1]
    z = route.z if zoff is None else route.z + zoff
    for rows in runs(segmask, route.closed):
        R = len(rows)
        xy = route.P[rows, None, :] + route.N[rows, None, :] * lat[rows, :, None]
        zz = z[rows, None] + dz[rows]
        verts = np.concatenate([xy, zz[..., None]], -1).reshape(-1, 3)
        idx = np.arange(R * K).reshape(R, K)
        a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
        faces = np.stack([a, b, c, d], -1).reshape(-1, 4).tolist()
        uvs = None
        if uv:
            v = np.arange(R) * route.step + route.s[rows[0]]
            uvs = np.stack([lat[rows], np.broadcast_to(v[:, None], (R, K))], -1)
        mesh.add(verts, faces, uvs)


def _side_profile(side, lat, dz):
    """左側用の断面を右側にも使えるよう、左右反転して並びも逆にする。"""
    lat, dz = np.asarray(lat, float), np.asarray(dz, float)
    if side > 0:
        return lat, dz
    return -lat[..., ::-1], dz[..., ::-1]


def box(mesh, center_xy, fwd, half_l, half_w, zb, zt):
    f = np.asarray(fwd, float)
    f = f / np.linalg.norm(f)
    l = np.array([-f[1], f[0]])
    c = np.asarray(center_xy, float)
    corners = [c - f * half_l - l * half_w, c + f * half_l - l * half_w,
               c + f * half_l + l * half_w, c - f * half_l + l * half_w]
    verts = [(*p, zb) for p in corners] + [(*p, zt) for p in corners]
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    mesh.add(verts, faces)


def disk(mesh, center, facing, radius, seg_n=24):
    """facing（2D）の向きを向いた垂直な円板。"""
    f = np.asarray(facing, float)
    f = f / np.linalg.norm(f)
    u = np.array([f[1], -f[0], 0.0])  # 正面から見て右
    w = np.array([0.0, 0.0, 1.0])
    a = np.linspace(0, 2 * np.pi, seg_n, endpoint=False)
    ring = np.asarray(center) + radius * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * w)
    verts = np.vstack([center, ring])
    faces = [[0, 1 + i, 1 + (i + 1) % seg_n] for i in range(seg_n)]
    mesh.add(verts, faces)


def diamond(mesh, center, facing, half):
    f = np.asarray(facing, float)
    f = f / np.linalg.norm(f)
    u = np.array([f[1], -f[0], 0.0])
    w = np.array([0.0, 0.0, 1.0])
    c = np.asarray(center)
    mesh.add([c + half * u, c + half * w, c - half * u, c - half * w], [[0, 1, 2, 3]])


# ---- 道路本体 -------------------------------------------------------------

def half_width(route):
    return route.width / 2


def deck(mesh, route, segmask, zoff=None):
    h = half_width(route)[:, None]
    sweep(mesh, route, np.hstack([-h, h]), 0.0, segmask, zoff=zoff, uv=True)


def barrier(mesh, route, side, segmask, zoff=None):
    e = half_width(route)[:, None]
    base, top, H = C.BARRIER_BASE, C.BARRIER_TOP, C.BARRIER_HEIGHT
    lat = e + np.array([0.0, 0.075, 0.15, base - top, base, base])
    dz = np.array([0.0, 0.10, 0.30, H, H, 0.0])
    lat, dz = _side_profile(side, lat, np.broadcast_to(dz, lat.shape))
    sweep(mesh, route, lat, dz, segmask, zoff=zoff)


def deck_body(mesh, route, segmask, zoff=None):
    o = (half_width(route) + C.BARRIER_BASE)[:, None]
    T = C.DECK_THICKNESS
    sweep(mesh, route, np.hstack([o, o, -o, -o]), np.array([0.0, -T, -T, 0.0]), segmask, zoff=zoff)


def tunnel_shell(mesh, route, segmask):
    o = (half_width(route) + C.BARRIER_BASE + 0.3)[:, None]
    H = C.TUNNEL_HEIGHT
    sweep(mesh, route, np.hstack([o, o, -o, -o]), np.array([0.0, H, H, 0.0]), segmask)


def portals(mesh, route, mask):
    """トンネルの出入口の壁（天井から地面まで）。"""
    m = np.asarray(mask, bool)
    prev = np.roll(m, 1) if route.closed else np.concatenate([[False], m[:-1]])
    nxt = np.roll(m, -1) if route.closed else np.concatenate([m[1:], [False]])
    ends = np.flatnonzero((m & ~prev) | (m & ~nxt))
    for i in ends:
        o = half_width(route)[i] + C.BARRIER_BASE + 0.3
        zb = route.z[i] + C.TUNNEL_HEIGHT
        zt = max(route.ground[i] + 0.5, zb + 1.0)
        box(mesh, route.P[i], route.T[i], 0.4, o + 0.6, zb, zt)
        for side in (1, -1):
            box(mesh, route.P[i] + route.N[i] * side * (o + 0.3), route.T[i], 0.4, 0.3, route.z[i], zb)


def retaining_wall(mesh, route, side, segmask):
    e = (half_width(route) + C.BARRIER_BASE)[:, None]
    top = np.maximum(route.ground - route.z + 0.5, C.BARRIER_HEIGHT)[:, None]
    lat = np.hstack([e, e, e + 0.4])
    dz = np.hstack([np.zeros_like(top), top, top])
    lat, dz = _side_profile(side, lat, dz)
    sweep(mesh, route, lat, dz, segmask)


def markings(mesh, route, segmask, zoff=None):
    """車道外側線（実線）と車線境界線（破線 8m/12m）。"""
    h = half_width(route)
    lift = C.MARK_LIFT
    w = C.MARK_WIDTH

    def line(y, width, mask):
        sweep(mesh, route, np.stack([y - width / 2, y + width / 2], 1), lift, mask & segmask, zoff=zoff)

    all_seg = np.ones(len(segmask), bool)
    line(h - C.SHOULDER_LEFT, 0.20, all_seg)
    line(-h + C.SHOULDER_RIGHT, w, all_seg)
    phase = np.mod(route.s, C.DASH_LEN + C.DASH_GAP) < C.DASH_LEN
    dash = phase if route.closed else phase[:-1]
    for k in range(1, int(route.lanes.max())):
        has = seg(route.lanes > k, route.closed)
        y = h - C.SHOULDER_LEFT - k * C.LANE_WIDTH
        line(y, w, has & dash)


def lane_centers(route):
    """車線ごとの中心線（1 = 一番左の第一通行帯）。無い所は NaN。"""
    h = half_width(route)
    out = []
    for j in range(1, int(route.lanes.max()) + 1):
        y = h - C.SHOULDER_LEFT - (j - 0.5) * C.LANE_WIDTH
        xy = route.P + route.N * y[:, None]
        pts = np.column_stack([xy, route.z])
        pts[route.lanes < j] = np.nan
        out.append(pts)
    return out


# ---- 高架・トンネル付帯物 --------------------------------------------------

def piers(mesh, route, blocked=None):
    """blocked(route, xy, z_lo, z_hi) が True の位置（下を別の路面が通る所）には柱を立てない。"""
    every = max(1, int(round(C.PIER_SPACING / route.step)))
    h = half_width(route)
    for i in range(0, len(route.s), every):
        if not route.elevated[i]:
            continue
        top = route.z[i] - C.DECK_THICKNESS
        box(mesh, route.P[i], route.T[i], 0.9, h[i] + C.BARRIER_BASE - 0.3, top - C.CROSSBEAM_DEPTH, top)
        col_top = top - C.CROSSBEAM_DEPTH
        if col_top - route.ground[i] < 0.5:
            continue
        offs = [0.0] if h[i] < 6.0 else [-h[i] / 2, h[i] / 2]
        for o in offs:
            if blocked is not None and blocked(route, route.P[i] + route.N[i] * o, -np.inf, col_top):
                continue
            box(mesh, route.P[i] + route.N[i] * o, route.T[i], C.PIER_SIZE / 2, C.PIER_SIZE / 2,
                route.ground[i] - 0.5, col_top)


def tunnel_lights(mesh, route):
    every = max(1, int(round(C.TUNNEL_LIGHT_SPACING / route.step)))
    h = half_width(route)
    for i in range(0, len(route.s), every):
        if not route.tunnel[i]:
            continue
        for o in (h[i] / 2, -h[i] / 2):
            box(mesh, route.P[i] + route.N[i] * o, route.T[i], 0.6, 0.15,
                route.z[i] + C.TUNNEL_HEIGHT - 0.15, route.z[i] + C.TUNNEL_HEIGHT - 0.02)


def light_poles(poles, lamps, route, skip=None, blocked=None):
    """左の壁高欄の上に立つ照明柱。skip[i] が True の所と、真上を別の路面が通る所には立てない。"""
    every = max(1, int(round(C.LIGHT_SPACING / route.step)))
    h = half_width(route)
    for i in range(0, len(route.s), every):
        if route.tunnel[i] or (skip is not None and skip[i]):
            continue
        e = h[i] + C.BARRIER_BASE / 2
        base = route.z[i] + C.BARRIER_HEIGHT
        top = base + C.LIGHT_POLE_HEIGHT
        p, n, t = route.P[i], route.N[i], route.T[i]
        if blocked is not None and blocked(route, p + n * (e - C.LIGHT_ARM / 2), base, top + C.DECK_THICKNESS + 0.5):
            continue
        box(poles, p + n * e, t, 0.1, 0.1, base, top)
        box(poles, p + n * (e - C.LIGHT_ARM / 2), t, 0.08, C.LIGHT_ARM / 2, top - 0.15, top)
        box(lamps, p + n * (e - C.LIGHT_ARM), t, 0.35, 0.18, top - 0.3, top - 0.15)


# ---- 標識 ----------------------------------------------------------------

def gantry(frames, panels, texts, route, i, lines, side):
    """門型の案内標識。lines は上から順の文字列。"""
    h = half_width(route)[i]
    p, n, t, z = route.P[i], route.N[i], route.T[i], route.z[i]
    post = h + C.BARRIER_BASE + 0.6
    top = z + C.SIGN_CLEARANCE + 2.9
    for o in (post, -post):
        box(frames, p + n * o, t, 0.2, 0.2, z, top)
    box(frames, p, t, 0.25, post, top - 0.5, top - 0.1)
    width = min(2 * h - 1.0, 7.0)
    center = p + n * side * max(0.0, h - width / 2 - 0.5)
    ph = 2.5
    zb = z + C.SIGN_CLEARANCE
    box(panels, center, t, 0.05, width / 2, zb, zb + ph)
    face = np.append(center - t * 0.07, 0.0)
    size = min(0.7, ph / (len(lines) + 0.6))
    for k, text in enumerate(lines):
        zc = zb + ph - (k + 0.8) * size * 1.2
        texts.append({"text": text, "pos": face + [0, 0, zc], "facing": -t, "size": size,
                      "material": "SignText", "max_width": width - 0.4})


def speed_sign(posts, red, white, texts, route, i, limit):
    h = half_width(route)[i]
    p, n, t, z = route.P[i], route.N[i], route.T[i], route.z[i]
    e = h + C.BARRIER_BASE / 2
    base = z + C.BARRIER_HEIGHT
    box(posts, p + n * e, t, 0.04, 0.04, base, base + 2.2)
    c = np.append(p + n * e - t * 0.06, base + 2.2)
    disk(red, c, -t, 0.30)
    disk(white, c - np.append(t, 0) * 0.01, -t, 0.24)
    texts.append({"text": str(limit), "pos": c - np.append(t, 0) * 0.02, "facing": -t, "size": 0.22,
                  "material": "SpeedText", "max_width": 0.4})


def warning_sign(posts, yellow, texts, route, i, side, text):
    h = half_width(route)[i]
    p, n, t, z = route.P[i], route.N[i], route.T[i], route.z[i]
    e = side * (h + C.BARRIER_BASE / 2)
    base = z + C.BARRIER_HEIGHT
    box(posts, p + n * e, t, 0.04, 0.04, base, base + 2.0)
    c = np.append(p + n * e - t * 0.06, base + 2.0)
    diamond(yellow, c, -t, 0.45)
    texts.append({"text": text, "pos": c - np.append(t, 0) * 0.01, "facing": -t, "size": 0.2,
                  "material": "WarnText", "max_width": 0.5})
