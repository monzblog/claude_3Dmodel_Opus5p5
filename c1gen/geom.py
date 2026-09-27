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


def sweep(mesh, route, lat, dz, segmask, zoff=None):
    """断面（N×K の lat, dz）を中心線に沿って押し出す。

    UV は u = 道路に沿った距離、v = 断面に沿った距離（どちらもメートル）。
    """
    lat = np.broadcast_to(np.asarray(lat, float), (len(route.s), np.shape(lat)[-1]))
    dz = np.broadcast_to(np.asarray(dz, float), lat.shape)
    K = lat.shape[1]
    z = route.z if zoff is None else route.z + zoff
    for rows in runs(segmask, route.closed):
        R = len(rows)
        xy = route.P[rows, None, :] + route.N[rows, None, :] * lat[rows, :, None]
        zz = z[rows, None] + route.bank[rows, None] * lat[rows] + dz[rows]
        verts = np.concatenate([xy, zz[..., None]], -1).reshape(-1, 3)
        idx = np.arange(R * K).reshape(R, K)
        a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
        faces = np.stack([a, b, c, d], -1).reshape(-1, 4).tolist()
        u = np.arange(R) * route.step + route.s[rows[0]]
        prof = np.stack([lat[rows], dz[rows]], -1)
        v = np.concatenate([np.zeros((R, 1)), np.cumsum(np.linalg.norm(np.diff(prof, axis=1), axis=-1), 1)], 1)
        uvs = np.stack([np.broadcast_to(u[:, None], (R, K)), v], -1)
        mesh.add(verts, faces, uvs)


def _side_profile(side, lat, dz):
    """左側用の断面を右側にも使えるよう、左右反転して並びも逆にする。"""
    lat, dz = np.asarray(lat, float), np.asarray(dz, float)
    if side > 0:
        return lat, dz
    return -lat[..., ::-1], dz[..., ::-1]


def box(mesh, center_xy, fwd, half_l, half_w, zb, zt, tilt=0.0):
    """tilt: 左へ 1m で上がる高さ。路面の横断勾配に合わせて底と上面を傾ける。"""
    f = np.asarray(fwd, float)
    f = f / np.linalg.norm(f)
    l = np.array([-f[1], f[0]])
    c = np.asarray(center_xy, float)
    corners = [c - f * half_l - l * half_w, c + f * half_l - l * half_w,
               c + f * half_l + l * half_w, c - f * half_l + l * half_w]
    side = [-half_w, -half_w, half_w, half_w]
    verts = [(*p, zb + tilt * y) for p, y in zip(corners, side)] + [(*p, zt + tilt * y) for p, y in zip(corners, side)]
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    mesh.add(verts, faces)


def cylinder(mesh, center_xy, fwd, radius, half_len, zc, seg_n=16):
    """道路に沿った向きの横向き円柱（ジェットファンなど）。"""
    f = np.append(np.asarray(fwd, float) / np.linalg.norm(fwd), 0.0)
    l = np.array([-f[1], f[0], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    c = np.append(np.asarray(center_xy, float), zc)
    a = np.linspace(0, 2 * np.pi, seg_n, endpoint=False)
    ring = radius * (np.cos(a)[:, None] * l + np.sin(a)[:, None] * up)
    verts = np.vstack([c - f * half_len + ring, c + f * half_len + ring])
    faces = [[i, (i + 1) % seg_n, seg_n + (i + 1) % seg_n, seg_n + i] for i in range(seg_n)]
    faces += [list(range(seg_n))[::-1], list(range(seg_n, 2 * seg_n))]
    mesh.add(verts, faces)


def vcylinder(mesh, center_xy, radius, zb, zt, seg_n=16):
    """縦向きの円柱。UV は u = 周方向（1周 = 3m 相当）、v = 高さ。"""
    a = np.linspace(0, 2 * np.pi, seg_n + 1)
    c = np.asarray(center_xy, float)
    ring = c + radius * np.column_stack([np.cos(a), np.sin(a)])
    verts = np.vstack([np.column_stack([ring, np.full(seg_n + 1, zb)]),
                       np.column_stack([ring, np.full(seg_n + 1, zt)])])
    u = a / (2 * np.pi) * 3.0
    uv = np.vstack([np.column_stack([u, np.zeros_like(u)]), np.column_stack([u, np.full_like(u, zt - zb)])])
    m = seg_n + 1
    faces = [[i, i + 1, m + i + 1, m + i] for i in range(seg_n)]
    faces += [list(range(m, 2 * m - 1))]
    mesh.add(verts, faces, uv)


def plate(mesh, center, facing, half_w, half_h, flip=False):
    """facing の向きを向いた縦の板。UV は 0〜1（flip で左右反転）。"""
    f = np.asarray(facing, float)
    f = f / np.linalg.norm(f)
    u = np.array([-f[1], f[0], 0.0])  # 正面から見て右
    w = np.array([0.0, 0.0, 1.0])
    c = np.asarray(center, float)
    verts = [c - u * half_w - w * half_h, c + u * half_w - w * half_h, c + u * half_w + w * half_h, c - u * half_w + w * half_h]
    uv = [[1, 0], [0, 0], [0, 1], [1, 1]] if flip else [[0, 0], [1, 0], [1, 1], [0, 1]]
    mesh.add(verts, [[0, 1, 2, 3]], uv)


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
    sweep(mesh, route, np.hstack([-h, h]), 0.0, segmask, zoff=zoff)


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


def tunnel_offset(route):
    return half_width(route) + C.WALKWAY_WIDTH


def tunnel_shell(tile, dirty, upper, route, segmask):
    """トンネルの壁（下は白いタイルパネル、下端付近は汚れたタイル）と、上の壁・天井。"""
    o = tunnel_offset(route)[:, None]
    H, W0, D, T = C.TUNNEL_HEIGHT, C.WALKWAY_HEIGHT, C.TILE_DIRTY_TOP, C.TILE_TOP
    for side in (1, -1):
        for mesh, lo, hi in ((dirty, W0, D), (tile, D, T)):
            # 壁の面が道路側を向くよう、左は下から上、右は上から下の順に並べる
            lat, dz = _side_profile(side, np.hstack([o, o]), np.broadcast_to([lo, hi], (len(o), 2)))
            sweep(mesh, route, lat, dz, segmask)
    sweep(upper, route, np.hstack([o, o, -o, -o]), np.array([T, H, H, T]), segmask)


def walkway(mesh, route, side, segmask):
    """トンネル内の点検用通路（縁石の高さの歩道）。"""
    e = half_width(route)[:, None]
    lat = np.hstack([e, e, e + C.WALKWAY_WIDTH])
    dz = np.broadcast_to([0.0, C.WALKWAY_HEIGHT, C.WALKWAY_HEIGHT], lat.shape)
    lat, dz = _side_profile(side, lat, dz)
    sweep(mesh, route, lat, dz, segmask)


def portals(mesh, route, mask):
    """トンネルの出入口の壁（天井から地面まで）。"""
    m = np.asarray(mask, bool)
    prev = np.roll(m, 1) if route.closed else np.concatenate([[False], m[:-1]])
    nxt = np.roll(m, -1) if route.closed else np.concatenate([m[1:], [False]])
    ends = np.flatnonzero((m & ~prev) | (m & ~nxt))
    for i in ends:
        o = tunnel_offset(route)[i]
        zb = route.z[i] + C.TUNNEL_HEIGHT
        zt = max(route.ground[i] + 0.5, zb + 1.0)
        box(mesh, route.P[i], route.T[i], 0.4, o + 0.6, zb, zt, tilt=route.bank[i])
        for side in (1, -1):
            y = side * (o + 0.3)
            box(mesh, route.P[i] + route.N[i] * y, route.T[i], 0.4, 0.3, route.zat(i, y), route.zat(i, y) + C.TUNNEL_HEIGHT)


def retaining_wall(mesh, route, side, segmask):
    e = (half_width(route) + C.BARRIER_BASE)[:, None]
    top = np.maximum(route.ground - route.z + 0.5, C.BARRIER_HEIGHT)[:, None]
    lat = np.hstack([e, e, e + 0.4])
    dz = np.hstack([np.zeros_like(top), top, top])
    lat, dz = _side_profile(side, lat, dz)
    sweep(mesh, route, lat, dz, segmask)


def markings(white, yellow, route, segmask, zoff=None, no_change=None):
    """車道外側線（実線）と車線境界線（破線 8m/12m、車線変更禁止の所は黄色の実線）。"""
    h = half_width(route)
    lift = C.MARK_LIFT
    w = C.MARK_WIDTH

    def line(mesh, y, width, mask):
        sweep(mesh, route, np.stack([y - width / 2, y + width / 2], 1), lift, mask & segmask, zoff=zoff)

    all_seg = np.ones(len(segmask), bool)
    line(white, h - C.SHOULDER_LEFT, 0.20, all_seg)
    line(white, -h + C.SHOULDER_RIGHT, w, all_seg)
    phase = np.mod(route.s, C.DASH_LEN + C.DASH_GAP) < C.DASH_LEN
    dash = phase if route.closed else phase[:-1]
    ban = seg(no_change, route.closed) if no_change is not None else np.zeros(len(segmask), bool)
    for k in range(1, int(route.lanes.max())):
        has = seg(route.lanes > k, route.closed)
        y = h - C.SHOULDER_LEFT - k * C.LANE_WIDTH
        line(white, y, w, has & dash & ~ban)
        line(yellow, y, w, has & ban)


def color_pavement(mesh, route, mask):
    """急カーブの赤いカラー舗装（車線の部分だけ）。"""
    h = half_width(route)[:, None]
    lat = np.hstack([-h + C.SHOULDER_RIGHT + 0.1, h - C.SHOULDER_LEFT - 0.1])
    sweep(mesh, route, lat, C.MARK_LIFT / 2, seg(mask, route.closed))


def slow_dots(mesh, route, progress):
    """急カーブ手前の減速ドット。車線の両端に白い四角が並び、カーブに近いほど内側へ寄る。

    progress は減速区間の中での進み具合（0〜1、区間外は NaN）。
    """
    h = half_width(route)
    every = max(1, int(round(4.0 / route.step)))
    for i in range(0, len(route.s), every):
        k = progress[i]
        if np.isnan(k):
            continue
        for j in range(int(route.lanes[i])):
            left = h[i] - C.SHOULDER_LEFT - j * C.LANE_WIDTH
            for y in (left - 0.25 - 0.45 * k, left - C.LANE_WIDTH + 0.25 + 0.45 * k):
                box(mesh, route.P[i] + route.N[i] * y, route.T[i], 0.15, 0.15,
                    route.zat(i, y), route.zat(i, y) + C.MARK_LIFT * 1.5)


def lane_centers(route):
    """車線ごとの中心線（1 = 一番左の第一通行帯）。無い所は NaN。"""
    h = half_width(route)
    out = []
    for j in range(1, int(route.lanes.max()) + 1):
        y = h - C.SHOULDER_LEFT - (j - 0.5) * C.LANE_WIDTH
        xy = route.P + route.N * y[:, None]
        pts = np.column_stack([xy, route.z + route.bank * y])
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
        box(mesh, route.P[i], route.T[i], 0.9, h[i] + C.BARRIER_BASE - 0.3, top - C.CROSSBEAM_DEPTH, top,
            tilt=route.bank[i])
        top -= abs(route.bank[i]) * (h[i] + C.BARRIER_BASE)
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
    """トンネル照明（壁の上の方に、ほぼ連続して並ぶ灯具）。"""
    every = max(1, int(round((C.TUNNEL_LIGHT_LEN + 0.5) / route.step)))
    o = tunnel_offset(route)
    for i in range(0, len(route.s), every):
        if not route.tunnel[i]:
            continue
        for side in (1, -1):
            y = side * (o[i] - 0.2)
            z = route.zat(i, y) + C.TUNNEL_HEIGHT
            box(mesh, route.P[i] + route.N[i] * y, route.T[i], C.TUNNEL_LIGHT_LEN / 2, 0.15, z - 0.9, z - 0.7)


def tunnel_equipment(boxes, red, green, fans, route):
    """非常用設備の箱（非常電話・消火栓）、避難誘導灯、ジェットファン。"""
    o = tunnel_offset(route)
    n = len(route.s)
    every = max(1, int(round(C.EQUIP_SPACING / route.step)))
    for i in range(0, n, every):
        if not route.tunnel[i]:
            continue
        p, nn, t, z = route.P[i], route.N[i], route.T[i], route.zat(i, o[i])
        box(boxes, p + nn * (o[i] - 0.12), t, 0.45, 0.12, z + 0.5, z + 1.9)
        box(red, p + nn * (o[i] - 0.25), t, 0.12, 0.02, z + 2.0, z + 2.2)
        j = (i + every // 2) % n
        if route.tunnel[j]:
            for side in (1, -1):
                y = side * (o[j] - 0.03)
                box(green, route.P[j] + route.N[j] * y, route.T[j], 0.25, 0.03,
                    route.zat(j, y) + 0.9, route.zat(j, y) + 1.1)
    every = max(1, int(round(C.JETFAN_SPACING / route.step)))
    for i in range(every // 2, n, every):
        if not route.tunnel[i]:
            continue
        for y in (-1.6, 1.6):
            cylinder(fans, route.P[i] + route.N[i] * y, route.T[i], 0.55, 2.0, route.zat(i, y) + C.TUNNEL_HEIGHT - 0.8)


def light_poles(poles, lamps, route, skip=None, blocked=None):
    """左の壁高欄の上に立つ照明柱。skip[i] が True の所と、真上を別の路面が通る所には立てない。"""
    every = max(1, int(round(C.LIGHT_SPACING / route.step)))
    h = half_width(route)
    for i in range(0, len(route.s), every):
        if route.tunnel[i] or (skip is not None and skip[i]):
            continue
        e = h[i] + C.BARRIER_BASE / 2
        base = route.zat(i, e) + C.BARRIER_HEIGHT
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
    top += abs(route.bank[i]) * post
    for o in (post, -post):
        box(frames, p + n * o, t, 0.2, 0.2, route.zat(i, o), top)
    box(frames, p, t, 0.25, post, top - 0.5, top - 0.1)
    width = min(2 * h - 1.0, 7.0)
    center = p + n * side * max(0.0, h - width / 2 - 0.5)
    ph = 2.5
    zb = z + C.SIGN_CLEARANCE + abs(route.bank[i]) * h
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
    base = route.zat(i, e) + C.BARRIER_HEIGHT
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
    base = route.zat(i, e) + C.BARRIER_HEIGHT
    box(posts, p + n * e, t, 0.04, 0.04, base, base + 2.0)
    c = np.append(p + n * e - t * 0.06, base + 2.0)
    diamond(yellow, c, -t, 0.45)
    texts.append({"text": text, "pos": c - np.append(t, 0) * 0.01, "facing": -t, "size": 0.2,
                  "material": "WarnText", "max_width": 0.5})


# ---- 壁高欄の上の設備 -----------------------------------------------------

def sound_wall(panels, clear, posts, route, segmask, side=1):
    """遮音壁。壁高欄の上に H形鋼の支柱を立て、下は金属の吸音板、上は透明板。"""
    e = (half_width(route) + C.BARRIER_BASE - 0.12)[:, None]
    z0 = C.BARRIER_HEIGHT
    z1 = z0 + C.SOUND_PANEL_HEIGHT
    z2 = z1 + C.SOUND_CLEAR_HEIGHT
    for mesh, lo, hi in ((panels, z0, z1), (clear, z1, z2)):
        lat, dz = _side_profile(side, np.hstack([e, e]), np.broadcast_to([lo, hi], (len(e), 2)))
        sweep(mesh, route, lat, dz, segmask)
    m = np.asarray(segmask, bool)
    every = max(1, int(round(C.SOUND_POST_SPACING / route.step)))
    for i in range(0, len(m), every):
        if not m[i]:
            continue
        o = side * (e[i, 0] + 0.05)
        box(posts, route.P[i] + route.N[i] * o, route.T[i], 0.1, 0.1, route.zat(i, o) + z0, route.zat(i, o) + z2 + 0.1)
    # 上端の笠木
    lat, dz = _side_profile(side, np.hstack([e - 0.1, e + 0.15]), np.broadcast_to([z2 + 0.1, z2 + 0.1], (len(e), 2)))
    sweep(posts, route, lat, dz, segmask)


def rail_fence(mesh, route, segmask, side=1):
    """壁高欄の上の金属パイプの防護柵（支柱＋横桟2本）。"""
    e = half_width(route) + C.BARRIER_BASE - 0.15
    H = C.BARRIER_HEIGHT
    for hgt in C.RAIL_HEIGHTS:
        lat = side * np.stack([e - 0.04, e + 0.04, e + 0.04, e - 0.04], 1)
        dz = np.broadcast_to([H + hgt - 0.04, H + hgt - 0.04, H + hgt + 0.04, H + hgt + 0.04], lat.shape)
        sweep(mesh, route, np.hstack([lat, lat[:, :1]]), np.hstack([dz, dz[:, :1]]), segmask)
    m = np.asarray(segmask, bool)
    every = max(1, int(round(C.RAIL_POST_SPACING / route.step)))
    for i in range(0, len(m), every):
        if m[i]:
            zb = route.zat(i, side * e[i]) + H
            box(mesh, route.P[i] + route.N[i] * side * e[i], route.T[i], 0.05, 0.05,
                zb, zb + max(C.RAIL_HEIGHTS) + 0.05)


def delineators(white, orange, route, mask_left, mask_right):
    """壁高欄の上の視線誘導標（左は白、右は橙）。"""
    every = max(1, int(round(C.DELINEATOR_SPACING / route.step)))
    h = half_width(route)
    for i in range(0, len(route.s), every):
        for side, mesh, m in ((1, white, mask_left), (-1, orange, mask_right)):
            if not m[i]:
                continue
            y = side * (h[i] + 0.05)
            zt = route.zat(i, y) + C.BARRIER_HEIGHT
            box(mesh, route.P[i] + route.N[i] * y, route.T[i], 0.03, 0.05, zt - 0.25, zt - 0.12)


def chevrons(mesh, route, curv, mask):
    """急カーブの外側に並べる矢羽根板（カーブの向きを指す）。"""
    every = max(1, int(round(C.CHEVRON_SPACING / route.step)))
    h = half_width(route)
    for i in range(0, len(route.s), every):
        if not mask[i]:
            continue
        side = -1 if curv[i] > 0 else 1  # 左カーブなら外側は右
        y = side * (h[i] + 0.02)
        c = np.append(route.P[i] + route.N[i] * y, route.zat(i, y) + C.BARRIER_HEIGHT + 0.45)
        # 矢印はカーブの向き（左カーブなら左向き）
        plate(mesh, c, -route.T[i], 0.30, 0.45, flip=curv[i] > 0)


def joints(mesh, route, mask):
    """高架の伸縮継手（路面を横切る鋼製の帯）。"""
    every = max(1, int(round(C.JOINT_SPACING / route.step)))
    h = half_width(route)
    for i in range(every // 2, len(route.s), every):
        if mask[i]:
            box(mesh, route.P[i], route.T[i], 0.12, h[i], route.z[i] - 0.01, route.z[i] + C.MARK_LIFT * 0.8,
                tilt=route.bank[i])


def cushion_drums(mesh, center_xy, fwd, z, count=3):
    """分岐の先端に置く衝撃緩和用のクッションドラム（黄と黒）。"""
    f = np.asarray(fwd, float) / np.linalg.norm(fwd)
    l = np.array([-f[1], f[0]])
    for k in range(count):
        for j in range(-(k // 2), k // 2 + 1):
            vcylinder(mesh, center_xy - f * (1.0 * k) + l * (0.95 * j), 0.45, z, z + 0.95)


def gore_zebra(mesh, pts_a, pts_b, z):
    """分岐部の導流帯。2本の境界線の間を斜めの白線で埋める。"""
    for a, b in zip(pts_a, pts_b):
        d = b - a
        L = np.linalg.norm(d)
        if L < 0.6:
            continue
        mid = (a + b) / 2
        box(mesh, mid, d, L / 2, 0.22, z, z + C.MARK_LIFT)
