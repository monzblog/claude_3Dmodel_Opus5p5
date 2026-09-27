"""PLATEAU（国交省 3D都市モデル 首都高速道路 2023年度）の実測メッシュを使って、高さと壁を実物に合わせる。

data/plateau_c1.npz は tools/plateau_extract.py で作る軽量版（IX系、三角形）。
"""
import os

import numpy as np

from . import config as C
from .route import _gauss, _limit_grade

# tools/plateau_extract.py の kind の番号
ROAD, ISLAND, BR_FLOOR, BR_CEIL, BR_SIDE, BR_GROUND, BR_INST, PIER = 1, 2, 3, 4, 5, 6, 7, 8
TUN_GROUND, TUN_WALL, TUN_ROOF, TUN_CLOSURE, TUN_INST = 9, 10, 11, 12, 13
SURFACE = (ROAD, ISLAND, BR_FLOOR, TUN_GROUND)


class TriIndex:
    """三角形を平面の格子に登録し、点の真上・真下にある三角形の高さを返す。"""

    def __init__(self, verts, tris, cell=10.0):
        self.v = verts.astype(float)
        self.t = tris
        self.cell = cell
        xy = self.v[tris][:, :, :2]
        lo = np.floor(xy.min(1) / cell).astype(int)
        hi = np.floor(xy.max(1) / cell).astype(int)
        grid = {}
        for n, (a, b) in enumerate(zip(lo, hi)):
            for i in range(a[0], b[0] + 1):
                for j in range(a[1], b[1] + 1):
                    grid.setdefault((i, j), []).append(n)
        self.grid = {k: np.array(v) for k, v in grid.items()}

    def heights(self, x, y):
        """点 (x, y) を含む三角形の z をすべて返す。"""
        cand = self.grid.get((int(np.floor(x / self.cell)), int(np.floor(y / self.cell))))
        if cand is None:
            return np.empty(0)
        p = self.v[self.t[cand]]
        a, b, c = p[:, 0], p[:, 1], p[:, 2]
        v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
        v2 = np.array([x, y]) - a[:, :2]
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        ok = np.abs(den) > 1e-9
        den = np.where(ok, den, 1.0)
        s = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        t = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        inside = ok & (s >= -1e-6) & (t >= -1e-6) & (s + t <= 1 + 1e-6)
        z = a[:, 2] + s * (b[:, 2] - a[:, 2]) + t * (c[:, 2] - a[:, 2])
        return z[inside]


class Plateau:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.verts = d["verts"].astype(float)
        self.tris = d["tris"].astype(np.int64)
        self.kind = d["kind"]
        self.obj = d["obj"]
        self.surface = TriIndex(self.verts, self.tris[np.isin(self.kind, SURFACE)])

    @staticmethod
    def load(path):
        return Plateau(path) if path and os.path.exists(path) else None

    def levels(self, P, N, offsets=(-1.0, 0.0, 1.0)):
        """中心線の各点で、路面の高さの候補（段ごと、0.5m 以内はまとめる）。"""
        out = []
        for p, n in zip(P, N):
            zs = np.concatenate([self.surface.heights(*(p + n * o)) for o in offsets])
            if len(zs) == 0:
                out.append(np.empty(0))
                continue
            zs = np.sort(zs)
            groups = np.split(zs, np.flatnonzero(np.diff(zs) > 0.5) + 1)
            out.append(np.array([np.median(g) for g in groups]))
        return out


def _pick(levels, prior, step, z_start=None, reach=400.0, skip=2.0, restart=50.0, w_prior=0.01):
    """段の候補から、前後につながる1本の高さの列を選ぶ（動的計画法）。

    勾配の上限を超える段差ではつながない。つながらない点は飛ばしてよい（1点ごとに skip の罰）。
    こうすると、途中で交差する別の道路の段に乗り移らず、長く続く自分の段が選ばれる。
    prior（推定の縦断）は同じくらいの候補を選び分けるためだけに少し効かせる。
    reach までの欠けは飛び越えてつなぐ（近くの10点と、その先は5点おきの点から）。
    それより長い欠けの先では、restart の罰でつながりを切り直せる。
    候補が無い・飛ばした点は NaN。z_start があれば、始点の高さからつながる段を選ぶ。
    """
    n = len(levels)
    out = np.full(n, np.nan)
    K = max(1, int(reach / step))
    big = 1e6
    cost = [None] * n
    back = [None] * n
    valid = []
    run = (np.inf, -1)  # reach より前の点の (最小コスト - skip*番号, 番号)。つながりを切り直す時に使う
    old = 0  # valid のうち、run に入れ終わった数
    for i in range(n):
        z = levels[i]
        if len(z) == 0:
            continue
        base = w_prior * np.abs(z - prior[i])
        while old < len(valid) and valid[old] < i - K:
            j = valid[old]
            if cost[j].min() - skip * j < run[0]:
                run = (cost[j].min() - skip * j, j)
            old += 1
        if z_start is None:
            best = base + skip * i
        else:
            ok = np.abs(z - z_start) <= C.MAX_GRADE * step * (i + 1) * 1.5 + 0.5
            best = base + np.where(ok, skip * i, big)
        arg = np.full(len(z), -1)
        prev_j = np.full(len(z), -1)
        if run[1] >= 0:
            j = run[1]
            t = base + run[0] + skip * (i - 1) + restart
            better = t < best
            best = np.where(better, t, best)
            arg = np.where(better, int(np.argmin(cost[j])), arg)
            prev_j = np.where(better, j, prev_j)
        for m, j in enumerate(reversed(valid)):
            gap = i - j
            if gap > K:
                break
            if m >= 10 and m % 5:
                continue
            d = np.abs(z[:, None] - levels[j][None, :])
            allow = C.MAX_GRADE * step * gap * 1.5 + 0.3
            tot = np.where(d <= allow, 0.3 * d, big) + cost[j][None, :] + skip * (gap - 1)
            k = np.argmin(tot, 1)
            t = tot[np.arange(len(z)), k] + base
            better = t < best
            best = np.where(better, t, best)
            arg = np.where(better, k, arg)
            prev_j = np.where(better, j, prev_j)
        cost[i] = best
        back[i] = (prev_j, arg)
        valid.append(i)
    if not valid:
        return out
    ends = [(cost[j].min() + skip * (n - 1 - j), j) for j in valid[-K:]]
    _, i = min(ends)
    k = int(np.argmin(cost[i]))
    while i >= 0:
        out[i] = levels[i][k]
        pj, pk = back[i]
        i, k = int(pj[k]), int(pk[k])
    return out


def _fill(z_meas, prior, closed):
    """実測の無い所を、前後の実測との差をなめらかにつないだ推定値で埋める。"""
    have = ~np.isnan(z_meas)
    if not have.any():
        return prior.copy()
    n = len(z_meas)
    diff = z_meas - prior
    x = np.arange(n)
    xs, ds = x[have], diff[have]
    if closed:
        xs = np.concatenate([xs - n, xs, xs + n])
        ds = np.concatenate([ds, ds, ds])
    fill = np.interp(x, xs, ds)
    out = np.where(have, z_meas, prior + fill)
    return out


def fit_profile(route, pl, z_start=None, smooth=10.0):
    """route の縦断を PLATEAU の路面に合わせる。合った点の割合を返す。"""
    lv = pl.levels(route.P, route.N)
    z_meas = _pick(lv, route.z, route.step, z_start)
    have = ~np.isnan(z_meas)
    z = _fill(z_meas, route.z, route.closed)
    z = _gauss(z, smooth / route.step, route.closed)
    if z_start is not None:
        w = np.clip(1 - route.s / 30.0, 0, 1)
        z = z * (1 - w) + z_start * w
    z = _limit_grade(z, route.step, route.closed)
    route.set_profile(route.ground, z)
    route.measured = have
    return float(have.mean())


def _left_edge(pl, route, i, ys):
    """i 番目の断面で、中心から左へ路面（自分の高さの段）が続く所の端。中心に路面が無ければ NaN。"""
    last = np.nan
    for y in ys:
        zs = pl.surface.heights(*(route.P[i] + route.N[i] * y))
        if not np.any(np.abs(zs - route.zat(i, y)) < 0.6):
            break
        last = y
    return last


def fit_plan(route, pl, avoid, every=5, max_shift=3.0):
    """中心線の平面位置を、PLATEAU の路面の左端（沿道側の壁高欄の足元）に合わせて横にずらす。

    右端（中央分離帯側）は上下線の床版がつながっていて端が取れないことが多いので使わない。
    avoid（分岐・合流の近くなど、路面が広がる所）と、ずれが max_shift を超える所は合わせない。
    ずらした量（点ごとの左向きの距離）を返す。
    """
    h = route.width / 2
    ys = np.arange(0.0, h.max() + max_shift + 0.5, 0.25)
    idx = np.arange(0, len(route.s), every)
    raw = np.full(len(idx), np.nan)
    for n, i in enumerate(idx):
        if route.measured[i] and not avoid[i]:
            edge = _left_edge(pl, route, i, ys)
            if abs(edge - h[i]) < max_shift:
                raw[n] = edge - h[i]
    # 外れ値を抑える（前後 50m の中央値）
    k = max(1, int(25 / (route.step * every)))
    med = np.array([np.nanmedian(raw[max(0, n - k): n + k + 1]) if np.isfinite(raw[max(0, n - k): n + k + 1]).any()
                    else np.nan for n in range(len(raw))])
    have = np.isfinite(med)
    if not have.any():
        return np.zeros(len(route.s))
    # 合わせられない所は 0 に戻す（長い欠けの途中で勝手にずれないように）
    shift = np.interp(route.s, route.s[idx][have], med[have], period=route.length if route.closed else None)
    gap = np.interp(route.s, route.s[idx], (~have).astype(float), period=route.length if route.closed else None) > 0.5
    shift[gap] = 0.0
    shift = _gauss(shift, 20.0 / route.step, route.closed)
    route.move(route.P + route.N * shift[:, None])
    return shift
