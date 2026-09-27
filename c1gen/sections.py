"""区間ごとの見た目の種類（壁の種類・カーブ・車線変更禁止）を決める。"""
import json
import os

import numpy as np
import shapely
from shapely.geometry import LineString

from . import config as C
from .proj import to_xy

# 壁の種類
SOUND = "sound"      # 壁高欄＋遮音壁（金属の吸音板＋透明板）
RAIL = "rail"        # 壁高欄＋金属の防護柵（川の上など、沿道に建物が迫らない所）
TUNNEL = "tunnel"    # トンネル（壁高欄の代わりに点検用通路）
CUTTING = "cutting"  # 掘割（壁高欄＋擁壁）


def _zone_polygon(water_path):
    """川の上を通る区間の範囲。OSM の川のデータがあればそれを、無ければ config の概略線を使う。"""
    lines = []
    if water_path and os.path.exists(water_path):
        with open(water_path) as f:
            data = json.load(f)
        for el in data.get("elements", []):
            if el.get("type") == "way" and "geometry" in el:
                g = el["geometry"]
                x, y = to_xy([p["lat"] for p in g], [p["lon"] for p in g])
                lines.append((np.column_stack([x, y]), C.WATER_BUFFER))
    if not lines:
        for pts in C.RAIL_ZONES:
            la, lo = zip(*pts)
            x, y = to_xy(la, lo)
            lines.append((np.column_stack([x, y]), C.RAIL_ZONE_RADIUS))
    polys = [LineString(p).buffer(w) for p, w in lines if len(p) >= 2]
    return shapely.union_all(polys) if polys else None


def wall_types(route, water_path=None):
    """左側（沿道側）の壁の種類を点ごとに返す。"""
    kind = np.full(len(route.s), SOUND, dtype=object)
    zone = _zone_polygon(water_path)
    if zone is not None:
        kind[shapely.contains_xy(zone, route.P[:, 0], route.P[:, 1]) & route.elevated] = RAIL
    kind[route.cutting] = CUTTING
    kind[route.tunnel] = TUNNEL
    return kind


def curvature(route):
    """符号付き曲率（左カーブが正）。"""
    h = np.unwrap(np.arctan2(route.T[:, 1], route.T[:, 0]))
    if route.closed:
        d = np.diff(np.append(h, h[0] + (h[-1] - h[0]) + (h[1] - h[0])))
    else:
        d = np.gradient(h)
    k = d / route.step
    r = int(10 / route.step)
    ker = np.ones(2 * r + 1) / (2 * r + 1)
    pad = np.concatenate([k[-r:], k, k[:r]]) if route.closed else np.pad(k, r, mode="edge")
    return np.convolve(pad, ker, mode="valid")


def superelevation(route, speed=None):
    """横断勾配 bank（左へ 1m 進むと上がる高さ）を曲率から決める。

    直線と緩い右カーブは路肩側（左）へ CROSS_SLOPE 下げたまま。
    必要な片勾配が CROSS_SLOPE を超えるカーブは内側へ下げる（右カーブなら左が上がる）。
    """
    if speed is None:
        speed = C.DESIGN_SPEED if route.closed else C.RAMP_DESIGN_SPEED
    k = curvature(route)
    need = C.CANT_FACTOR * speed ** 2 * np.abs(k) / 127.0
    e = np.clip(need, C.CROSS_SLOPE, C.MAX_CANT)
    bank = np.where(k > 0, -e, np.where(need > C.CROSS_SLOPE, e, -C.CROSS_SLOPE))
    sigma = C.CANT_TRANSITION / 3 / route.step
    r = int(3 * sigma)
    if r == 0:
        return bank
    ker = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    ker /= ker.sum()
    pad = np.concatenate([bank[-r:], bank, bank[:r]]) if route.closed else np.pad(bank, r, mode="edge")
    return np.convolve(pad, ker, mode="valid")


def dilate(mask, meters, step, closed):
    n = int(meters / step)
    out = mask.copy()
    for s in range(1, n + 1):
        if closed:
            out |= np.roll(mask, s) | np.roll(mask, -s)
        else:
            out[s:] |= mask[:-s]
            out[:-s] |= mask[s:]
    return out


def no_lane_change(route, junction_idx):
    """車線変更禁止（黄色の実線）にする所: トンネル内、分岐・合流の前後、急カーブ。"""
    m = route.tunnel.copy()
    j = np.zeros(len(route.s), bool)
    j[list(junction_idx)] = True
    m |= dilate(j, C.YELLOW_NEAR_JUNCTION, route.step, route.closed)
    m |= np.abs(curvature(route)) > 1.0 / C.SHARP_CURVE_RADIUS
    return m
