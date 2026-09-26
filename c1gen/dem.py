"""国土地理院の標高タイル（テキスト形式）の取得と標高の補間。

dem5a（5mメッシュ, z15）を優先し、欠けている所は dem（10mメッシュ, z14）で埋める。
"""
import math
import os
import time

import numpy as np
import requests

from . import config

SOURCES = [("dem5a", 15), ("dem", 14)]
URL = "https://cyberjapandata.gsi.go.jp/xyz/{name}/{z}/{x}/{y}.txt"


def _tile_xy(lat, lon, z):
    n = 2 ** z
    px = (lon + 180.0) / 360.0 * n * 256
    lat_r = np.radians(lat)
    py = (1 - np.log(np.tan(lat_r) + 1 / np.cos(lat_r)) / math.pi) / 2 * n * 256
    return px, py


def _tile_path(root, name, z, x, y):
    return os.path.join(root, name, str(z), str(x), f"{y}.txt")


def fetch(root, bbox=config.BBOX):
    s, w, n, e = bbox
    for name, z in SOURCES:
        (x0, y1), (x1, y0) = [
            tuple(int(v // 256) for v in _tile_xy(la, lo, z)) for la, lo in ((s, w), (n, e))
        ]
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                path = _tile_path(root, name, z, x, y)
                if os.path.exists(path):
                    continue
                r = requests.get(URL.format(name=name, z=z, x=x, y=y),
                                 headers={"User-Agent": config.USER_AGENT}, timeout=60)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    # 404 はその範囲にデータが無いという意味なので空ファイルで記録
                    f.write(r.text if r.status_code == 200 else "")
                time.sleep(0.2)


def _load_tile(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read().strip()
    if not text:
        return None
    rows = [[float(v) if v != "e" else np.nan for v in line.split(",")] for line in text.splitlines()]
    return np.array(rows, float)


class DEM:
    """ディスク上のタイルから標高を引く。タイルが無ければ一定値を返す。"""

    def __init__(self, root, fallback=4.0):
        self.root = root
        self.fallback = fallback
        self._cache = {}
        self.available = any(
            os.path.isdir(os.path.join(root, name)) for name, _ in SOURCES
        )

    def _tile(self, name, z, x, y):
        key = (name, z, x, y)
        if key not in self._cache:
            self._cache[key] = _load_tile(_tile_path(self.root, name, z, x, y))
        return self._cache[key]

    def _sample_source(self, name, z, lat, lon):
        px, py = _tile_xy(lat, lon, z)
        px, py = px - 0.5, py - 0.5
        out = np.full(lat.shape, np.nan)
        ix0, iy0 = np.floor(px).astype(int), np.floor(py).astype(int)
        fx, fy = px - ix0, py - iy0
        acc = np.zeros(lat.shape)
        wsum = np.zeros(lat.shape)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            gx, gy = ix0 + dx, iy0 + dy
            w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
            vals = np.full(lat.shape, np.nan)
            tx, ty = gx // 256, gy // 256
            for key in set(zip(tx.tolist(), ty.tolist())):
                t = self._tile(name, z, *key)
                if t is None:
                    continue
                m = (tx == key[0]) & (ty == key[1])
                vals[m] = t[gy[m] % 256, gx[m] % 256]
            ok = ~np.isnan(vals)
            acc[ok] += w[ok] * vals[ok]
            wsum[ok] += w[ok]
        good = wsum > 0.25
        out[good] = acc[good] / wsum[good]
        return out

    def sample(self, lat, lon):
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        if not self.available:
            return np.full(lat.shape, self.fallback)
        out = np.full(lat.shape, np.nan)
        for name, z in SOURCES:
            miss = np.isnan(out)
            if not miss.any():
                break
            out[miss] = self._sample_source(name, z, lat[miss], lon[miss])
        # 水面などデータの無い所は周りの平均で埋める
        if np.isnan(out).all():
            out[:] = self.fallback
        else:
            out[np.isnan(out)] = np.nanmedian(out)
        return out
