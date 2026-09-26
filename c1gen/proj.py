"""緯度経度と Blender 座標（原点からのメートル、X=東, Y=北）の変換。"""
import numpy as np
from pyproj import Transformer

from . import config

_fwd = Transformer.from_crs("EPSG:4326", config.CRS, always_xy=True)
_inv = Transformer.from_crs(config.CRS, "EPSG:4326", always_xy=True)
_ox, _oy = _fwd.transform(config.ORIGIN_LATLON[1], config.ORIGIN_LATLON[0])


def to_xy(lat, lon):
    x, y = _fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x) - _ox, np.asarray(y) - _oy


def to_latlon(x, y):
    lon, lat = _inv.transform(np.asarray(x, float) + _ox, np.asarray(y, float) + _oy)
    return np.asarray(lat), np.asarray(lon)
