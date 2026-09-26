"""中心線を等間隔に取り直し、幅・高さ（縦断）を決める。"""
import numpy as np

from . import config
from .proj import to_latlon


def _gauss(values, sigma_samples, closed):
    if sigma_samples <= 0:
        return values.copy()
    r = int(3 * sigma_samples)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2)
    k /= k.sum()
    if closed:
        pad = np.concatenate([values[-r:], values, values[:r]]) if r else values
    else:
        pad = np.concatenate([np.repeat(values[:1], r, 0), values, np.repeat(values[-1:], r, 0)])
    if values.ndim == 1:
        return np.convolve(pad, k, mode="valid")
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(values.shape[1])], 1)


class Route:
    """等間隔（config.SAMPLE_STEP）の中心線と、各点の属性。"""

    def __init__(self, name, pts, attrs, closed, step=config.SAMPLE_STEP):
        self.name = name
        self.closed = closed
        pts = np.asarray(pts, float)
        if closed:
            pts_c = np.vstack([pts, pts[:1]])
        else:
            pts_c = pts
        seg = np.linalg.norm(np.diff(pts_c, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        total = cum[-1]
        n = max(2, int(round(total / step)))
        if closed:
            s = np.linspace(0, total, n, endpoint=False)
        else:
            s = np.linspace(0, total, n + 1)
        self.length = float(total)
        self.step = float(total / n)
        self.s = s
        x = np.interp(s, cum, pts_c[:, 0])
        y = np.interp(s, cum, pts_c[:, 1])
        P = np.stack([x, y], 1)
        # OSMの折れ線の角を少しだけ丸める（始点・終点の位置は保つ）
        sm = _gauss(P, config.XY_SMOOTH_SIGMA / self.step, closed)
        if not closed:
            w = np.clip(np.minimum(s, total - s) / 20.0, 0, 1)[:, None]
            sm = P * (1 - w) + sm * w
        self.P = sm
        seg_idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(attrs) - 1)
        self.lanes = np.array([attrs[i]["lanes"] for i in seg_idx], int)
        self.bridge = np.array([attrs[i]["bridge"] for i in seg_idx], bool)
        self.tunnel_tag = np.array([attrs[i]["tunnel"] for i in seg_idx], bool)
        self.cutting_tag = np.array([attrs[i]["cutting"] for i in seg_idx], bool)
        self.layer = np.array([attrs[i]["layer"] for i in seg_idx], int)
        self._frames()
        width = (
            self.lanes * config.LANE_WIDTH + config.SHOULDER_LEFT + config.SHOULDER_RIGHT
        ).astype(float)
        # 車線数の変わり目は 60m ほどでなだらかに幅を変える
        self.width = _gauss(width, 20.0 / self.step, closed)
        self.lat, self.lon = to_latlon(self.P[:, 0], self.P[:, 1])

    def _frames(self):
        P = self.P
        if self.closed:
            d = np.roll(P, -1, 0) - np.roll(P, 1, 0)
        else:
            d = np.gradient(P, axis=0)
        T = d / np.linalg.norm(d, axis=1, keepdims=True)
        self.T = T
        self.N = np.stack([-T[:, 1], T[:, 0]], 1)  # 進行方向の左

    def target_offset(self):
        """タグから決めた、地面に対する路面の高さ（平滑化前）。"""
        off = np.zeros(len(self.s))
        br = self.bridge | (self.layer > 0)
        off[br] = config.BRIDGE_BASE_HEIGHT + config.BRIDGE_LAYER_STEP * np.clip(self.layer[br] - 1, 0, None)
        off[self.cutting_tag] = -config.CUTTING_DEPTH
        tun = self.tunnel_tag | (self.layer < 0)
        off[tun] = -config.TUNNEL_DEPTH + config.BRIDGE_LAYER_STEP * np.clip(self.layer[tun] + 1, None, 0)
        return off

    def set_profile(self, ground, z):
        self.ground = np.asarray(ground, float)
        self.z = np.asarray(z, float)
        clear = self.z - self.ground
        self.elevated = clear > config.ELEVATED_MIN
        self.tunnel = self.z + config.TUNNEL_HEIGHT + config.TUNNEL_COVER_MIN < self.ground
        self.cutting = (self.z < self.ground - 0.5) & ~self.tunnel


def loop_profile(route, dem):
    ground = dem.sample(route.lat, route.lon)
    target = ground + route.target_offset()
    z = _gauss(target, config.PROFILE_SIGMA / route.step, route.closed)
    z = _limit_grade(z, route.step, route.closed)
    route.set_profile(ground, z)


def ramp_profile(route, dem, z0):
    """分岐点の本線の高さ z0 から、ランプ自身の目標高さへ勾配制限付きで近づける。"""
    ground = dem.sample(route.lat, route.lon)
    target = _gauss(ground + route.target_offset(), config.PROFILE_SIGMA / route.step, False)
    reach = config.MAX_GRADE * 0.8 * route.s
    z = z0 + np.clip(target - z0, -reach, reach)
    route.set_profile(ground, z)


def _limit_grade(z, step, closed, iters=200):
    lim = config.MAX_GRADE * step
    z = z.copy()
    for _ in range(iters):
        nxt = np.roll(z, -1) if closed else np.append(z[1:], z[-1])
        d = nxt - z
        over = np.abs(d) > lim
        if not over.any():
            break
        corr = np.where(over, (np.abs(d) - lim) * np.sign(d) / 2, 0)
        z += corr
        if closed:
            z -= np.roll(corr, 1)
        else:
            z[1:] -= corr[:-1]
    return z
