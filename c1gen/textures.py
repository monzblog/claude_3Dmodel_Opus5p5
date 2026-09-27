"""材質用のテクスチャを numpy で作る（継ぎ目なく繰り返せる RGBA、0〜1）。

どれも「u = 道路に沿った向き、v = 横または上向き」に貼る前提で、
TILES に 1 枚が何メートル分かを持つ。
"""
import numpy as np

RNG = np.random.default_rng(1)


def _noise(h, w, scale, rng=RNG):
    """周期的（端でつながる）なノイズ。scale は特徴の大きさ（ピクセル）。"""
    white = rng.standard_normal((h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    filt = np.exp(-((fx ** 2 + fy ** 2) * (scale ** 2) * 4))
    n = np.real(np.fft.ifft2(np.fft.fft2(white) * filt))
    return (n - n.mean()) / (n.std() + 1e-9)


def _rgba(gray_or_rgb, alpha=1.0):
    a = np.asarray(gray_or_rgb, float)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    out = np.concatenate([np.clip(a, 0, 1), np.full(a.shape[:2] + (1,), alpha)], -1)
    return out.astype(np.float32)


def asphalt(n=512):
    g = 0.16 + 0.025 * _noise(n, n, 40) + 0.02 * _noise(n, n, 3) + 0.015 * _noise(n, n, 1)
    speck = RNG.random((n, n)) > 0.985
    g = np.where(speck, g + 0.12, g)
    return _rgba(g)


def red_pavement(n=256):
    base = np.array([0.55, 0.16, 0.13])
    v = 0.05 * _noise(n, n, 20)[..., None] + 0.03 * _noise(n, n, 2)[..., None]
    return _rgba(base + v)


def concrete(n=512, dirt=1.0):
    g = 0.62 + 0.04 * _noise(n, n, 30) + 0.02 * _noise(n, n, 2)
    # 上から垂れた雨だれの筋（v が上向き）
    streak = np.clip(_noise(1, n, 4)[0], 0, None) * 0.05 * dirt
    v = np.linspace(0, 1, n)[:, None]
    g = g - streak[None, :] * (0.3 + 0.7 * v)
    # 下の方は排気ガスで黒ずむ
    g = g - 0.10 * dirt * np.exp(-v / 0.15)
    # 型枠の継ぎ目（u の端）
    g[:, :2] -= 0.08
    rgb = np.repeat(g[..., None], 3, -1) * np.array([1.0, 0.99, 0.96])
    return _rgba(rgb)


def sound_panel(w=512, h=256):
    """金属の吸音板。横長の板の中に細い横リブ。板の境目に枠。"""
    v = np.arange(h)[:, None] / h
    rib = 0.5 + 0.5 * np.cos(2 * np.pi * v * 10)
    g = 0.72 - 0.08 * rib + 0.015 * _noise(h, w, 8)
    g[: max(2, h // 40)] -= 0.2
    g[:, : max(2, w // 60)] -= 0.25
    rgb = np.repeat(g[..., None], 3, -1) * np.array([0.97, 0.97, 0.93])
    return _rgba(rgb)


def tunnel_tile(w=512, h=256, dirty=False):
    """白いタイルパネル（1枚 1m×0.5m）。目地は灰色。"""
    g = 0.86 + 0.015 * _noise(h, w, 10)
    if dirty:
        g = g - 0.18 - 0.05 * _noise(h, w, 20)
    for k in range(2):
        g[:, k:: w // 2] *= 0.6 + 0.15 * k
        g[k:: h // 2, :] *= 0.6 + 0.15 * k
    rgb = np.repeat(g[..., None], 3, -1) * np.array([1.0, 0.99, 0.95])
    return _rgba(rgb)


def chevron(w=128, h=192):
    """カーブの矢羽根板（黄色地に黒い「＞」）。右向き。"""
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    d = np.abs(y - 0.5)
    band = np.mod((x - 0.35 + d * 0.9) * 2.2, 1.0) < 0.45
    rgb = np.where(band[..., None], [0.04, 0.04, 0.04], [0.95, 0.75, 0.05])
    rgb[:4] = rgb[-4:] = rgb[:, :4] = rgb[:, -4:] = 0.05
    return _rgba(rgb)


def cushion_drum(w=128, h=128):
    v = np.arange(h)[:, None] / h
    band = np.mod(v * 4, 1.0) < 0.3
    rgb = np.where(band[..., None], [0.05, 0.05, 0.05], [0.95, 0.72, 0.05]) * np.ones((h, w, 1))
    return _rgba(rgb)


# 名前: (作る関数, 1枚が u 方向に何m, v 方向に何m)
def facade(wall, glass, win_w=0.6, win_h=0.55, w=256, h=150, seed=0):
    """ビルの外壁（1枚 = 横 6m・1階分）。窓 2 つ、ガラスは少しずつ明るさを変える。

    win_w, win_h は窓が 1 区画に占める割合。
    """
    rng = np.random.default_rng(seed)
    img = np.ones((h, w, 3)) * np.asarray(wall, float)
    img *= (1 + 0.04 * _noise(h, w, 6, rng))[..., None]
    for k in range(2):
        x0 = int((k + (1 - win_w) / 2) * w / 2)
        x1 = int((k + (1 + win_w) / 2) * w / 2)
        y0 = int(h * (1 - win_h) * 0.6)
        y1 = y0 + int(h * win_h)
        tone = np.asarray(glass, float) * (0.85 + 0.3 * rng.random())
        grad = np.linspace(1.15, 0.85, y1 - y0)[:, None, None]
        img[y0:y1, x0:x1] = tone * grad
    return _rgba(img)


TILES = {
    "Asphalt": (asphalt, 6.0, 6.0),
    "RedPavement": (red_pavement, 4.0, 4.0),
    "Concrete": (concrete, 4.0, 1.2),
    "RetainingWall": (lambda: concrete(dirt=2.0), 6.0, 8.0),
    "SoundPanel": (sound_panel, 2.0, 0.5),
    "TunnelTile": (tunnel_tile, 2.0, 1.0),
    "TunnelTileDirty": (lambda: tunnel_tile(dirty=True), 2.0, 1.0),
    "Chevron": (chevron, 1.0, 1.0),
    "CushionDrum": (cushion_drum, 3.0, 1.0),
    "FacadeGlass": (lambda: facade((0.55, 0.57, 0.6), (0.22, 0.3, 0.38), win_w=0.9, win_h=0.8, seed=1), 6.0, 3.5),
    "FacadeBeige": (lambda: facade((0.74, 0.68, 0.57), (0.14, 0.17, 0.21), seed=2), 6.0, 3.5),
    "FacadeGray": (lambda: facade((0.6, 0.6, 0.6), (0.16, 0.19, 0.23), win_w=0.7, seed=3), 6.0, 3.5),
    "FacadeDark": (lambda: facade((0.26, 0.26, 0.28), (0.3, 0.36, 0.42), win_w=0.8, win_h=0.7, seed=4), 6.0, 3.5),
}
