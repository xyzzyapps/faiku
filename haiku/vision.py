"""Tiny MLP: glyph raster → vision channels. Not a fly retina."""
from __future__ import annotations

import numpy as np

from .channels import N_VISION


def _pool(img: np.ndarray, side: int = 8) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim > 2:
        a = a[..., 0]
    h, w = a.shape
    out = np.zeros((side, side), dtype=np.float32)
    for y in range(side):
        for x in range(side):
            y0, y1 = h * y // side, h * (y + 1) // side
            x0, x1 = w * x // side, w * (x + 1) // side
            out[y, x] = float(a[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else 0.0
    return out.ravel()


class VisionMLP:
    def __init__(self, n_out: int = N_VISION, seed: int = 3) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.12, (64, 16)).astype(np.float32)
        self.b1 = np.zeros(16, dtype=np.float32)
        self.W2 = rng.normal(0, 0.12, (16, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, dtype=np.float32)
        self._x = np.zeros(64, dtype=np.float32)
        self._h = np.zeros(16, dtype=np.float32)
        self._y = np.zeros(n_out, dtype=np.float32)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x = _pool(img, 8)
        h = np.tanh(x @ self.W1 + self.b1)
        y = np.tanh(h @ self.W2 + self.b2)
        self._x, self._h, self._y = x, h, y
        return y

    def reinforce(self, reward: float, eta: float = 0.01) -> None:
        r = float(reward)
        if abs(r) < 1e-8:
            return
        dy = r * (1.0 - self._y * self._y)
        self.W2 += eta * np.outer(self._h, dy)
        self.b2 += eta * dy
        dh = (dy @ self.W2.T) * (1.0 - self._h * self._h)
        self.W1 += eta * np.outer(self._x, dh)
        self.b1 += eta * dh
