"""Hiragana as fly-font polylines (0..1). Unique parametric strokes per mora."""
from __future__ import annotations

import math

import numpy as np

N = 48


def _hash_path(ch: str, n: int = 28) -> np.ndarray:
    s = (ord(ch) * 1315423911) & 0xFFFFFFFF
    pts = []
    x, y = 0.18 + (s % 97) / 400.0, 0.22 + ((s >> 8) % 97) / 400.0
    for i in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        a = (s % 628) / 100.0
        step = 0.035 + (s % 17) / 900.0
        x = min(0.92, max(0.08, x + math.cos(a) * step))
        y = min(0.92, max(0.08, y + math.sin(a * 1.3) * step))
        pts.append((x, y))
    return np.asarray(pts, dtype=np.float32)


def _arc(cx, cy, r, a0, a1, n=16) -> np.ndarray:
    t = np.linspace(a0, a1, n, dtype=np.float32)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


HAND: dict[str, np.ndarray] = {
    "ふ": np.concatenate([_arc(0.45, 0.38, 0.22, 3.4, 6.0), _arc(0.52, 0.62, 0.18, 5.5, 8.2)]),
    "る": _arc(0.5, 0.5, 0.28, 0.2, 5.6),
    "い": np.array([[0.28, 0.2], [0.32, 0.8], [0.62, 0.22], [0.7, 0.78]], np.float32),
    "け": np.array([[0.3, 0.18], [0.32, 0.82], [0.3, 0.42], [0.72, 0.38], [0.58, 0.22], [0.6, 0.8]], np.float32),
    "や": np.array([[0.25, 0.35], [0.75, 0.28], [0.5, 0.22], [0.52, 0.82]], np.float32),
    "か": np.array([[0.28, 0.2], [0.3, 0.8], [0.28, 0.4], [0.7, 0.35], [0.55, 0.25], [0.62, 0.75]], np.float32),
    "わ": np.concatenate([np.array([[0.28, 0.18], [0.3, 0.82]], np.float32), _arc(0.52, 0.55, 0.22, 4.0, 7.2)]),
    "ず": np.concatenate([_arc(0.48, 0.55, 0.26, 3.8, 7.0), np.array([[0.62, 0.18], [0.78, 0.28]], np.float32)]),
    "と": np.array([[0.25, 0.28], [0.75, 0.25], [0.5, 0.22], [0.52, 0.78], [0.7, 0.7]], np.float32),
    "び": np.array([[0.3, 0.2], [0.32, 0.8], [0.55, 0.35], [0.72, 0.22], [0.7, 0.55], [0.78, 0.18]], np.float32),
    "こ": np.array([[0.28, 0.32], [0.72, 0.3], [0.3, 0.68], [0.74, 0.7]], np.float32),
    "む": _arc(0.5, 0.5, 0.3, 5.5, 11.5),
    "み": np.array([[0.25, 0.25], [0.7, 0.22], [0.35, 0.25], [0.4, 0.8], [0.72, 0.62]], np.float32),
    "の": _arc(0.5, 0.5, 0.28, 0.0, 6.0),
    "お": np.concatenate([np.array([[0.35, 0.18], [0.38, 0.8]], np.float32), _arc(0.55, 0.5, 0.22, 3.5, 7.4)]),
    "し": np.array([[0.35, 0.18], [0.38, 0.72], [0.62, 0.78]], np.float32),
    "ず": _arc(0.5, 0.52, 0.26, 3.5, 7.2),
    "せ": np.array([[0.25, 0.35], [0.75, 0.32], [0.5, 0.2], [0.52, 0.8], [0.28, 0.7], [0.72, 0.68]], np.float32),
    "ん": _arc(0.48, 0.48, 0.26, 4.2, 7.6),
}


def path_for(ch: str) -> np.ndarray:
    if ch in HAND:
        return HAND[ch]
    return _hash_path(ch)


def raster(ch: str, size: int = N) -> np.ndarray:
    path = path_for(ch)
    img = np.zeros((size, size), dtype=np.float32)
    if path.shape[0] < 2:
        return img
    xy = path * (size - 1)
    for i in range(len(xy) - 1):
        x0, y0 = xy[i]
        x1, y1 = xy[i + 1]
        steps = int(max(2, abs(x1 - x0), abs(y1 - y0)))
        for t in range(steps + 1):
            u = t / steps
            x = int(round(x0 + (x1 - x0) * u))
            y = int(round(y0 + (y1 - y0) * u))
            if 0 <= x < size and 0 <= y < size:
                img[y, x] = 1.0
                if x + 1 < size:
                    img[y, x + 1] = max(img[y, x + 1], 0.5)
    return img


def overlap(ink: np.ndarray, target: np.ndarray) -> float:
    a = ink.ravel()
    b = target.ravel()
    den = float(b.sum()) + 1e-6
    hit = float((a * b).sum())
    extra = float((a * (1.0 - np.clip(b, 0, 1))).sum())
    return max(0.0, hit / den - 0.35 * extra / (a.size))
