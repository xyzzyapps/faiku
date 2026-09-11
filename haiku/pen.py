"""Invented writing motor. Not a MaleCNS output."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fly import Fly


@dataclass(frozen=True)
class Stroke:
    """Canvas command produced by a PenClass, not by the connectome."""

    dx: float
    dy: float
    down: bool

    def vector(self) -> np.ndarray:
        return np.array([self.dx, self.dy, 1.0 if self.down else -1.0], dtype=np.float32)


class PenClass(Fly):
    """Fly plus a decoder that turns `code` into a 2-D stroke.

    HaikuBrain / TransformerFly inherit this. `Fly` stays pen-free.
    """

    def init_pen(self, n: int | None = None, seed: int = 7) -> None:
        n = int(n or self.n_channels)
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.08, (n, 3)).astype(np.float32)
        self._dn_bias = 0.0

    def stroke(self) -> Stroke:
        code = np.asarray(self.code, dtype=np.float32).reshape(-1)
        w = np.asarray(self.W, dtype=np.float32)
        if w.shape[0] != code.size:
            w = w.reshape(code.size, -1) if w.size >= code.size * 3 else np.zeros((code.size, 3), np.float32)
        act = np.tanh(code @ w[:, :3])
        act = np.asarray(act, dtype=np.float32).reshape(-1)
        if act.size < 3:
            act = np.pad(act, (0, 3 - act.size))
        act[0] += 0.15 * float(getattr(self, "_dn_bias", 0.0))
        act = np.tanh(act)
        return Stroke(dx=float(act[0]), dy=float(act[1]), down=bool(act[2] > 0.05))

    def learn_pen(self, action: np.ndarray, reward: float) -> None:
        r = float(reward)
        vec = np.tanh(np.asarray(action, dtype=np.float32).reshape(-1)[:3])
        if vec.size < 3:
            vec = self.stroke().vector()[:3]
        code = np.asarray(self.code, dtype=np.float32).reshape(-1)
        if self.W.shape[0] != code.size:
            return
        self.W += np.float32(0.18 * r) * np.outer(code, vec[:3])
        self.W = np.clip(self.W, -1.5, 1.5)
