"""Writing canvas: fly motor → ink, reward vs target kana glyph."""
from __future__ import annotations

import numpy as np

from . import glyphs
from .corpus import Poem, get_poem


class HaikuEnv:
    def __init__(self, poem: str | Poem = "basho") -> None:
        self.poem = poem if isinstance(poem, Poem) else get_poem(poem)
        self.morae = self.poem.morae()
        self.i = 0
        self.size = glyphs.N
        self.ink = np.zeros((self.size, self.size), dtype=np.float32)
        self.target = glyphs.raster(self.morae[0])
        self.x = 0.5
        self.y = 0.5
        self.pen = False
        self.t = 0
        self.steps_per_mora = 48

    @property
    def mora(self) -> str:
        return self.morae[self.i]

    def reset_glyph(self) -> None:
        self.ink.fill(0.0)
        self.target = glyphs.raster(self.mora)
        self.x, self.y = 0.22, 0.22
        self.pen = False
        self.t = 0

    def step(self, action) -> tuple[float, bool]:
        from .pen import Stroke

        if isinstance(action, Stroke):
            dx, dy, down = action.dx, action.dy, action.down
        else:
            dx, dy = float(action[0]), float(action[1])
            down = float(action[2]) > 0.05
        self.pen = down
        nx = min(0.97, max(0.03, self.x + 0.06 * dx))
        ny = min(0.97, max(0.03, self.y + 0.06 * dy))
        if self.pen:
            self._stroke(self.x, self.y, nx, ny)
        self.x, self.y = nx, ny
        self.t += 1
        done = self.t >= self.steps_per_mora
        score = glyphs.overlap(self.ink, self.target)
        reward = (score - 0.12) if done else 0.02 * (score - 0.08)
        return float(np.clip(reward, -1.0, 1.0)), done

    def _stroke(self, x0: float, y0: float, x1: float, y1: float) -> None:
        s = self.size - 1
        steps = max(2, int(max(abs(x1 - x0), abs(y1 - y0)) * s))
        for k in range(steps + 1):
            u = k / steps
            x = int(round((x0 + (x1 - x0) * u) * s))
            y = int(round((y0 + (y1 - y0) * u) * s))
            if 0 <= x < self.size and 0 <= y < self.size:
                self.ink[y, x] = 1.0

    def advance_mora(self) -> bool:
        self.i = (self.i + 1) % len(self.morae)
        self.reset_glyph()
        return self.i == 0

    def snapshot_pair(self) -> np.ndarray:
        a = self.target
        b = np.clip(self.ink, 0, 1)
        z = np.zeros_like(a)
        rgb = np.stack([b, a, z], axis=-1)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
