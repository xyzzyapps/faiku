"""Invented writing motor. Not a MaleCNS output."""
from __future__ import annotations

from abc import abstractmethod
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

    HaikuBrain inherits this. The ABC `Fly` surface stays pen-free.
    """

    @abstractmethod
    def stroke(self) -> Stroke:
        """Map the current neural readout onto paper."""
