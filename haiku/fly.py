"""Abstract fly the inference loop calls.

`infer()` only sees this surface. A MaleCNS mushroom body, the 8-channel
stand-in, or a fake test double all implement the same three methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Fly(ABC):
    """Sensory encode → one tick → a motor action and a KC-like code."""

    n_channels: int = 8

    @abstractmethod
    def encode(self, mora_index: int) -> None:
        """Map a script cue (mora / odor bin) onto the fly's input."""

    @abstractmethod
    def forward(
        self,
        dt: float = 0.016,
        odor: float = 1.0,
        sugar: float = 0.0,
        shock: float = 0.0,
    ) -> np.ndarray:
        """One step. Returns motor `(dx, dy, pen)` or a same-shaped stand-in."""

    @property
    @abstractmethod
    def code(self) -> np.ndarray:
        """Readout used by decode hooks (Kenyon-cell rates, or equivalent)."""
