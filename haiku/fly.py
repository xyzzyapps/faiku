"""Abstract fly the inference loop calls.

Neural I/O only: odor in, a tick, a KC-like code. No writing tip.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Fly(ABC):
    """Sensory encode → one network tick → a population readout."""

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
    ) -> None:
        """One neural step. Updates `code`. Does not return a pen."""

    @property
    @abstractmethod
    def code(self) -> np.ndarray:
        """Kenyon-cell rates, or an equivalent readout. Not motor."""
