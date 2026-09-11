"""Event-ish Shiu LIF on the packed MaleCNS CSR (doomfly constants)."""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix

from .connectome import MaleCNSGraph

V_REST = -52.0
V_THRESH = -45.0
TAU_V = 20.0
TAU_G = 5.0
DELAY_MS = 1.8
REFRACT_MS = 2.2


def create_lif(graph, dt_ms: float = 1.0, prefer: str = "vulkan"):
    """Build a MaleCNS LIF. prefer='vulkan' | 'cpu'. Vulkan falls back to CPU."""
    prefer = (prefer or "vulkan").lower()
    if prefer != "cpu":
        try:
            from .lif_vulkan import try_vulkan_lif

            net = try_vulkan_lif(graph, dt_ms=dt_ms)
            if net is not None:
                return net
        except Exception as exc:
            print(f"haiku: Vulkan LIF skipped ({exc})", flush=True)
    net = LifNet(graph, dt_ms=dt_ms)
    print(
        f"haiku: MaleCNS LIF on CPU  {graph.n} cells  {graph.n_edges} edges",
        flush=True,
    )
    return net


class LifNet:
    def __init__(self, graph: MaleCNSGraph, dt_ms: float = 1.0) -> None:
        self.gph = graph
        self.n = graph.n
        self.backend = "cpu"
        self.dt = float(dt_ms)
        self.ptr = graph.ptr.astype(np.int64, copy=False)
        self.post = graph.post.astype(np.int32, copy=False)
        self.weight = graph.weight
        self.W = csr_matrix(
            (self.weight, self.post, self.ptr),
            shape=(self.n, self.n),
        )
        self.v = np.full(self.n, V_REST, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.drive = np.zeros(self.n, dtype=np.float32)
        self.refract = np.zeros(self.n, dtype=np.int16)
        self.delay = max(1, int(round(DELAY_MS / self.dt)))
        self.rfc = max(1, int(round(REFRACT_MS / self.dt)))
        self._q: list[np.ndarray] = [np.zeros(0, dtype=np.int32) for _ in range(self.delay)]
        self._qi = 0
        a = math.exp(-self.dt / TAU_V)
        b = math.exp(-self.dt / TAU_G)
        self._a = a
        self._b = b
        self._gscale = (a - b) / 3.0
        self.counts = np.zeros(self.n, dtype=np.int32)

    def reset_drive(self) -> None:
        self.drive.fill(0.0)

    def inject(self, idx: np.ndarray, current: float) -> None:
        if idx.size == 0 or abs(current) < 1e-6:
            return
        self.drive[idx] += np.float32(current)

    def advance(self, steps: int = 1) -> np.ndarray:
        fired_all: list[np.ndarray] = []
        a, b, gs = self._a, self._b, self._gscale
        for _ in range(max(1, steps)):
            arrivals = self._q[self._qi]
            if arrivals.size:
                extra = np.asarray(self.W[arrivals].sum(axis=0)).ravel()
                self.g += extra.astype(np.float32, copy=False)
            self.v = np.float32(V_REST) + (self.v - np.float32(V_REST)) * a + self.drive * (1.0 - a) + self.g * gs
            self.g *= b
            self.refract = np.maximum(self.refract - 1, 0)
            fire = (self.refract == 0) & (self.v > V_THRESH)
            idx = np.flatnonzero(fire).astype(np.int32)
            if idx.size:
                self.v[idx] = V_REST
                self.g[idx] = 0.0
                self.refract[idx] = self.rfc
                self.counts[idx] += 1
            # Store in this slot after reading it; it is visited again in `delay` steps.
            self._q[self._qi] = idx
            self._qi = (self._qi + 1) % self.delay
            fired_all.append(idx)
        if not fired_all:
            return np.zeros(0, dtype=np.int32)
        return np.concatenate(fired_all)
