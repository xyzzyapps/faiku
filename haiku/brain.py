"""Mushroom-body RL (Hige / PAM LTD · PPL1 LTP) plus optional MaleCNS LIF."""
from __future__ import annotations

import math

import numpy as np

N_KC = 8


def _exp(v: float, tau: float, dt: float) -> float:
    return 0.0 if tau <= 1e-6 else v * math.exp(-dt / tau)


class HaikuBrain:
    """8 KC place/mora channels, PAM/PPL1 traces, 8×3 motor map."""

    def __init__(self, connectome: bool = True, backend: str = "vulkan") -> None:
        self.kc = np.zeros(N_KC, dtype=np.float32)
        self.mbon_avoid = np.ones(N_KC, dtype=np.float32)
        self.pam = 0.0
        self.ppl1 = 0.0
        self.valence = 0.0
        self.W = np.zeros((N_KC, 3), dtype=np.float32)
        rng = np.random.default_rng(7)
        self.W[:] = rng.normal(0, 0.08, self.W.shape).astype(np.float32)
        self.channel = 0
        self.using_connectome = False
        self.lif_backend = "off"
        self.spikes_last = 0
        self.n_neurons = N_KC + 9
        self.n_edges = 0
        self._net = None
        self._graph = None
        self._kc_edges: dict[int, np.ndarray] = {}
        self._last_spiked = np.zeros(0, dtype=np.int32)
        self.kc_mbon_updates = 0
        self.kc_mbon_dw = 0.0
        if connectome:
            self._load_connectome(backend)

    def _load_connectome(self, backend: str) -> None:
        from .connectome import available, load_graph
        from .lif import create_lif

        if not available():
            print("haiku: MaleCNS pack missing; run python tools/pack_malecns.py", flush=True)
            return
        graph = load_graph()
        if graph is None:
            return
        prefer = "cpu" if backend == "cpu" else "vulkan"
        net = create_lif(graph, dt_ms=1.0, prefer=prefer)
        self._graph = graph
        self._net = net
        self.using_connectome = True
        self.lif_backend = getattr(net, "backend", prefer)
        self.n_neurons = graph.n
        self.n_edges = graph.n_edges
        self._index_kc_edges()
        print(f"haiku: MaleCNS LIF {self.lif_backend}  n={graph.n}  edges={graph.n_edges}", flush=True)

    def _index_kc_edges(self) -> None:
        g = self._graph
        if g is None or g.kc_mbon_pre.size == 0:
            return
        buckets: dict[int, list[int]] = {}
        for e, pre in enumerate(g.kc_mbon_pre.tolist()):
            buckets.setdefault(int(pre), []).append(int(g.kc_mbon_edge[e]))
        self._kc_edges = {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}

    def encode_mora(self, mora_index: int) -> None:
        self.channel = int(mora_index) % N_KC

    def step(self, dt: float, odor: float, sugar: float, shock: float) -> np.ndarray:
        if self.using_connectome and self._net is not None:
            return self._step_cns(dt, odor, sugar, shock)
        return self._step_mb(dt, odor, sugar, shock)

    def _step_mb(self, dt: float, odor: float, sugar: float, shock: float) -> np.ndarray:
        self.kc *= np.float32(math.exp(-dt / 0.18))
        self.kc[self.channel] = min(1.0, float(self.kc[self.channel]) + 0.55 * odor)
        self.pam = min(1.4, _exp(self.pam, 0.45, dt) + 0.9 * sugar)
        self.ppl1 = min(1.4, _exp(self.ppl1, 0.35, dt) + 0.9 * shock)
        eta_p = 0.12 * min(1.0, self.pam)
        eta_n = 0.10 * min(1.0, self.ppl1)
        for i in range(N_KC):
            if self.kc[i] > 0.2 and eta_p > 0.02:
                self.mbon_avoid[i] *= 1.0 - eta_p * float(self.kc[i])
            if self.kc[i] > 0.2 and eta_n > 0.02:
                self.mbon_avoid[i] = min(1.6, float(self.mbon_avoid[i]) + eta_n * 0.25 * float(self.kc[i]))
            self.mbon_avoid[i] = min(1.6, max(0.05, float(self.mbon_avoid[i])))
        avoid = float(np.dot(self.kc, self.mbon_avoid) / N_KC)
        self.valence = max(-1.0, min(1.0, 0.55 - avoid + 0.25 * self.pam - 0.35 * self.ppl1))
        act = self.kc @ self.W
        return np.tanh(act).astype(np.float32)

    def _step_cns(self, dt: float, odor: float, sugar: float, shock: float) -> np.ndarray:
        net = self._net
        gph = self._graph
        g = gph.groups
        net.reset_drive()
        net.inject(g.get("LB3c", np.zeros(0, np.uint32)), 10.0 * sugar)
        net.inject(g.get("PAM", np.zeros(0, np.uint32)), 11.0 * sugar)
        net.inject(g.get("PPL1", np.zeros(0, np.uint32)), 11.3 * shock)
        kc = g.get("KC", np.zeros(0, np.uint32))
        if kc.size:
            chunk = max(1, kc.size // N_KC)
            lo = self.channel * chunk
            hi = min(kc.size, lo + chunk)
            net.inject(kc[lo:hi], 7.5 * (0.2 + 0.8 * odor))
        steps = max(4, min(12, int(round(dt / 0.001))))
        if getattr(net, "device", None) is not None:
            steps = min(steps, 8)
        spiked = net.advance(steps)
        self._last_spiked = spiked
        self.spikes_last = int(spiked.size)
        hit = np.zeros(gph.n, dtype=bool)
        if spiked.size:
            hit[spiked] = True
        for i in range(N_KC):
            if kc.size:
                chunk = max(1, kc.size // N_KC)
                part = kc[i * chunk : min(kc.size, (i + 1) * chunk)]
                rate = float(hit[part].mean()) if part.size else 0.0
            else:
                rate = 0.0
            self.kc[i] = 0.7 * float(self.kc[i]) + 0.3 * min(1.0, rate * 8)
        self.pam = min(1.4, _exp(self.pam, 0.45, dt) + 0.85 * sugar)
        self.ppl1 = min(1.4, _exp(self.ppl1, 0.35, dt) + 0.90 * shock)
        eta_p = 0.12 * min(1.0, self.pam)
        eta_n = 0.10 * min(1.0, self.ppl1)
        self._touch_kc_mbon(eta_p, eta_n)
        for i in range(N_KC):
            if self.kc[i] > 0.25 and eta_p > 1e-8:
                self.mbon_avoid[i] *= 1.0 - eta_p * float(self.kc[i])
            if self.kc[i] > 0.25 and eta_n > 1e-8:
                self.mbon_avoid[i] = min(1.6, float(self.mbon_avoid[i]) + eta_n * 0.25 * float(self.kc[i]))
        avoid = float(np.dot(self.kc, self.mbon_avoid) / N_KC)
        self.valence = max(-1.0, min(1.0, 0.55 - avoid + 0.25 * self.pam - 0.35 * self.ppl1))
        dn = g.get("DN", np.zeros(0, np.uint32))
        extra = 0.0
        if dn.size:
            extra = float(hit[dn].mean()) * 2.0 - 0.2
        act = self.kc @ self.W
        act[0] += 0.15 * extra
        return np.tanh(act).astype(np.float32)

    def _eligible_kc(self) -> np.ndarray:
        gph = self._graph
        if gph is None:
            return np.zeros(0, dtype=np.int32)
        kc = gph.groups.get("KC", np.zeros(0, np.uint32))
        if kc.size == 0:
            return np.zeros(0, dtype=np.int32)
        chunk = max(1, kc.size // N_KC)
        lo = self.channel * chunk
        hi = min(kc.size, lo + chunk)
        channel = kc[lo:hi].astype(np.int32, copy=False)
        spiked = self._last_spiked
        if spiked.size == 0:
            return channel
        hit = np.intersect1d(channel, spiked, assume_unique=False)
        return hit if hit.size else channel

    def _touch_kc_mbon(self, eta_p: float, eta_n: float) -> None:
        """Hige-style PAM LTD / PPL1 LTP on KC→MBON CSR edges."""
        net = self._net
        gph = self._graph
        if net is None or gph is None or not self._kc_edges:
            return
        if eta_p <= 1e-8 and eta_n <= 1e-8:
            return
        data = net.W.data
        base = gph.weight0
        dw = 0.0
        n_up = 0
        for i in self._eligible_kc().tolist():
            edges = self._kc_edges.get(int(i))
            if edges is None or len(edges) == 0:
                continue
            ei = np.asarray(edges, dtype=np.int64)
            before = data[ei].copy()
            if eta_p > 1e-8:
                data[ei] *= 1.0 - eta_p
            if eta_n > 1e-8:
                data[ei] += eta_n * 0.08 * np.abs(base[ei])
            lo = 0.05 * np.abs(base[ei])
            hi = 1.6 * np.abs(base[ei])
            mag = np.clip(np.abs(data[ei]), lo, hi)
            data[ei] = mag * np.sign(base[ei] + 1e-12)
            dw += float(np.abs(data[ei] - before).sum())
            n_up += int(ei.size)
            if hasattr(net, "update_weights"):
                net.update_weights(ei, data[ei])
        self.kc_mbon_updates += n_up
        self.kc_mbon_dw += dw

    def learn(self, action: np.ndarray, reward: float) -> None:
        """Credit this tick: motor W and MaleCNS KC→MBON from the same dopamine."""
        r = float(reward)
        sugar = max(0.0, r)
        shock = max(0.0, -r)
        self.pam = min(1.4, self.pam + 0.55 * sugar)
        self.ppl1 = min(1.4, self.ppl1 + 0.55 * shock)
        eta = 0.18 * r
        self.W += eta * np.outer(self.kc, np.tanh(action))
        self.W = np.clip(self.W, -1.5, 1.5)
        if self.using_connectome:
            self._touch_kc_mbon(0.12 * min(1.0, self.pam), 0.10 * min(1.0, self.ppl1))

    def reinforce(self, action: np.ndarray, reward: float) -> None:
        self.learn(action, reward)
