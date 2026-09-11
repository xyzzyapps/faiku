"""Mushroom-body RL (Hige / PAM LTD · PPL1 LTP) plus optional MaleCNS LIF."""
from __future__ import annotations

import math

import numpy as np

from .channels import N_CHANNELS, N_KC, N_VISION, layout
from .pen import PenClass
from .vision import VisionMLP


def _exp(v: float, tau: float, dt: float) -> float:
    return 0.0 if tau <= 1e-6 else v * math.exp(-dt / tau)


class HaikuBrain(PenClass):
    """MaleCNS / 8-channel MB, plus a PenClass stroke decoder."""

    n_channels = N_CHANNELS

    def __init__(
        self,
        connectome: bool = True,
        backend: str = "vulkan",
        vision: bool = True,
    ) -> None:
        self.use_vision = bool(vision)
        self.n_odor, self.n_vis, n = layout(self.use_vision)
        self.n_channels = n
        self.kc = np.zeros(n, dtype=np.float32)
        self.mbon_avoid = np.ones(n, dtype=np.float32)
        self.pam = 0.0
        self.ppl1 = 0.0
        self.valence = 0.0
        self.init_pen(n)
        self.vision = VisionMLP(N_VISION) if self.use_vision else None
        self._vis = np.zeros(max(self.n_vis, 1), dtype=np.float32)
        if not self.use_vision:
            self._vis = np.zeros(0, dtype=np.float32)
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
        self._dn_bias = 0.0
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
        mode = "4 odor + 4 vision" if self.use_vision else "8 odor (no vision)"
        print(
            f"haiku: MaleCNS LIF {self.lif_backend}  n={graph.n}  edges={graph.n_edges}  {mode}",
            flush=True,
        )

    def _index_kc_edges(self) -> None:
        g = self._graph
        if g is None or g.kc_mbon_pre.size == 0:
            return
        buckets: dict[int, list[int]] = {}
        for e, pre in enumerate(g.kc_mbon_pre.tolist()):
            buckets.setdefault(int(pre), []).append(int(g.kc_mbon_edge[e]))
        self._kc_edges = {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}

    def encode(self, mora_index: int) -> None:
        self.channel = int(mora_index) % max(1, self.n_odor)

    def see(self, frame: np.ndarray) -> None:
        if not self.use_vision or self.vision is None:
            return
        self._vis = self.vision(frame)
        self.kc[self.n_odor :] = self._vis

    def encode_mora(self, mora_index: int) -> None:
        self.encode(mora_index)

    def forward(
        self,
        dt: float = 0.016,
        odor: float = 1.0,
        sugar: float = 0.0,
        shock: float = 0.0,
    ) -> None:
        if self.using_connectome and self._net is not None:
            self._step_cns(dt, odor, sugar, shock)
        else:
            self._step_mb(dt, odor, sugar, shock)

    def step(self, dt: float, odor: float, sugar: float, shock: float) -> None:
        self.forward(dt, odor, sugar, shock)

    @property
    def code(self) -> np.ndarray:
        return self.kc

    def _step_mb(self, dt: float, odor: float, sugar: float, shock: float) -> None:
        self.kc[: self.n_odor] *= np.float32(math.exp(-dt / 0.18))
        self.kc[self.channel] = min(1.0, float(self.kc[self.channel]) + 0.55 * odor)
        if self.use_vision and self.n_vis:
            self.kc[self.n_odor :] = 0.7 * self.kc[self.n_odor :] + 0.3 * self._vis
        self.pam = min(1.4, _exp(self.pam, 0.45, dt) + 0.9 * sugar)
        self.ppl1 = min(1.4, _exp(self.ppl1, 0.35, dt) + 0.9 * shock)
        eta_p = 0.12 * min(1.0, self.pam)
        eta_n = 0.10 * min(1.0, self.ppl1)
        n = self.n_channels
        for i in range(n):
            if self.kc[i] > 0.2 and eta_p > 0.02:
                self.mbon_avoid[i] *= 1.0 - eta_p * float(self.kc[i])
            if self.kc[i] > 0.2 and eta_n > 0.02:
                self.mbon_avoid[i] = min(1.6, float(self.mbon_avoid[i]) + eta_n * 0.25 * float(self.kc[i]))
            self.mbon_avoid[i] = min(1.6, max(0.05, float(self.mbon_avoid[i])))
        avoid = float(np.dot(self.kc, self.mbon_avoid) / n)
        self.valence = max(-1.0, min(1.0, 0.55 - avoid + 0.25 * self.pam - 0.35 * self.ppl1))
        self._dn_bias = 0.0

    def _step_cns(self, dt: float, odor: float, sugar: float, shock: float) -> None:
        net = self._net
        gph = self._graph
        g = gph.groups
        net.reset_drive()
        net.inject(g.get("LB3c", np.zeros(0, np.uint32)), 10.0 * sugar)
        net.inject(g.get("PAM", np.zeros(0, np.uint32)), 11.0 * sugar)
        net.inject(g.get("PPL1", np.zeros(0, np.uint32)), 11.3 * shock)
        kc = g.get("KC", np.zeros(0, np.uint32))
        if kc.size:
            if self.use_vision:
                mid = kc.size // 2
                odor_kc, vis_kc = kc[:mid], kc[mid:]
                oc = max(1, odor_kc.size // self.n_odor)
                lo, hi = self.channel * oc, min(odor_kc.size, (self.channel + 1) * oc)
                net.inject(odor_kc[lo:hi], 7.5 * (0.2 + 0.8 * odor))
                vc = max(1, vis_kc.size // max(1, self.n_vis))
                for i, amp in enumerate(self._vis.tolist()):
                    a, b = i * vc, min(vis_kc.size, (i + 1) * vc)
                    net.inject(vis_kc[a:b], 7.5 * float(amp))
            else:
                chunk = max(1, kc.size // self.n_odor)
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
        if kc.size:
            if self.use_vision:
                mid = kc.size // 2
                odor_kc, vis_kc = kc[:mid], kc[mid:]
                oc = max(1, odor_kc.size // self.n_odor)
                vc = max(1, vis_kc.size // max(1, self.n_vis))
                for i in range(self.n_odor):
                    part = odor_kc[i * oc : min(odor_kc.size, (i + 1) * oc)]
                    rate = float(hit[part].mean()) if part.size else 0.0
                    self.kc[i] = 0.7 * float(self.kc[i]) + 0.3 * min(1.0, rate * 8)
                for i in range(self.n_vis):
                    part = vis_kc[i * vc : min(vis_kc.size, (i + 1) * vc)]
                    rate = float(hit[part].mean()) if part.size else 0.0
                    j = self.n_odor + i
                    self.kc[j] = 0.7 * float(self.kc[j]) + 0.3 * min(1.0, rate * 8)
                self.kc[self.n_odor :] = 0.6 * self.kc[self.n_odor :] + 0.4 * self._vis
            else:
                chunk = max(1, kc.size // self.n_odor)
                for i in range(self.n_odor):
                    part = kc[i * chunk : min(kc.size, (i + 1) * chunk)]
                    rate = float(hit[part].mean()) if part.size else 0.0
                    self.kc[i] = 0.7 * float(self.kc[i]) + 0.3 * min(1.0, rate * 8)
        self.pam = min(1.4, _exp(self.pam, 0.45, dt) + 0.85 * sugar)
        self.ppl1 = min(1.4, _exp(self.ppl1, 0.35, dt) + 0.90 * shock)
        eta_p = 0.12 * min(1.0, self.pam)
        eta_n = 0.10 * min(1.0, self.ppl1)
        self._touch_kc_mbon(eta_p, eta_n)
        n = self.n_channels
        for i in range(n):
            if self.kc[i] > 0.25 and eta_p > 1e-8:
                self.mbon_avoid[i] *= 1.0 - eta_p * float(self.kc[i])
            if self.kc[i] > 0.25 and eta_n > 1e-8:
                self.mbon_avoid[i] = min(1.6, float(self.mbon_avoid[i]) + eta_n * 0.25 * float(self.kc[i]))
        avoid = float(np.dot(self.kc, self.mbon_avoid) / n)
        self.valence = max(-1.0, min(1.0, 0.55 - avoid + 0.25 * self.pam - 0.35 * self.ppl1))
        dn = g.get("DN", np.zeros(0, np.uint32))
        extra = 0.0
        if dn.size:
            extra = float(hit[dn].mean()) * 2.0 - 0.2
        self._dn_bias = extra

    def _eligible_kc(self) -> np.ndarray:
        gph = self._graph
        if gph is None:
            return np.zeros(0, dtype=np.int32)
        kc = gph.groups.get("KC", np.zeros(0, np.uint32))
        if kc.size == 0:
            return np.zeros(0, dtype=np.int32)
        if self.use_vision:
            mid = kc.size // 2
            odor_kc = kc[:mid]
            oc = max(1, odor_kc.size // self.n_odor)
            lo = self.channel * oc
            hi = min(odor_kc.size, (self.channel + 1) * oc)
            channel = odor_kc[lo:hi]
            if self._vis.size and float(np.max(np.abs(self._vis))) > 0.05:
                channel = np.concatenate([channel, kc[mid:]])
        else:
            chunk = max(1, kc.size // self.n_odor)
            lo = self.channel * chunk
            hi = min(kc.size, lo + chunk)
            channel = kc[lo:hi]
        channel = channel.astype(np.int32, copy=False)
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
        self.learn_pen(action, r)
        if self.vision is not None:
            self.vision.reinforce(r)
        if self.using_connectome:
            self._touch_kc_mbon(0.12 * min(1.0, self.pam), 0.10 * min(1.0, self.ppl1))

    def reinforce(self, action: np.ndarray, reward: float) -> None:
        self.learn(action, reward)
