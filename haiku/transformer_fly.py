"""Tiny RL transformer with the same Fly I/O as MaleCNS / MB.

Tokens: odor, vision MLP, dopamine. Output: `code` of length N_CHANNELS.
The pen is still PenClass (8×3 W), not a transformer motor head.
"""
from __future__ import annotations

import numpy as np

from .channels import N_CHANNELS, N_ODOR, N_VISION
from .pen import PenClass
from .vision import VisionMLP

D = 16


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(np.clip(z, -20, 20))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-8)


class TransformerFly(PenClass):
    """One-layer attention fly. Same encode / see / forward / code as HaikuBrain."""

    n_channels = N_CHANNELS

    def __init__(self, seed: int = 11, vision: bool = True) -> None:
        rng = np.random.default_rng(seed)
        self.use_vision = bool(vision)
        from .channels import layout

        self.n_odor, self.n_vis, n = layout(self.use_vision)
        self.n_channels = n
        self.init_pen(n, seed=seed)
        self.vision = VisionMLP(N_VISION, seed=seed + 1) if self.use_vision else None
        self.channel = 0
        self._vis = np.zeros(self.n_vis, dtype=np.float32) if self.n_vis else np.zeros(0, dtype=np.float32)
        self.pam = 0.0
        self.ppl1 = 0.0
        self.valence = 0.0
        self.using_connectome = False
        self.lif_backend = "transformer"
        self.spikes_last = 0
        self.n_neurons = 3 * D
        self.n_edges = 0
        self.kc_mbon_updates = 0
        self.kc_mbon_dw = 0.0
        self.Wq = rng.normal(0, 0.2, (D, D)).astype(np.float32)
        self.Wk = rng.normal(0, 0.2, (D, D)).astype(np.float32)
        self.Wv = rng.normal(0, 0.2, (D, D)).astype(np.float32)
        self.W_code = rng.normal(0, 0.2, (D, n)).astype(np.float32)
        self.E_odor = rng.normal(0, 0.2, (self.n_odor, D)).astype(np.float32)
        self.E_vis = rng.normal(0, 0.2, (max(self.n_vis, 1), D)).astype(np.float32)
        self.E_da = rng.normal(0, 0.2, (2, D)).astype(np.float32)
        self._code = np.zeros(n, dtype=np.float32)
        self._pooled = np.zeros(D, dtype=np.float32)
        self.kc = self._code

    def encode(self, mora_index: int) -> None:
        self.channel = int(mora_index) % max(1, self.n_odor)

    def encode_mora(self, mora_index: int) -> None:
        self.encode(mora_index)

    def see(self, frame: np.ndarray) -> None:
        if not self.use_vision or self.vision is None:
            return
        self._vis = self.vision(frame)

    @property
    def code(self) -> np.ndarray:
        return self._code

    def forward(
        self,
        dt: float = 0.016,
        odor: float = 1.0,
        sugar: float = 0.0,
        shock: float = 0.0,
    ) -> None:
        self.pam = min(1.4, 0.85 * self.pam + 0.55 * sugar)
        self.ppl1 = min(1.4, 0.85 * self.ppl1 + 0.55 * shock)
        tok_o = self.E_odor[self.channel] * float(odor)
        tok_d = self.pam * self.E_da[0] - self.ppl1 * self.E_da[1]
        if self.use_vision and self._vis.size:
            tok_v = self._vis @ self.E_vis[: self.n_vis]
            x = np.stack([tok_o, tok_v, tok_d], axis=0)
        else:
            x = np.stack([tok_o, tok_d], axis=0)
        q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
        attn = _softmax((q @ k.T) / np.sqrt(D))
        h = attn @ v
        pooled = h.mean(axis=0)
        self._pooled = pooled.astype(np.float32)
        self._code = np.tanh(pooled @ self.W_code).astype(np.float32)
        self.kc = self._code
        self.valence = float(np.clip(0.25 * self.pam - 0.35 * self.ppl1, -1, 1))
        self._dn_bias = 0.0

    def learn(self, action: np.ndarray, reward: float) -> None:
        r = float(reward)
        self.pam = min(1.4, self.pam + 0.55 * max(0.0, r))
        self.ppl1 = min(1.4, self.ppl1 + 0.55 * max(0.0, -r))
        self.learn_pen(action, r)
        if self.vision is not None:
            self.vision.reinforce(r)
        if abs(r) < 1e-8:
            return
        eta = np.float32(0.05 * r)
        self.W_code += eta * np.outer(self._pooled, self._code)
        self.W_code = np.clip(self.W_code, -2.0, 2.0)
        # three-factor nudge on odor embedding for the active bin
        self.E_odor[self.channel] += eta * 0.25 * self._pooled
        if self.use_vision and self._vis.size:
            self.E_vis[: self.n_vis] += eta * 0.15 * np.outer(self._vis, self._pooled)
