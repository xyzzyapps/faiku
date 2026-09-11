"""Read a 5-7-5 out of the trained KC code and motor map."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .brain import N_KC, HaikuBrain

# One- and two-syllable pool, 8 bins = 8 Kenyon channels.
WORDS = [
    ("dusk", "glass", "fly", "ink", "pond", "rain", "wing", "old"),
    ("quiet", "cursor", "sugar", "letter", "ripple", "window", "wiring", "autumn"),
    ("traces", "takeoff", "waiting", "humming", "stillness", "looms", "jumps", "dries"),
    ("on", "the", "a", "and", "in", "of", "to", "for"),
    ("five", "seven", "mora", "spike", "weight", "synapse", "kenyon", "dopamine"),
    ("old", "male", "fruit", "nerve", "cord", "brain", "cell", "code"),
    ("sound", "water", "rock", "leaf", "frost", "wind", "light", "shade"),
    ("pane", "trail", "glyph", "stroke", "odor", "taste", "shock", "reward"),
]


def _syllables(word: str) -> int:
    w = word.lower()
    n = 0
    prev = False
    for ch in w:
        v = ch in "aeiouy"
        if v and not prev:
            n += 1
        prev = v
    return max(1, n)


def _pick(brain: HaikuBrain, rng: np.random.Generator, bank: int) -> str:
    kc = np.clip(brain.kc, 0, None)
    if float(kc.sum()) < 1e-6:
        kc = np.ones(N_KC, dtype=np.float32)
    p = kc / float(kc.sum())
    p = np.clip(p, 1e-6, 1.0)
    p = p / p.sum()
    i = int(rng.choice(N_KC, p=p))
    j = (i + bank) % N_KC
    return WORDS[bank][j]


def compose(brain: HaikuBrain, seed: int = 0) -> tuple[str, str, str]:
    rng = np.random.default_rng(seed)
    lines: list[str] = []
    for need, banks in ((5, (0, 3, 1, 6, 7)), (7, (2, 0, 3, 1, 4, 6, 7)), (5, (5, 2, 3, 6, 0))):
        words: list[str] = []
        n = 0
        k = 0
        while n < need:
            brain.encode_mora(k % N_KC)
            act = brain.step(0.016, odor=1.0, sugar=0.0, shock=0.0)
            brain.learn(act, float(brain.valence) * 0.05)
            remain = need - n
            if remain == 1:
                w = WORDS[3][int(np.argmax(brain.kc))]
            else:
                w = _pick(brain, rng, banks[k % len(banks)])
            s = _syllables(w)
            if s > remain:
                w = WORDS[3][int(np.argmax(brain.kc))]
                s = _syllables(w)
            if s > remain:
                break
            words.append(w)
            n += s
            k += 1
            if k > 24:
                break
        lines.append(" ".join(words))
    return lines[0], lines[1], lines[2]


def save_weights(brain: HaikuBrain, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "W": brain.W,
        "mbon_avoid": brain.mbon_avoid,
        "kc": brain.kc,
        "pam": np.float32(brain.pam),
        "ppl1": np.float32(brain.ppl1),
        "kc_mbon_dw": np.float32(brain.kc_mbon_dw),
        "kc_mbon_updates": np.int64(brain.kc_mbon_updates),
        "connectome": np.int8(1 if brain.using_connectome else 0),
        "backend": np.array(brain.lif_backend),
    }
    if brain.using_connectome and brain._graph is not None and brain._net is not None:
        ei = brain._graph.kc_mbon_edge.astype(np.int64)
        payload["kc_mbon_edge"] = ei.astype(np.uint32)
        payload["kc_mbon_w"] = brain._net.W.data[ei].astype(np.float32)
    np.savez_compressed(path, **payload)


def load_weights(brain: HaikuBrain, path: Path) -> None:
    z = np.load(path, allow_pickle=False)
    brain.W[:] = z["W"]
    brain.mbon_avoid[:] = z["mbon_avoid"]
    if "kc" in z.files:
        brain.kc[:] = z["kc"]
    brain.pam = float(z["pam"])
    brain.ppl1 = float(z["ppl1"])
    if (
        brain.using_connectome
        and brain._net is not None
        and "kc_mbon_w" in z.files
        and "kc_mbon_edge" in z.files
    ):
        ei = z["kc_mbon_edge"].astype(np.int64)
        brain._net.W.data[ei] = z["kc_mbon_w"]
        if hasattr(brain._net, "update_weights"):
            brain._net.update_weights(ei, z["kc_mbon_w"])
