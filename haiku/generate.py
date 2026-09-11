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


def _pick(fly, rng: np.random.Generator, bank: int) -> str:
    raw = fly.code if hasattr(fly, "code") else fly.kc
    kc = np.clip(np.asarray(raw), 0, None)
    if float(kc.sum()) < 1e-6:
        kc = np.ones(N_KC, dtype=np.float32)
    p = kc / float(kc.sum())
    p = np.clip(p, 1e-6, 1.0)
    p = p / p.sum()
    i = int(rng.choice(N_KC, p=p))
    j = (i + bank) % N_KC
    return WORDS[bank][j]


def compose(brain: HaikuBrain, seed: int = 0) -> tuple[str, str, str]:
    from .infer import infer_haiku

    text = infer_haiku(brain, seed=seed)
    parts = [p.strip() for p in text.split(" / ")]
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1], parts[2]


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
