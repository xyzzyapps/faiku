"""Custom inference script: define a generator + decode hook, then call infer().

    python examples/custom_script.py --mb-only
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from haiku.brain import HaikuBrain
from haiku.fly import Fly
from haiku.generate import load_weights
from haiku.infer import Hooks, infer, _default_encode, _default_forward


def kana_row_script(kana: str = "ふるいけや"):
    def gen(fly: Fly):
        n = max(1, int(fly.n_channels))
        for i, ch in enumerate(kana):
            yield {"mora": i % n, "char": ch, "odor": 1.0}

    return gen


def decode_chars(brain, action, cue, acc):
    acc.setdefault("chars", []).append(cue["char"])
    acc.setdefault("pen", []).append(float(action[2]))
    return acc


if __name__ == "__main__":
    brain = HaikuBrain(connectome=False, backend="cpu")
    w = Path("runs/weights.npz")
    if w.exists():
        load_weights(brain, w)
    text = infer(
        brain,
        kana_row_script(),
        Hooks(
            encode=_default_encode,
            forward=_default_forward,
            decode=decode_chars,
            finish=lambda acc: "".join(acc.get("chars") or []),
        ),
    )
    print(text)
