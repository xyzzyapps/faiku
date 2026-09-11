"""Inference as a higher-order loop.

The fly is an abstract `Fly` (`encode` / `forward` / `code`). Scripts and
decode hooks never import MaleCNS. Swap the generator (5-7-5 words, kana,
your own format) or swap the fly class without touching the other.

    from haiku.infer import Hooks, infer, haiku_word_script, word_hooks

    text = infer(fly, haiku_word_script(seed=0), word_hooks())

Hook points (all optional except decode):

- script(fly) -> iterable of cues
- encode(fly, cue, i)     default: fly.encode(mora)
- forward(fly, cue)       default: fly.forward(...) then fly.code
- decode(fly, action, cue, acc)
- after(fly, action, cue, acc)
- finish(acc) -> result
- halt(acc) -> bool

A cue is a plain dict. Built-in scripts only set `mora` and `line`.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .brain import HaikuBrain
from .fly import Fly
from .generate import WORDS, _pick, _syllables, load_weights

Cue = dict[str, Any]
Acc = dict[str, Any]


@dataclass
class Hooks:
    encode: Callable[[Fly, Cue, int], None] | None = None
    forward: Callable[[Fly, Cue], np.ndarray] | None = None
    decode: Callable[[Fly, np.ndarray, Cue, Acc], Acc] | None = None
    after: Callable[[Fly, np.ndarray, Cue, Acc], None] | None = None
    halt: Callable[[Acc], bool] | None = None
    finish: Callable[[Acc], Any] = field(default=lambda acc: acc)
    acc0: Acc = field(default_factory=dict)


def infer(
    fly: Fly,
    script: Iterable[Cue] | Callable[[Fly], Iterable[Cue]],
    hooks: Hooks | None = None,
) -> Any:
    """Run one script through a Fly. `script` is a cue iterable or a factory."""
    hooks = hooks or Hooks()
    acc: Acc = dict(hooks.acc0)
    cues = script(fly) if callable(script) else script
    for i, cue in enumerate(cues):
        if hooks.encode:
            hooks.encode(fly, cue, i)
        else:
            _default_encode(fly, cue, i)
        if hooks.forward:
            action = hooks.forward(fly, cue)
        else:
            action = _default_forward(fly, cue)
        if hooks.decode:
            acc = hooks.decode(fly, action, cue, acc)
        if hooks.after:
            hooks.after(fly, action, cue, acc)
        if hooks.halt and hooks.halt(acc):
            break
    return hooks.finish(acc)


def _default_encode(fly: Fly, cue: Cue, i: int) -> None:
    mora = cue.get("mora", i)
    fly.encode(int(mora) % int(fly.n_channels))


def _default_forward(fly: Fly, cue: Cue) -> np.ndarray:
    odor = float(cue.get("odor", 1.0))
    dt = float(cue.get("dt", 0.016))
    sugar = float(cue.get("sugar", 0.0))
    shock = float(cue.get("shock", 0.0))
    fly.forward(dt=dt, odor=odor, sugar=sugar, shock=shock)
    return np.asarray(fly.code, dtype=np.float32)


# --- built-in 5-7-5 word script ------------------------------------------------

def haiku_word_script(seed: int = 0, max_tries: int = 24) -> Callable[[Fly], Iterator[Cue]]:
    """Yields cues until three lines reach 5, 7, 5 syllables (word-bank decode)."""

    banks = (
        (5, (0, 3, 1, 6, 7)),
        (7, (2, 0, 3, 1, 4, 6, 7)),
        (5, (5, 2, 3, 6, 0)),
    )

    def gen(fly: Fly) -> Iterator[Cue]:
        rng = np.random.default_rng(seed)
        for line_i, (need, bank) in enumerate(banks):
            for k in range(max_tries):
                yield {
                    "mora": k,
                    "line": line_i,
                    "need": need,
                    "bank": bank[k % len(bank)],
                    "rng": rng,
                    "odor": 1.0,
                }

    return gen


def decode_word(fly: Fly, action: np.ndarray, cue: Cue, acc: Acc) -> Acc:
    lines: list[list[str]] = acc.setdefault("lines", [[], [], []])
    counts: list[int] = acc.setdefault("counts", [0, 0, 0])
    line = int(cue["line"])
    need = int(cue["need"])
    if counts[line] >= need:
        return acc
    remain = need - counts[line]
    rng: np.random.Generator = cue["rng"]
    kc = np.asarray(fly.code)
    if remain == 1:
        w = WORDS[3][int(np.argmax(kc))]
    else:
        w = _pick(fly, rng, int(cue["bank"]))
    s = _syllables(w)
    if s > remain:
        w = WORDS[3][int(np.argmax(kc))]
        s = _syllables(w)
    if s > remain:
        counts[line] = need
        return acc
    lines[line].append(w)
    counts[line] += s
    return acc


def halt_575(acc: Acc) -> bool:
    counts = acc.get("counts") or [0, 0, 0]
    return len(counts) >= 3 and counts[0] >= 5 and counts[1] >= 7 and counts[2] >= 5


def finish_575(acc: Acc) -> str:
    lines = acc.get("lines") or [[], [], []]
    return " / ".join(" ".join(part) for part in lines)


def word_hooks() -> Hooks:
    return Hooks(
        encode=_default_encode,
        forward=_default_forward,
        decode=decode_word,
        halt=halt_575,
        finish=finish_575,
        acc0={"lines": [[], [], []], "counts": [0, 0, 0]},
    )


def infer_haiku(fly: Fly, seed: int = 0) -> str:
    return infer(fly, haiku_word_script(seed=seed), word_hooks())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run inference hooks over a script generator")
    p.add_argument("--weights", type=Path, default=Path("runs/weights.npz"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mb-only", action="store_true")
    p.add_argument("--connectome-cpu", action="store_true")
    args = p.parse_args(argv)
    backend = "cpu" if args.connectome_cpu or args.mb_only else "vulkan"
    brain = HaikuBrain(connectome=not args.mb_only, backend=backend)
    if args.weights.exists():
        load_weights(brain, args.weights)
        print(f"haiku: loaded {args.weights}", flush=True)
    else:
        print(f"haiku: no weights at {args.weights}; using init W", flush=True)
    print(infer_haiku(brain, seed=args.seed), flush=True)


if __name__ == "__main__":
    main()
