"""Build a Fly with the same I/O: encode, see, forward, code, stroke."""
from __future__ import annotations

from .brain import HaikuBrain
from .fly import Fly
from .pen import PenClass
from .transformer_fly import TransformerFly


def make_fly(arch: str = "malecns", backend: str = "vulkan") -> PenClass:
    arch = (arch or "malecns").lower().replace("_", "-")
    if arch in ("odor", "mb-only", "mbonly", "mb-odor", "odor-only"):
        print("haiku: odor-only 8-channel MB (previous fly, no vision)", flush=True)
        return HaikuBrain(connectome=False, backend="cpu", vision=False)
    if arch in ("transformer-odor", "tf-odor"):
        print("haiku: TransformerFly odor-only (no vision token)", flush=True)
        return TransformerFly(vision=False)
    if arch in ("transformer", "tf", "attn"):
        print("haiku: TransformerFly (4 odor + 4 vision)", flush=True)
        return TransformerFly(vision=True)
    if arch in ("mb", "mb-vision"):
        print("haiku: 8-channel MB with vision MLP", flush=True)
        return HaikuBrain(connectome=False, backend="cpu", vision=True)
    if arch in ("malecns-odor", "cns-odor"):
        fly = HaikuBrain(connectome=True, backend=backend, vision=False)
        if not fly.using_connectome:
            raise SystemExit("haiku: MaleCNS pack missing — run python tools/pack_malecns.py")
        return fly
    fly = HaikuBrain(connectome=True, backend=backend, vision=True)
    if not fly.using_connectome:
        raise SystemExit("haiku: MaleCNS pack missing — run python tools/pack_malecns.py")
    return fly


def as_fly(obj: Fly) -> Fly:
    return obj
