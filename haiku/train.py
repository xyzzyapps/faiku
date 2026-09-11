"""RL loop: mora odor → brain → pen → PAM/PPL1."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .brain import HaikuBrain, N_KC
from .env import HaikuEnv


def _save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(rgb, "RGB").resize((192, 192), Image.NEAREST).save(path)
        return
    except Exception:
        pass
    # PPM fallback
    h, w, _ = rgb.shape
    ppm = path.with_suffix(".ppm")
    header = f"P6\n{w} {h}\n255\n".encode()
    ppm.write_bytes(header + rgb.tobytes())


def train(
    poem: str = "basho",
    steps: int = 2500,
    connectome: bool = False,
    backend: str = "cpu",
    ui: bool = True,
    out: Path | None = None,
) -> dict:
    env = HaikuEnv(poem)
    env.reset_glyph()
    brain = HaikuBrain(connectome=connectome, backend=backend)
    brain.encode_mora(0)
    out = out or Path("runs")
    scores: list[float] = []
    window: list[float] = []
    tk_root = None
    canvas = None
    photo = None
    if ui:
        try:
            import tkinter as tk
            from tkinter import PhotoImage

            tk_root = tk.Tk()
            tk_root.title(f"haiku · {env.poem.lines}")
            canvas = tk.Canvas(tk_root, width=400, height=440, bg="#111")
            canvas.pack()
            photo = PhotoImage(width=192, height=192)
            canvas.create_image(104, 16, image=photo, anchor="nw")
            label = canvas.create_text(200, 400, fill="#ddd", font=("Segoe UI", 11), text="")
        except Exception as exc:
            print(f"haiku: no UI ({exc})", flush=True)
            tk_root = None

    def paint() -> None:
        if tk_root is None:
            return
        rgb = env.snapshot_pair()
        scale = 4
        rows = []
        for y in range(rgb.shape[0]):
            row = []
            for x in range(rgb.shape[1]):
                r, g, b = (int(v) for v in rgb[y, x])
                cell = f"#{r:02x}{g:02x}{b:02x} "
                row.append(cell * scale)
            line = "{" + "".join(row).strip() + "}"
            rows.extend([line] * scale)
        photo.put(" ".join(rows))
        txt = (
            f"{env.mora}  mora {env.i + 1}/{len(env.morae)}  "
            f"PAM {brain.pam:.2f} PPL1 {brain.ppl1:.2f}  "
            f"{'MaleCNS ' + brain.lif_backend if brain.using_connectome else 'MB loop'}"
        )
        canvas.itemconfigure(label, text=txt)
        tk_root.update()

    for t in range(steps):
        brain.encode_mora(env.i % N_KC)
        action = brain.step(0.016, odor=1.0, sugar=0.0, shock=0.0)
        reward, done = env.step(action)
        if done:
            brain.reinforce(action, reward)
            window.append(reward)
            if len(window) > 40:
                window.pop(0)
            scores.append(float(np.mean(window)))
            if t % 20 == 0 or t == steps - 1:
                _save_png(out / f"ink_{t:05d}_{env.mora}.png", env.snapshot_pair())
                mean = scores[-1] if scores else 0.0
                print(
                    f"t={t:5d}  {env.mora}  R={reward:+.3f}  mean40={mean:+.3f}  "
                    f"spikes={brain.spikes_last}",
                    flush=True,
                )
            env.advance_mora()
        if ui and tk_root is not None and t % 3 == 0:
            paint()
    if tk_root is not None:
        try:
            tk_root.destroy()
        except Exception:
            pass
    return {
        "poem": env.poem.lines,
        "steps": steps,
        "final_mean": scores[-1] if scores else 0.0,
        "connectome": brain.using_connectome,
        "backend": brain.lif_backend,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="MaleCNS / MB RL that writes Japanese haiku")
    p.add_argument("--poem", default="basho")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--connectome", action="store_true")
    p.add_argument("--connectome-cpu", action="store_true")
    p.add_argument("--no-ui", action="store_true")
    args = p.parse_args(argv)
    cns = args.connectome or args.connectome_cpu
    backend = "cpu" if args.connectome_cpu or not args.connectome else "vulkan"
    if args.connectome_cpu:
        backend = "cpu"
    elif args.connectome:
        backend = "vulkan"
    info = train(
        poem=args.poem,
        steps=args.steps,
        connectome=cns,
        backend=backend,
        ui=not args.no_ui,
    )
    print(info, flush=True)
