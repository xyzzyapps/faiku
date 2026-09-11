"""RL loop: mora odor → brain → pen → PAM/PPL1."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .env import HaikuEnv
from .factory import make_fly
from .generate import compose, save_weights


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
    arch: str = "malecns",
) -> dict:
    env = HaikuEnv(poem)
    env.reset_glyph()
    if arch:
        brain = make_fly(arch, backend=backend)
    else:
        from .brain import HaikuBrain

        brain = HaikuBrain(connectome=connectome, backend=backend)
        if connectome and not brain.using_connectome:
            raise SystemExit("haiku: MaleCNS pack missing — run python tools/pack_malecns.py")
    brain.encode(0)
    if getattr(brain, "use_vision", False):
        brain.see(env.target)
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
        n_odor = int(getattr(brain, "n_odor", 8))
        brain.encode(env.i % max(1, n_odor))
        if getattr(brain, "use_vision", False):
            brain.see(env.target)
        brain.forward(0.016, odor=1.0, sugar=0.0, shock=0.0)
        stroke = brain.stroke()
        reward, done = env.step(stroke)
        brain.learn(stroke.vector(), reward)
        if done:
            window.append(reward)
            if len(window) > 40:
                window.pop(0)
            scores.append(float(np.mean(window)))
            if t % 20 == 0 or t == steps - 1:
                _save_png(out / f"ink_{t:05d}_{env.mora}.png", env.snapshot_pair())
                mean = scores[-1] if scores else 0.0
                print(
                    f"t={t:5d}  {env.mora}  R={reward:+.3f}  mean40={mean:+.3f}  "
                    f"spikes={brain.spikes_last}  kc_mbon_dw={brain.kc_mbon_dw:.4f}  "
                    f"kc_mbon_n={brain.kc_mbon_updates}",
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
    weights = out / "weights.npz"
    save_weights(brain, weights)
    generated = compose(brain, seed=steps)
    gen_path = out / "generated.txt"
    gen_path.write_text("\n".join(generated) + "\n", encoding="utf-8")
    print("generated haiku:", flush=True)
    print("\n".join(generated), flush=True)
    return {
        "poem": env.poem.lines,
        "steps": steps,
        "final_mean": scores[-1] if scores else 0.0,
        "connectome": brain.using_connectome,
        "backend": brain.lif_backend,
        "kc_mbon_updates": brain.kc_mbon_updates,
        "kc_mbon_dw": brain.kc_mbon_dw,
        "generated": " / ".join(generated),
        "weights": str(weights),
        "arch": type(brain).__name__,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="MaleCNS / MB RL that writes Japanese haiku")
    p.add_argument("--poem", default="basho")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--connectome", action="store_true", help="MaleCNS (default)")
    p.add_argument("--connectome-cpu", action="store_true")
    p.add_argument(
        "--mb-only",
        action="store_true",
        help="previous odor-only 8-channel fly (no MaleCNS, no vision)",
    )
    p.add_argument(
        "--arch",
        default="",
        help="malecns | malecns-odor | mb | odor | transformer | transformer-odor",
    )
    p.add_argument("--no-ui", action="store_true")
    args = p.parse_args(argv)
    arch = (args.arch or "").strip().lower()
    if not arch:
        arch = "odor" if args.mb_only else "malecns"
    if args.connectome_cpu:
        backend = "cpu"
    else:
        backend = "vulkan"
    info = train(
        poem=args.poem,
        steps=args.steps,
        connectome=arch == "malecns",
        backend=backend,
        ui=not args.no_ui,
        arch=arch,
    )
    print(info, flush=True)
