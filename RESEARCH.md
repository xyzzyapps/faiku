# Haiku research notes

This file is the flycoding / fly-font / mushroom-body write-up for this **independent** repo. It does not import or require any desktop-pet / Holmsy checkout.

---

## Flycoding, fly fonts, and haiku (September 2026)

Viral clips from the same week as MaleCNS are the **same three-layer sandwich** as Doom/Mario/Holmsy: engineered sensors → real (or reduced) connectome dynamics → engineered actuators. They are not a fly that knows Postgres or typography.

### Flycoding (Supabase)

[Supabase, 10 Sep 2026](https://x.com/supabase/status/2097999894632149229) (“the fly is deploying databases”, ~24 s). Public description (including Grok replies in that thread): a simulated *Drosophila* driven by **MaleCNS v1.0** and a **NeuroMechFly-style** body. Neural activity is mapped to **leg movements that type CLI** on a virtual keyboard (`supabase link`, migrations / `db push`).

How it actually works:

1. **Sensors (author-built).** The terminal, a scripted “deploy this project” cue, or a visual of the CLI is turned into currents on **labeled cell classes** (photoreceptors, JO, GRNs, etc.). The fly does not parse SQL.
2. **Connectome (measured).** ~166k LIF cells, Shiu constants, Berg graph. Spikes propagate through real synapses. This is the scientific middle.
3. **Motor → keys (author-built).** Descending / leg motor-neuron rates drive a simulated body. Tarsi collide with a **virtual keyboard**. Collisions become keystrokes.

The command sequence lives in the **decoder and the stage**, not in Kenyon cells. Same pattern as Minecraft-fly, SM64-fly, Holmsy: world → labeled sensory neurons → LIF → labeled motor neurons → actuators. Coding is “the actuator is a keyboard.”

No public repo from Supabase for this clip was found at write-up; treat it as a DevRel demo, not a released training loop.

### Fly fonts (alphabets from a fly)

Posts of Latin alphabets “designed” by a fly are **ink from a trajectory**, not a type foundry in the mushroom body.

Typical pipeline:

1. Drive walking (real video, or simulated MN/DN readout).
2. Record **centroid or tarsus XY** each frame — a polyline.
3. Treat that polyline as a glyph: overlay on `A`…`Z`, or shape the arena/odor so the path looks like a letter.
4. Export traces as a font (each letter = one walk).

Biology in the loop is optional. Classic centroid-tracking papers (e.g. Pop et al. 2020, *eLife*, “undead walking”: blob → XY → plot) already draw pretty traces with no connectome. Connectome versions swap the tracker for motor-neuron decoding, then draw the same XY.

What is **not** happening: Kenyon cells storing Unicode. A letter is a **2-D path**. Matching a template is a graphics loss (overlap / Hausdorff), not language.

### Shared sandwich

| Layer | Flycoding | Fly font | Haiku (this repo) |
|---|---|---|---|
| Neural input | engineered currents on labeled cells | odor / light / walls (or none) | mora → KC slice; sugar → PAM/LB3c; shock → PPL1 |
| Neural output | who spiked (then a body decoder) | walk XY or MN/DN rates | who spiked, collapsed to 8 KC rates (`Fly.code`) |
| Extra decoder | legs → virtual keyboard | polyline → letter | `PenClass.stroke()` → paper; word hook → 5–7–5 |
| Learning | usually none / scripted | usually none | PAM LTD / PPL1 LTP on KC→MBON; `W` on the pen only |

### Classes: `Fly` vs `PenClass`

```text
Fly            encode / forward / code     neurons only — no writing tip
  PenClass     stroke() → Stroke           invented decoder onto paper
    HaikuBrain
```

`infer()` talks to **`Fly`**. The glyph canvas talks to **`PenClass.stroke()`**. MaleCNS never sees a pen.

### Neural network input (actual)

Default run is packed **MaleCNS v1.0**: 166,700 Shiu LIF cells, ~25.6M directed edges (`haiku/lif.py`, `connectome.py`). `--mb-only` skips the graph.

Each `Fly.forward(dt, odor, sugar, shock)` tick, after `encode(mora_index)`:

| Knob | Into MaleCNS | Into `--mb-only` |
|---|---|---|
| `mora_index` | which eighth of **KC** gets current | which of 8 floats is the odor bin |
| `odor` | `inject(KC[lo:hi], 7.5 * (0.2 + 0.8 * odor))` | add to `kc[channel]` |
| `sugar` | **PAM** (11 nA-scale) and sugar GRN **LB3c** (10) | PAM trace |
| `shock` | **PPL1** (11.3) | PPL1 trace |
| `dt` | 4–8 × 1 ms LIF steps | exponential decay on the 8 floats |

The graph does **not** take Unicode, 5–7–5, or pixels. Mora is just “which KC bin.”

### Neural network output (actual)

MaleCNS output is a **spike index list** (who fired). In the animal you would read identified **DNs / MNs** (walk, turn, jump, proboscis). This repo does **not** expose a full motor-neuron head.

What `Fly` exposes:

- **`code`**: 8 numbers. Each is the mean spike rate of one KC eighth (smoothed). `--mb-only`: the 8 floats themselves.
- Spikes also drive Hige-like **KC→MBON** weight changes when PAM/PPL1 are up.
- Mean **DN** spike rate is stored as `_dn_bias` for the pen only (mixed into `dx`). It is not a walk controller.

`Fly.forward()` returns **nothing**. No `(dx, dy, pen)` on this class.

### Pen (`PenClass`, not the network)

`Stroke(dx, dy, down)` is a **canvas command we invented**. The connectome has no stylus.

`HaikuBrain.stroke()`:

```text
v = tanh(code @ W)     # W is 8×3, learned, not in MaleCNS
v.dx += 0.15 * DN_bias
Stroke(dx, dy, down = v[2] > 0.05)
```

The writing env steps the tip by `dx, dy` and lays ink if `down`. Reward is **glyph overlap**, not “is this a haiku.” That scalar becomes `sugar` / `shock` on the **next** neural tick. The stroke does not feed back as JO or tarsal touch.

`W` is updated with `W += reward * outer(code, tanh(stroke))`. That is pen learning. Graph learning is PAM LTD / PPL1 LTP on KC→MBON synapses only.

Word-bank 5–7–5 (`infer` decode hook) reads **`Fly.code`**, not `Stroke`. The fly does not compose Bashō.

This is **not** a reconstructed fly mind. It is Shiu LIF on measured wiring (or eight floats), plus an engineered pen.

### Related Grok-Bot trading clips (same week, not fly)

X also filled with Grok Bot “trading desk” videos (HyperGrok / Bull Desk / Polymarket). Those are **multi-agent prompt packs + broker APIs**, not MaleCNS. They share the viral “agent floor” aesthetic with flycoding but zero connectome. Stock path: Webull OpenAPI, human-gated tickets (`approve BD-…`). Crypto path: Hyperliquid. Polymarket clips (cross-venue odds, both-legs &lt; 95¢) are prediction markets, not US equities. Treat P&amp;L screenshots as unverified unless the author labels paper/scripted (several did).
