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
| Input | terminal / task cue | odor / light / walls | mora as odor on one KC channel |
| Brain | MaleCNS LIF | walk circuit or tracker | 8-channel MB RL; MaleCNS optional (packed in this repo) |
| Output | legs → keys | XY trail → Latin letter | XY + pen → hiragana |
| Learning | usually none / scripted | usually none | PAM if ink matches glyph, PPL1 if miss |

### Haiku RL loop

This repo points a Hige-style mushroom-body law at Japanese 5–7–5 (fly-font kana, not Latin):

```text
mora as odor → KC sparse code → motor (dx, dy, pen)
reward (ink vs glyph) → PAM if match / PPL1 if miss → KC→MBON + W_motor
```

- Default: fast 8-channel MB loop in `haiku/brain.py`.
- Opt-in: `python tools/pack_malecns.py` then `--connectome`; PAM/PPL1/KC injections; Hige-style **PAM LTD / PPL1 LTP** on KC→MBON.
- Motor: KC rates × a learned 8×3 map; three-factor `W += reward * outer(KC, action)`.
- Corpus: classic 5–7–5 in hiragana (Bashō, Buson, Issa). Glyphs are fly-font polylines rasterized to a small ink grid.
- LIF, packer, and CSR loader all live in this tree (`haiku/lif.py`, `tools/pack_malecns.py`, `haiku/connectome.py`).

This is **not** a reconstructed fly mind and **not** a claim that fruit flies compose Bashō. It is dopamine-gated plasticity on a real or reduced mushroom body, trained on stroke overlap.

### Related Grok-Bot trading clips (same week, not fly)

X also filled with Grok Bot “trading desk” videos (HyperGrok / Bull Desk / Polymarket). Those are **multi-agent prompt packs + broker APIs**, not MaleCNS. They share the viral “agent floor” aesthetic with flycoding but zero connectome. Stock path: Webull OpenAPI, human-gated tickets (`approve BD-…`). Crypto path: Hyperliquid. Polymarket clips (cross-venue odds, both-legs &lt; 95¢) are prediction markets, not US equities. Treat P&amp;L screenshots as unverified unless the author labels paper/scripted (several did).
