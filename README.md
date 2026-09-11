# Haiku

A *Drosophila* mushroom-body RL loop that learns to write **Japanese haiku** as a fly font.

Supabase’s flycoding demo maps MaleCNS spikes onto legs that type. Other people drew Latin alphabets as fly walk-traces. This repo does the same job for **kana** (and English letters): Kenyon-cell odor channels encode each mora, PAM / PPL1 write the lesson onto KC→MBON (and a thin motor map), and the fly’s “pen” is a 2-D ink trail.

This tree is **standalone**. Default is packed **MaleCNS v1.0** (166,700 LIF cells): each tick’s ink reward is dopamine, and that dopamine changes **KC→MBON** weights on the real graph. `--mb-only` is the 8-channel fallback. Pack once before the default run.

Target example (`--poem glass`):

```text
dusk on the glass
a fly traces each letter
ink of the old pond
```

## Generated from trained MaleCNS weights

After a Vulkan run (166,700 cells, ~25.6M edges, KC→MBON `dw ≈ 1357`, ~1.7M edge updates) the Kenyon-cell code was read out into words. This is that 5–7–5, not a hand-written poem:

```text
rain on window
jumps rain and quiet spike
nerve humming in
```

`python -m haiku --no-ui --poem glass` trains, writes `runs/weights.npz`, then prints a new haiku from those weights (`haiku/generate.py`).

```text
idea  →  mora as odor  →  KC sparse code  →  motor (dx, dy, pen)
reward (ink vs glyph)  →  PAM if match / PPL1 if miss  →  KC→MBON + W_motor
```

This is **not** a reconstructed fly mind and **not** a claim that fruit flies compose Bashō. It is Hige-style dopamine-gated plasticity on a real (or reduced) mushroom body, trained on 5–7–5.

## The 8-channel fallback (`--mb-only`)

Before MaleCNS was the default, every run used a **reduced mushroom-body loop**: eight Kenyon-cell numbers, not 166k LIF neurons. `--mb-only` is that loop, unchanged.

In a real fly, KCs are a **sparse odor code** (many cells, few active). Holmsy-style pets collapse that to **8 bins**. Haiku uses the same collapse: mora index `i` maps to channel `i % 8`. Channel `c` is a single float `kc[c] ∈ [0, 1]`. There is no spike, no delay, no synapse table.

What exists in `--mb-only`:

| Piece | Size | Role |
|---|---|---|
| `kc` | 8 floats | “Which mora / odor is on” |
| `mbon_avoid` | 8 floats | KC→MBON avoidance weight per channel (start at 1) |
| `W` | 8×3 | Motor map: `(dx, dy, pen)` |
| `pam`, `ppl1` | 2 floats | Appetitive / aversive dopamine traces |

Tick:

1. Decay `kc`, add odor into the current channel.
2. Decay PAM/PPL1; add sugar / shock.
3. If that channel is on and PAM is up, **shrink** `mbon_avoid[c]` (Hige LTD: cue now means approach).
4. If PPL1 is up, **grow** `mbon_avoid[c]` (aversive).
5. `action = tanh(kc @ W)` — the only motor.

Learning: `W += 0.18 * reward * outer(kc, tanh(action))`. That is three-factor plasticity (eligibility × action × dopamine) on **24 numbers**. Nothing in MaleCNS is touched because MaleCNS is not loaded.

Use `--mb-only` when you have no pack, when you want a 16 ms tick on CPU, or when you are debugging the writing env. It is a **control law in the shape of a mushroom body**, not a connectome.

Default `python -m haiku` is different: Shiu LIF on the packed graph, PAM/PPL1 injected into real PAM/PPL1 cells, odor into a slice of real KCs, and the same reward writes **KC→MBON synapses on the CSR** (plus the 8×3 map, which still sits on top as the pen). `--mb-only` skips all of that.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

MaleCNS (required for the default run):

```powershell
pip install pyarrow pandas
python tools/pack_malecns.py
```

Pack files land in `haiku/data/malecns_v1/` (gitignored). Raw feathers stay in `.cache/`.

## Run

```powershell
python -m haiku                  # MaleCNS LIF; PAM/PPL1 write KC→MBON every tick
python -m haiku --connectome-cpu
python -m haiku --mb-only        # 8-channel loop only (no 166k graph)
python -m haiku --steps 4000 --no-ui
python -m haiku --poem glass
```

Ink snapshots and `runs/weights.npz` land in `runs/`.

## Poems

Classic 5–7–5 in hiragana (see `haiku/corpus.py`):

Japanese (`--poem basho` and friends):

- ふるいけや / かわずとびこむ / みずのおと
- しずかさや / いわにしみいる / せみのこえ
- なのはなや / つきはひがしに / ひはにしに

English (`--poem glass`, `sugar`, `loom`, `wiring`, `pond`):

- dusk on the glass / a fly traces each letter / ink of the old pond
- sugar on the glass / the fly learns five seven five / autumn in the code
- the cursor looms / wings blur a sudden takeoff / quiet on the pane
- male wiring hums / Kenyon cells taste the mora / rain on the window
- old pond waiting / a frog and a fly both jump / one sound, two ripples

## Credit

- MaleCNS: Berg et al., *Cell* (2026); pack with `tools/pack_malecns.py`
- LIF: Shiu et al. 2024
- KC→MBON: Hige 2015; Aso & Rubin compartments
