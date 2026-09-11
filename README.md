# Haiku

A *Drosophila* mushroom-body RL loop that learns to write **Japanese haiku** as a fly font.

Supabase’s flycoding demo maps MaleCNS spikes onto legs that type. Other people drew Latin alphabets as fly walk-traces. This repo does the same job for **kana**: Kenyon-cell odor channels encode each mora, PAM / PPL1 write the lesson onto KC→MBON (and a thin motor map), and the fly’s “pen” is a 2-D ink trail.

This tree is **standalone**. Default brain is a fast 8-channel mushroom-body loop. Packed **MaleCNS v1.0** (166,700 LIF cells) is opt-in after you pack it **here**.

```text
idea  →  mora as odor  →  KC sparse code  →  motor (dx, dy, pen)
reward (ink vs glyph)  →  PAM if match / PPL1 if miss  →  KC→MBON + W_motor
```

This is **not** a reconstructed fly mind and **not** a claim that fruit flies compose Bashō. It is Hige-style dopamine-gated plasticity on a real (or reduced) mushroom body, trained on 5–7–5.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

MaleCNS (optional):

```powershell
pip install pyarrow pandas
python tools/pack_malecns.py
```

Pack files land in `haiku/data/malecns_v1/` (gitignored). Raw feathers stay in `.cache/`.

## Run

```powershell
python -m haiku                  # MB loop, live canvas
python -m haiku --connectome     # MaleCNS LIF if packed
python -m haiku --connectome-cpu
python -m haiku --steps 4000 --no-ui
python -m haiku --poem basho
```

Ink snapshots land in `runs/`.

## Poems

Classic 5–7–5 in hiragana (see `haiku/corpus.py`):

- ふるいけや / かわずとびこむ / みずのおと
- しずかさや / いわにしみいる / せみのこえ
- なのはなや / つきはひがしに / ひはにしに

## Credit

- MaleCNS: Berg et al., *Cell* (2026); pack with `tools/pack_malecns.py`
- LIF: Shiu et al. 2024
- KC→MBON: Hige 2015; Aso & Rubin compartments
