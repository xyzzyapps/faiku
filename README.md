# Haiku

A *Drosophila* mushroom-body RL loop that learns to write **Japanese haiku** as a fly font.

Supabase’s flycoding demo maps MaleCNS spikes onto legs that type. Other people drew Latin alphabets as fly walk-traces. This repo does the same job for **kana**: Kenyon-cell odor channels encode each mora, PAM / PPL1 write the lesson onto KC→MBON (and a thin motor map), and the fly’s “pen” is a 2-D ink trail.

Default brain is the same **fast 8-channel MB loop** Holmsy uses. Packed **MaleCNS v1.0** (166,700 LIF cells) is opt-in if you already packed it for Holmsy.

```text
idea  →  mora as odor  →  KC sparse code  →  motor (dx, dy, pen)
reward (ink vs glyph)  →  PAM if match / PPL1 if miss  →  KC→MBON + W_motor
```

This is **not** a reconstructed fly mind and **not** a claim that fruit flies compose Bashō. It is Hige-style dopamine-gated plasticity on a real (or reduced) mushroom body, trained on 5–7–5.

## Install

```powershell
cd haiku
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

MaleCNS (optional): pack once in Holmsy, then point here.

```powershell
cd ..\desktop-pet
python tools\pack_malecns.py
```

`haiku` looks for `../desktop-pet/flypet` on `PYTHONPATH` (or `HOLMSY_ROOT`).

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

- MaleCNS: Berg et al., *Cell* (2026); pack via Holmsy `tools/pack_malecns.py`
- LIF: Shiu et al. 2024
- KC→MBON: Hige 2015; Aso & Rubin compartments
- Holmsy / flypet: Xyzzy Apps
