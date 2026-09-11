# MaleCNS v1.0 packed graph

The packed CSR (166,700 neurons, ~25.6M directed edges) is **not in git**. Build it in this repo:

```text
python tools/pack_malecns.py
```

That downloads Janelia feathers into `.cache/` and writes the CSR here. Upstream: [MaleCNS](https://male-cns.janelia.org/) (Berg et al., *Cell* 2026). LIF constants: Shiu et al. 2024.
