"""Download MaleCNS v1.0 feathers and pack a CSR graph for this repo.

Runtime only needs haiku/data/malecns_v1/. Raw feathers stay in .cache/ (gitignored).
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "malecns_v1"
OUT = ROOT / "haiku" / "data" / "malecns_v1"
FILES = {
    "annotations.feather": (
        "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
        "flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather"
    ),
    "neurotransmitters.feather": (
        "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
        "flat-connectome/body-neurotransmitters-male-cns-v1.0.feather"
    ),
    "edges.feather": (
        "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
        "flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    ),
}
SHARD = 90 * 1024 * 1024
WEIGHT_SCALE = 0.275  # Shiu / doomfly mV per synaptic contact


def log(msg: str) -> None:
    print(msg, flush=True)


def download(name: str, url: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / name
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and dest.stat().st_size > 0:
        log(f"have {name} ({dest.stat().st_size} bytes)")
        return dest
    already = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, method="GET")
    if already:
        req.add_header("Range", f"bytes={already}-")
        log(f"resume {name} at {already}")
    else:
        log(f"download {name}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        mode = "ab" if already and resp.status == 206 else "wb"
        if mode == "wb":
            already = 0
        total = already + int(resp.headers.get("Content-Length") or 0)
        n = already
        with part.open(mode) as out:
            while True:
                chunk = resp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                n += len(chunk)
                if n == total or n % (64 * 1024 * 1024) < 8 * 1024 * 1024:
                    log(f"  {name} {n / 1e6:.1f} MB")
    part.replace(dest)
    log(f"saved {name} ({dest.stat().st_size} bytes)")
    return dest


def zstd_compress(data: bytes, level: int = 12) -> bytes:
    from compression import zstd

    return zstd.compress(data, level=level)


def write_shards(stem: str, blob: bytes) -> list[str]:
    names = []
    if len(blob) <= SHARD:
        name = f"{stem}.zst"
        (OUT / name).write_bytes(blob)
        names.append(name)
        log(f"wrote {name} ({len(blob)} bytes)")
        return names
    for i in range(0, len(blob), SHARD):
        name = f"{stem}.{i // SHARD:02d}.zst"
        piece = blob[i : i + SHARD]
        (OUT / name).write_bytes(piece)
        names.append(name)
        log(f"wrote {name} ({len(piece)} bytes)")
    return names


def exact_ids(values) -> np.ndarray:
    items = np.asarray(values)
    if items.dtype.kind == "f":
        raise ValueError("Neuron IDs must be integers, never floats.")
    if items.dtype.kind in "iu":
        return items.astype(np.uint64)
    return np.asarray([str(v) for v in items], dtype=np.uint64)


def transmitter_sign(value: str) -> int:
    tokens = set(str(value).lower().split(","))
    fast = set()
    if "acetylcholine" in tokens:
        fast.add(1)
    if tokens & {"gaba", "glutamate", "histamine"}:
        fast.add(-1)
    if len(fast) == 1:
        return next(iter(fast))
    return 1


def group_mask(types: np.ndarray, pattern: str) -> np.ndarray:
    rx = re.compile(pattern, re.IGNORECASE)
    return np.fromiter((bool(rx.search(t)) for t in types), dtype=bool, count=len(types))


def pack() -> dict:
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
    import pandas as pd

    for name, url in FILES.items():
        download(name, url)

    OUT.mkdir(parents=True, exist_ok=True)
    log("reading annotations")
    frame = feather.read_table(CACHE / "annotations.feather").to_pandas()
    nt_frame = feather.read_table(CACHE / "neurotransmitters.feather").to_pandas()
    retain = frame.superclass.notna() & frame.superclass.astype(str).ne("")
    retain = retain & ~frame.status.eq("Glia")
    body = exact_ids(frame.bodyId)
    nt_frame = nt_frame.set_index("body")
    predicted = frame.bodyId.map(nt_frame["consensus_nt"])
    nodes = pd.DataFrame(
        {
            "source_id": body,
            "retained": np.asarray(retain, dtype=bool),
            "superclass": np.asarray(frame.superclass),
            "cell_type": np.asarray(frame.type),
            "neurotransmitter": np.asarray(predicted),
        }
    )
    nodes = nodes.loc[nodes.retained].sort_values("source_id", ignore_index=True)
    ids = exact_ids(nodes.source_id)
    n = len(ids)
    log(f"retained neurons {n}")

    pres: list[np.ndarray] = []
    posts: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    log("reading edges")
    reader = ipc.open_file(pa.memory_map(str(CACHE / "edges.feather"), "r"))
    for number in range(reader.num_record_batches):
        batch = reader.get_batch(number)
        pre = exact_ids(batch.column(batch.schema.get_field_index("body_pre")).to_numpy(zero_copy_only=False))
        post = exact_ids(batch.column(batch.schema.get_field_index("body_post")).to_numpy(zero_copy_only=False))
        w = np.asarray(batch.column(batch.schema.get_field_index("weight")).to_numpy(zero_copy_only=False))
        i = np.searchsorted(ids, pre)
        j = np.searchsorted(ids, post)
        keep = (i < n) & (j < n)
        keep &= ids[np.minimum(i, n - 1)] == pre
        keep &= ids[np.minimum(j, n - 1)] == post
        if not np.any(keep):
            continue
        pres.append(i[keep].astype(np.uint32))
        posts.append(j[keep].astype(np.uint32))
        counts.append(np.clip(np.rint(w[keep]), 1, 65535).astype(np.uint16))
        if number % 20 == 0:
            log(f"  edge batch {number}/{reader.num_record_batches}")
    pre = np.concatenate(pres)
    post = np.concatenate(posts)
    count = np.concatenate(counts)
    log(f"retained edges {len(pre)}")
    order = np.argsort(pre, kind="stable")
    pre = pre[order]
    post = post[order]
    count = count[order]
    ptr = np.zeros(n + 1, dtype=np.uint32)
    ptr[1:] = np.cumsum(np.bincount(pre, minlength=n)).astype(np.uint32)
    # sort posts inside each row for delta-friendly compression
    for i in range(n):
        a, b = int(ptr[i]), int(ptr[i + 1])
        if b - a > 1:
            sl = slice(a, b)
            idx = np.argsort(post[sl], kind="stable")
            post[sl] = post[sl][idx]
            count[sl] = count[sl][idx]

    types = nodes.cell_type.fillna("").astype(str).to_numpy()
    superclasses = nodes.superclass.fillna("unassigned").astype(str).to_numpy()
    nts = nodes.neurotransmitter.fillna("missing").astype(str).to_numpy()
    signs = np.fromiter((transmitter_sign(v) for v in nts), dtype=np.int8, count=n)

    groups = {
        "JO-A/B": group_mask(types, r"^JO-[AB]\b|^JO-[AB]$"),
        "JO-E/C": group_mask(types, r"^JO-[CEF]"),
        "JO": group_mask(types, r"^JO-"),
        "GF": group_mask(types, r"^DNp01$"),
        "DNp01": group_mask(types, r"^DNp01$"),
        "DNp09": group_mask(types, r"^DNp09$"),
        "DNa02": group_mask(types, r"^DNa02$"),
        "DNp20": group_mask(types, r"^DNp20$"),
        "MDN": group_mask(types, r"^MDN$"),
        "PAM": group_mask(types, r"^PAM"),
        "PPL1": group_mask(types, r"^PPL1"),
        "PPL101": group_mask(types, r"^PPL101$"),
        "KC": group_mask(types, r"^KC"),
        "MBON": group_mask(types, r"^MBON"),
        "MBON11": group_mask(types, r"^MBON11$"),
        "R1-R6": types == "R1-R6",
        "LB3c": types == "LB3c",
        "DN": group_mask(types, r"^DN"),
        "dopamine": np.array(["dopamine" in v.lower() for v in nts], dtype=bool),
    }
    group_idx = {name: np.flatnonzero(mask).astype(np.uint32) for name, mask in groups.items()}
    for name, idx in group_idx.items():
        log(f"  group {name}: {len(idx)}")

    kc = set(int(x) for x in group_idx["KC"])
    mbon = set(int(x) for x in group_idx["MBON"])
    plastic_edge = []
    plastic_pre = []
    plastic_post = []
    for i in group_idx["KC"]:
        a, b = int(ptr[i]), int(ptr[i + 1])
        for e in range(a, b):
            j = int(post[e])
            if j in mbon:
                plastic_edge.append(e)
                plastic_pre.append(int(i))
                plastic_post.append(j)
    log(f"KC→MBON edges {len(plastic_edge)}")

    # display order: superclass bands so the raster is readable
    sc_order = [
        "ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal",
        "cb_sensory", "cb_intrinsic", "descending_neuron", "ascending_neuron",
        "vnc_sensory", "vnc_intrinsic", "vnc_motor", "cb_motor",
    ]
    rank = {s: k for k, s in enumerate(sc_order)}
    sc_rank = np.fromiter((rank.get(s, 50) for s in superclasses), dtype=np.int16, count=n)
    display = np.argsort(sc_rank, kind="stable").astype(np.uint32)

    sc_table, sc_code = np.unique(superclasses, return_inverse=True)
    nt_table, nt_code = np.unique(nts, return_inverse=True)

    meta = {
        "dataset": "malecns_v1",
        "release": "MaleCNS v1.0",
        "neurons": int(n),
        "edges": int(len(post)),
        "synaptic_contacts": int(count.astype(np.uint64).sum()),
        "weight_scale": WEIGHT_SCALE,
        "lif": {
            "v_rest": -52.0,
            "v_thresh": -45.0,
            "tau_v_ms": 20.0,
            "tau_g_ms": 5.0,
            "delay_ms": 1.8,
            "refract_ms": 2.2,
            "weight_uS": WEIGHT_SCALE,
            "sign": "ACh +; GABA/glutamate/histamine -; else +",
        },
        "source": "https://male-cns.janelia.org/download/",
        "paper": "https://doi.org/10.1016/j.cell.2026.08.015",
        "groups": {k: int(v.size) for k, v in group_idx.items()},
        "kc_mbon_edges": int(len(plastic_edge)),
        "files": {},
    }

    log("compressing CSR")
    meta["files"]["post"] = write_shards("post.u32", zstd_compress(np.ascontiguousarray(post).tobytes()))
    meta["files"]["count"] = write_shards("count.u16", zstd_compress(np.ascontiguousarray(count).tobytes()))
    np.save(OUT / "ptr.u32.npy", ptr)
    np.save(OUT / "ids.u64.npy", ids)
    np.save(OUT / "sign.i8.npy", signs)
    np.save(OUT / "superclass.u8.npy", sc_code.astype(np.uint8))
    np.save(OUT / "nt.u8.npy", nt_code.astype(np.uint8))
    np.save(OUT / "display.u32.npy", display)
    np.save(OUT / "kc_mbon_edge.u32.npy", np.asarray(plastic_edge, dtype=np.uint32))
    np.save(OUT / "kc_mbon_pre.u32.npy", np.asarray(plastic_pre, dtype=np.uint32))
    np.save(OUT / "kc_mbon_post.u32.npy", np.asarray(plastic_post, dtype=np.uint32))
    (OUT / "superclass_table.json").write_text(json.dumps(sc_table.tolist()) + "\n", encoding="utf-8")
    (OUT / "nt_table.json").write_text(json.dumps(nt_table.tolist()) + "\n", encoding="utf-8")
    np.savez_compressed(OUT / "groups.npz", **group_idx)
    meta["files"]["ptr"] = "ptr.u32.npy"
    meta["files"]["ids"] = "ids.u64.npy"
    meta["files"]["sign"] = "sign.i8.npy"
    meta["files"]["groups"] = "groups.npz"
    (OUT / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(json.dumps({k: meta[k] for k in ("dataset", "neurons", "edges", "groups", "kc_mbon_edges")}, indent=2))
    return meta


if __name__ == "__main__":
    pack()
    sys.exit(0)
