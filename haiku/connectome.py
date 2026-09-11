"""Load packed MaleCNS v1.0 from haiku/data/malecns_v1/ (built by tools/pack_malecns.py)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data" / "malecns_v1"
WEIGHT_SCALE = 0.275


@dataclass
class MaleCNSGraph:
    n: int
    n_edges: int
    ids: np.ndarray
    ptr: np.ndarray
    post: np.ndarray
    weight: np.ndarray
    weight0: np.ndarray
    signs: np.ndarray
    superclass: np.ndarray
    superclass_table: list[str]
    nt_table: list[str]
    nt_code: np.ndarray
    display: np.ndarray
    groups: dict[str, np.ndarray]
    kc_mbon_edge: np.ndarray
    kc_mbon_pre: np.ndarray
    kc_mbon_post: np.ndarray
    xy: np.ndarray | None = None
    manifest: dict = field(default_factory=dict)


def _decompress_shards(names: list[str]) -> bytes:
    from compression import zstd

    parts = []
    for name in names:
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(zstd.decompress(path.read_bytes()))
    return b"".join(parts)


def available() -> bool:
    return (DATA_DIR / "manifest.json").exists() and (DATA_DIR / "ptr.u32.npy").exists()


def load_graph() -> MaleCNSGraph | None:
    if not available():
        return None
    meta = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    n = int(meta["neurons"])
    ptr = np.load(DATA_DIR / "ptr.u32.npy")
    ids = np.load(DATA_DIR / "ids.u64.npy")
    signs = np.load(DATA_DIR / "sign.i8.npy")
    post = np.frombuffer(_decompress_shards(meta["files"]["post"]), dtype=np.uint32).copy()
    count = np.frombuffer(_decompress_shards(meta["files"]["count"]), dtype=np.uint16)
    if post.size != int(meta["edges"]) or count.size != int(meta["edges"]):
        raise ValueError("Packed CSR length does not match the MaleCNS manifest.")
    degree = np.diff(ptr.astype(np.int64))
    pre = np.repeat(np.arange(n, dtype=np.int32), degree)
    weight = (
        count.astype(np.float32)
        * signs[pre].astype(np.float32)
        * float(meta.get("weight_scale", WEIGHT_SCALE))
    )
    groups = dict(np.load(DATA_DIR / "groups.npz"))
    sc_table = json.loads((DATA_DIR / "superclass_table.json").read_text(encoding="utf-8"))
    nt_table = json.loads((DATA_DIR / "nt_table.json").read_text(encoding="utf-8"))
    return MaleCNSGraph(
        n=n,
        n_edges=int(post.size),
        ids=ids,
        ptr=ptr,
        post=post,
        weight=weight,
        weight0=weight.copy(),
        signs=signs,
        superclass=np.load(DATA_DIR / "superclass.u8.npy"),
        superclass_table=sc_table,
        nt_table=nt_table,
        nt_code=np.load(DATA_DIR / "nt.u8.npy"),
        display=np.load(DATA_DIR / "display.u32.npy"),
        groups={k: np.asarray(v, dtype=np.uint32) for k, v in groups.items()},
        kc_mbon_edge=np.load(DATA_DIR / "kc_mbon_edge.u32.npy"),
        kc_mbon_pre=np.load(DATA_DIR / "kc_mbon_pre.u32.npy"),
        kc_mbon_post=np.load(DATA_DIR / "kc_mbon_post.u32.npy"),
        xy=(
            np.load(DATA_DIR / "xy.f32.npy")
            if (DATA_DIR / "xy.f32.npy").exists()
            else None
        ),
        manifest=meta,
    )
