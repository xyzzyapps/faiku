"""Shiu LIF on Vulkan compute (wgpu-native → SPIR-V).

Picks a Vulkan adapter (discrete GPU if one exists, otherwise any
non-CPU Vulkan device). No vendor-specific paths.
Falls back to the CPU net if the adapter or shader fails.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from .connectome import MaleCNSGraph
from .lif import DELAY_MS, REFRACT_MS, TAU_G, TAU_V, V_REST, V_THRESH

WG = 256

WGSL = r"""
struct Sim {
    n: u32,
    delay: u32,
    rfc: u32,
    slot: u32,
    a: f32,
    b: f32,
    gscale: f32,
    vrest: f32,
    vthresh: f32,
    one_minus_a: f32,
}

@group(0) @binding(0) var<storage, read_write> sim: Sim;
@group(0) @binding(1) var<storage, read> ptr: array<u32>;
@group(0) @binding(2) var<storage, read> post: array<u32>;
@group(0) @binding(3) var<storage, read> w_i32: array<i32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> g: array<f32>;
@group(0) @binding(6) var<storage, read> drive: array<f32>;
@group(0) @binding(7) var<storage, read_write> refract: array<i32>;
@group(0) @binding(8) var<storage, read_write> spike_q: array<u32>;
@group(0) @binding(9) var<storage, read_write> counts: array<u32>;
@group(0) @binding(10) var<storage, read_write> fired: array<u32>;
@group(0) @binding(11) var<storage, read_write> g_delta: array<atomic<i32>>;

@compute @workgroup_size(256)
fn deliver(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = sim.n;
    if (i >= n) { return; }
    if (spike_q[sim.slot * n + i] == 0u) { return; }
    let start = ptr[i];
    let stop = ptr[i + 1u];
    for (var e = start; e < stop; e++) {
        atomicAdd(&g_delta[post[e]], w_i32[e]);
    }
}

@compute @workgroup_size(256)
fn evolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = sim.n;
    if (i >= n) { return; }
    let extra = f32(atomicLoad(&g_delta[i])) / 10000.0;
    atomicStore(&g_delta[i], 0);
    var gi = g[i] + extra;
    var vi = sim.vrest + (v[i] - sim.vrest) * sim.a + drive[i] * sim.one_minus_a + gi * sim.gscale;
    gi = gi * sim.b;
    var r = refract[i];
    if (r > 0) { r = r - 1; }
    var sp = 0u;
    if (r == 0 && vi > sim.vthresh) {
        vi = sim.vrest;
        gi = 0.0;
        r = i32(sim.rfc);
        sp = 1u;
        counts[i] = counts[i] + 1u;
    }
    v[i] = vi;
    g[i] = gi;
    refract[i] = r;
    spike_q[sim.slot * n + i] = sp;
    fired[i] = fired[i] | sp;
}

@compute @workgroup_size(1)
fn inc_slot(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    sim.slot = (sim.slot + 1u) % sim.delay;
}
"""


def _pick_vulkan_adapter():
    os.environ.setdefault("WGPU_BACKEND_TYPE", "Vulkan")
    import wgpu

    discrete = None
    any_vk = None
    for adapter in wgpu.gpu.enumerate_adapters_sync():
        info = adapter.info
        backend = str(info.get("backend_type", "")).lower()
        kind = str(info.get("adapter_type", "")).lower()
        if backend != "vulkan":
            continue
        if kind in ("cpu", "software"):
            continue
        any_vk = adapter
        if kind == "discretegpu":
            discrete = adapter
            break
    return discrete or any_vk


@dataclass
class VulkanInfo:
    backend: str
    device: str
    adapter_type: str


class VulkanLifNet:
    def __init__(self, graph: MaleCNSGraph, dt_ms: float = 1.0) -> None:
        import wgpu

        adapter = _pick_vulkan_adapter()
        if adapter is None:
            raise RuntimeError("No Vulkan adapter (wgpu).")
        info = adapter.info
        self.info = VulkanInfo(
            backend=str(info.get("backend_type", "Vulkan")),
            device=str(info.get("device", "GPU")),
            adapter_type=str(info.get("adapter_type", "")),
        )
        self.device = adapter.request_device_sync(
            required_limits={
                "max-buffer-size": min(
                    int(adapter.limits.get("max-buffer-size", 1 << 30)),
                    1 << 31,
                ),
                "max-storage-buffer-binding-size": min(
                    int(adapter.limits.get("max-storage-buffer-binding-size", 1 << 30)),
                    1 << 31,
                ),
            }
        )
        self.gph = graph
        self.n = graph.n
        self.backend = "vulkan"
        self.dt = float(dt_ms)
        self.ptr = graph.ptr.astype(np.int64, copy=False)
        self.post = graph.post.astype(np.int32, copy=False)
        self.weight = graph.weight
        self.W = csr_matrix((self.weight, self.post, self.ptr), shape=(self.n, self.n))
        self.delay = max(1, int(round(DELAY_MS / self.dt)))
        self.rfc = max(1, int(round(REFRACT_MS / self.dt)))
        a = math.exp(-self.dt / TAU_V)
        b = math.exp(-self.dt / TAU_G)
        self._a = a
        self._b = b
        self._gscale = (a - b) / 3.0
        self.drive = np.zeros(self.n, dtype=np.float32)
        self.v = np.full(self.n, V_REST, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.counts = np.zeros(self.n, dtype=np.int32)
        self._weights_dirty = False
        self._ptr_u32 = graph.ptr.astype(np.uint32, copy=False)
        self._post_u32 = graph.post.astype(np.uint32, copy=False)
        self._w_i32 = np.clip(np.rint(self.weight * 10000.0), -2_000_000_000, 2_000_000_000).astype(np.int32)
        self._wgpu = wgpu
        self._n_groups = (self.n + WG - 1) // WG
        self._build_gpu()
        vram = (
            self.b_ptr.size + self.b_post.size + self.b_w.size
            + self.b_v.size + self.b_g.size + self.b_drive.size
            + self.b_refract.size + self.b_spike_q.size
            + self.b_counts.size + self.b_fired.size + self.b_g_delta.size
        )
        print(
            f"haiku: MaleCNS LIF on Vulkan\n"
            f"  adapter {self.info.device}  ({self.info.adapter_type or 'unknown'})\n"
            f"  backend {self.info.backend}\n"
            f"  buffers {vram / 1e6:.0f} MB  {self.n} cells  {int(self.post.size)} edges",
            flush=True,
        )
        self.advance(4)  # compile pipelines so the first live frame is not a stall

    def _buf(self, data: np.ndarray, usage):
        arr = np.ascontiguousarray(data)
        return self.device.create_buffer_with_data(data=arr, usage=usage)

    def _build_gpu(self) -> None:
        wgpu = self._wgpu
        ST = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        RO = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        raw = np.zeros(12, dtype=np.uint32)
        raw[0] = self.n
        raw[1] = self.delay
        raw[2] = self.rfc
        raw[3] = 0
        raw.view(np.float32)[4] = np.float32(self._a)
        raw.view(np.float32)[5] = np.float32(self._b)
        raw.view(np.float32)[6] = np.float32(self._gscale)
        raw.view(np.float32)[7] = np.float32(V_REST)
        raw.view(np.float32)[8] = np.float32(V_THRESH)
        raw.view(np.float32)[9] = np.float32(1.0 - self._a)
        self._meta_cpu = raw
        self.b_meta = self._buf(raw, ST)
        self.b_ptr = self._buf(self._ptr_u32, RO)
        self.b_post = self._buf(self._post_u32, RO)
        self.b_w = self._buf(self._w_i32, RO | wgpu.BufferUsage.COPY_DST)
        self.b_v = self._buf(self.v, ST)
        self.b_g = self._buf(self.g, ST)
        self.b_drive = self._buf(self.drive, RO)
        self.b_refract = self._buf(np.zeros(self.n, dtype=np.int32), ST)
        self.b_spike_q = self._buf(np.zeros(self.n * self.delay, dtype=np.uint32), ST)
        self.b_counts = self._buf(np.zeros(self.n, dtype=np.uint32), ST)
        self.b_fired = self._buf(np.zeros(self.n, dtype=np.uint32), ST)
        self.b_g_delta = self._buf(np.zeros(self.n, dtype=np.int32), ST)

        buffers = [
            self.b_meta, self.b_ptr, self.b_post, self.b_w,
            self.b_v, self.b_g, self.b_drive, self.b_refract,
            self.b_spike_q, self.b_counts, self.b_fired, self.b_g_delta,
        ]
        rw = {0, 4, 5, 7, 8, 9, 10, 11}
        layouts = []
        entries = []
        for i, buf in enumerate(buffers):
            layouts.append({
                "binding": i,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": (
                        wgpu.BufferBindingType.storage
                        if i in rw
                        else wgpu.BufferBindingType.read_only_storage
                    ),
                    "has_dynamic_offset": False,
                },
            })
            entries.append({
                "binding": i,
                "resource": {"buffer": buf, "offset": 0, "size": buf.size},
            })
        bgl = self.device.create_bind_group_layout(entries=layouts)
        self._bgl = self.device.create_bind_group(layout=bgl, entries=entries)
        pipe_layout = self.device.create_pipeline_layout(bind_group_layouts=[bgl])
        shader = self.device.create_shader_module(code=WGSL)
        self.pipe_deliver = self.device.create_compute_pipeline(
            layout=pipe_layout,
            compute={"module": shader, "entry_point": "deliver"},
        )
        self.pipe_evolve = self.device.create_compute_pipeline(
            layout=pipe_layout,
            compute={"module": shader, "entry_point": "evolve"},
        )
        self.pipe_inc = self.device.create_compute_pipeline(
            layout=pipe_layout,
            compute={"module": shader, "entry_point": "inc_slot"},
        )

    def reset_drive(self) -> None:
        self.drive.fill(0.0)

    def inject(self, idx: np.ndarray, current: float) -> None:
        if idx.size == 0 or abs(current) < 1e-6:
            return
        self.drive[idx] += np.float32(current)

    def update_weights(self, csr_edges: np.ndarray, values: np.ndarray) -> None:
        if csr_edges.size == 0:
            return
        self._w_i32[csr_edges] = np.clip(
            np.rint(values.astype(np.float64) * 10000.0),
            -2_000_000_000,
            2_000_000_000,
        ).astype(np.int32)
        self._weights_dirty = True

    def _flush_weights(self) -> None:
        if not self._weights_dirty:
            return
        self.device.queue.write_buffer(self.b_w, 0, np.ascontiguousarray(self._w_i32))
        self._weights_dirty = False

    def advance(self, steps: int = 1) -> np.ndarray:
        self._flush_weights()
        q = self.device.queue
        q.write_buffer(self.b_drive, 0, np.ascontiguousarray(self.drive))
        zeros = np.zeros(self.n, dtype=np.uint32)
        q.write_buffer(self.b_fired, 0, zeros)
        enc = self.device.create_command_encoder()
        nsteps = max(1, int(steps))
        for _ in range(nsteps):
            p = enc.begin_compute_pass()
            p.set_pipeline(self.pipe_deliver)
            p.set_bind_group(0, self._bgl)
            p.dispatch_workgroups(self._n_groups, 1, 1)
            p.end()
            p = enc.begin_compute_pass()
            p.set_pipeline(self.pipe_evolve)
            p.set_bind_group(0, self._bgl)
            p.dispatch_workgroups(self._n_groups, 1, 1)
            p.end()
            p = enc.begin_compute_pass()
            p.set_pipeline(self.pipe_inc)
            p.set_bind_group(0, self._bgl)
            p.dispatch_workgroups(1, 1, 1)
            p.end()
        q.submit([enc.finish()])
        mv = q.read_buffer(self.b_fired)
        fired = np.frombuffer(mv, dtype=np.uint32, count=self.n).copy()
        return np.flatnonzero(fired).astype(np.int32)


def try_vulkan_lif(graph: MaleCNSGraph, dt_ms: float = 1.0) -> VulkanLifNet | None:
    try:
        return VulkanLifNet(graph, dt_ms=dt_ms)
    except Exception as exc:
        print(f"haiku: Vulkan LIF unavailable ({exc})", flush=True)
        return None
