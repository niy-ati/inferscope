"""A fused Mixture-of-Experts layer in Triton, checked and timed against the
obvious PyTorch implementation and a bandwidth model.

An MoE layer routes each token to k of E expert MLPs (SwiGLU:
down(silu(gate(x)) * up(x))). The naive version loops over experts in
Python: one small GEMM per expert, lots of launches, poor reuse. The fused
version sorts (token, expert) pairs by expert, pads each expert's group to
the block size, and runs ONE grouped-GEMM kernel per projection where every
program block knows which expert's weights to load.

Shapes come from real models (one layer at a time fits in 6 GB):
  qwen3-30b-a3b   hidden 2048, expert dim 768,  128 experts, top-8
  olmoe-1b-7b     hidden 2048, expert dim 1024,  64 experts, top-8

    python -m inferscope.moe            # correctness + timing table
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

RESULTS = Path(__file__).resolve().parent.parent / "results"

SHAPES = {
    "qwen3-30b-a3b": dict(hidden=2048, inter=768, experts=128, top_k=8),
    "olmoe-1b-7b": dict(hidden=2048, inter=1024, experts=64, top_k=8),
}


GEMM_CONFIGS = [
    triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=st)
    for bn, bk, w, st in [(64, 64, 4, 3), (64, 128, 4, 3), (128, 64, 4, 3), (128, 64, 8, 3), (128, 128, 8, 2),
                          (64, 64, 4, 4), (128, 32, 4, 4), (256, 64, 8, 3), (64, 256, 4, 2)]
]


@triton.autotune(configs=GEMM_CONFIGS, key=["N", "K", "num_pairs_bucket", "BLOCK_M"])
@triton.jit
def grouped_gemm_kernel(
    a_ptr, w_ptr, c_ptr,  # a: [rows_a, K]; w: [E, N, K]; c: [num_pairs, N]
    sorted_ids_ptr, block_expert_ptr, pair_weight_ptr,
    num_pairs, N, K, a_row_div, num_experts, num_pairs_bucket,
    stride_am, stride_we, stride_wn, stride_cm,
    APPLY_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """c[pair] = a[pair // a_row_div] @ w[expert(pair)].T for every
    (token, expert) pair; pairs are pre-sorted so each M block has one expert."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert = tl.load(block_expert_ptr + pid_m)
    if expert >= num_experts:  # spare block from the worst-case sizing
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair = tl.load(sorted_ids_ptr + offs_m)
    valid = pair < num_pairs  # padding slots hold num_pairs
    a_row = pair // a_row_div
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :]
    w_ptrs = w_ptr + expert * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kmask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=valid[:, None] & kmask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=kmask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, w)
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    if APPLY_WEIGHT:
        acc *= tl.load(pair_weight_ptr + pair, mask=valid, other=0.0)[:, None]
    c_ptrs = c_ptr + pair[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=valid[:, None] & (offs_n[None, :] < N))


@triton.jit
def align_kernel(ids_ptr, sorted_ptr, block_expert_ptr, num_pairs, max_blocks,
                 E_P2: tl.constexpr, BLOCK_M: tl.constexpr, CHUNK: tl.constexpr):
    """One program does the whole alignment: histogram of experts, padded
    block offsets (prefix sum), a stable scatter of pair ids into their
    expert's slots, and the expert id of every block."""
    offs_e = tl.arange(0, E_P2)
    counts = tl.zeros([E_P2], dtype=tl.int32)
    for start in range(0, num_pairs, CHUNK):
        offs = start + tl.arange(0, CHUNK)
        e = tl.load(ids_ptr + offs, mask=offs < num_pairs, other=E_P2)
        counts += tl.sum((e[:, None] == offs_e[None, :]).to(tl.int32), 0)
    blocks = (counts + BLOCK_M - 1) // BLOCK_M
    cum_blocks = tl.cumsum(blocks, 0)
    base = (cum_blocks - blocks) * BLOCK_M
    for start in range(0, num_pairs, CHUNK):
        offs = start + tl.arange(0, CHUNK)
        e = tl.load(ids_ptr + offs, mask=offs < num_pairs, other=E_P2)
        onehot = (e[:, None] == offs_e[None, :]).to(tl.int32)
        rank = tl.cumsum(onehot, 0) - 1  # position within this chunk, per expert
        dest = tl.sum(onehot * (rank + base[None, :]), 1)
        tl.store(sorted_ptr + dest, offs, mask=offs < num_pairs)
        base += tl.sum(onehot, 0)
    for start in range(0, max_blocks, CHUNK):
        b = start + tl.arange(0, CHUNK)
        expert = tl.sum((cum_blocks[None, :] <= b[:, None]).to(tl.int32), 1)
        tl.store(block_expert_ptr + b, expert, mask=b < max_blocks)


def align_by_expert(topk_ids: torch.Tensor, experts: int, block_m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Same result as align_by_expert_torch, in one kernel launch (plus the
    fill). The torch version is ~30 small kernels; under WSL each launch
    costs ~24 us of CPU, so it took ~3 ms for ~70 us of GPU work."""
    flat = topk_ids.flatten().to(torch.int32)
    num_pairs = flat.numel()
    max_blocks = num_pairs // block_m + experts
    sorted_ids = torch.full((max_blocks * block_m,), num_pairs, dtype=torch.int32, device=flat.device)
    block_expert = torch.empty(max_blocks, dtype=torch.int32, device=flat.device)
    align_kernel[(1,)](flat, sorted_ids, block_expert, num_pairs, max_blocks,
                       E_P2=triton.next_power_of_2(experts), BLOCK_M=block_m, CHUNK=64, num_warps=4)
    return sorted_ids, block_expert


def align_by_expert_torch(topk_ids: torch.Tensor, experts: int, block_m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort flattened (token, slot) pairs by expert and pad each expert's
    run to a multiple of block_m. Returns padded pair ids and, per block,
    which expert it belongs to (== experts for spare blocks).

    Never reads a value back to the host: buffers are sized for the worst
    case (sum ceil(count/block_m) <= pairs/block_m + experts), so the whole
    thing stays asynchronous on the GPU. The first version sized them
    exactly with int(padded.sum()) + repeat_interleave, and those two host
    syncs were 75% of a single-token MoE layer."""
    flat = topk_ids.flatten()
    num_pairs = flat.numel()
    dev = flat.device
    counts = torch.bincount(flat, minlength=experts)
    blocks = (counts + block_m - 1) // block_m
    starts = (torch.cumsum(blocks, 0) - blocks) * block_m
    seg_starts = torch.cumsum(counts, 0) - counts
    order = torch.argsort(flat, stable=True)
    e_sorted = flat[order]
    rank = torch.arange(num_pairs, device=dev) - seg_starts[e_sorted]
    max_blocks = num_pairs // block_m + experts
    sorted_ids = torch.full((max_blocks * block_m,), num_pairs, dtype=torch.int32, device=dev)
    sorted_ids[starts[e_sorted] + rank] = order.to(torch.int32)
    block_expert = torch.searchsorted(torch.cumsum(blocks, 0), torch.arange(max_blocks, device=dev), right=True).to(torch.int32)
    return sorted_ids, block_expert


def _gemm(a, w, sorted_ids, block_expert, num_pairs, a_row_div, block_m, pair_weight=None):
    n, k = w.shape[1], w.shape[2]
    c = torch.empty(num_pairs, n, device=a.device, dtype=a.dtype)
    grid = lambda meta: (block_expert.numel(), triton.cdiv(n, meta["BLOCK_N"]))
    grouped_gemm_kernel[grid](
        a, w, c, sorted_ids, block_expert, pair_weight if pair_weight is not None else a,
        num_pairs, n, k, a_row_div, w.shape[0], triton.next_power_of_2(num_pairs), a.stride(0), w.stride(0), w.stride(1), c.stride(0),
        APPLY_WEIGHT=pair_weight is not None, BLOCK_M=block_m,
    )
    return c


def fused_moe(x, w_gate_up, w_down, topk_w, topk_ids, block_m: int | None = None):
    """x [T, H]; w_gate_up [E, 2I, H]; w_down [E, H, I]; topk_* [T, k].
    Small batches give each expert only a few tokens, so a 16-row M tile
    wastes less of each tensor-core op than 32/64."""
    t, k = topk_ids.shape
    e = w_gate_up.shape[0]
    block_m = block_m or (16 if t * k <= 2 * e else 32 if t * k <= 16 * e else 64)
    sorted_ids, block_expert = align_by_expert(topk_ids, e, block_m)
    h = _gemm(x, w_gate_up, sorted_ids, block_expert, t * k, k, block_m)  # [T*k, 2I]
    inter = w_down.shape[2]
    h = F.silu(h[:, :inter]) * h[:, inter:]
    y = _gemm(h, w_down, sorted_ids, block_expert, t * k, 1, block_m, pair_weight=topk_w.flatten().to(x.dtype))
    return y.view(t, k, -1).sum(1)


def naive_moe(x, w_gate_up, w_down, topk_w, topk_ids):
    """The straightforward version: loop over experts."""
    out = torch.zeros_like(x)
    inter = w_down.shape[2]
    for e in range(w_gate_up.shape[0]):
        tok, slot = torch.where(topk_ids == e)
        if tok.numel() == 0:
            continue
        h = x[tok] @ w_gate_up[e].T
        h = F.silu(h[:, :inter]) * h[:, inter:]
        out.index_add_(0, tok, (h @ w_down[e].T) * topk_w[tok, slot, None].to(x.dtype))
    return out


def reference_moe(x, w_gate_up, w_down, topk_w, topk_ids):
    """fp32 ground truth."""
    return naive_moe(x.float(), w_gate_up.float(), w_down.float(), topk_w.float(), topk_ids)


def make(shape: dict, tokens: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    h, i, e, k = shape["hidden"], shape["inter"], shape["experts"], shape["top_k"]
    x = torch.randn(tokens, h, device="cuda", dtype=torch.float16, generator=g)
    w1 = torch.randn(e, 2 * i, h, device="cuda", dtype=torch.float16, generator=g) / math.sqrt(h)
    w2 = torch.randn(e, h, i, device="cuda", dtype=torch.float16, generator=g) / math.sqrt(i)
    logits = torch.randn(tokens, e, device="cuda", generator=g)
    topk_w, topk_ids = torch.topk(torch.softmax(logits, -1), k, -1)
    return x, w1, w2, topk_w / topk_w.sum(-1, keepdim=True), topk_ids


def bytes_moved(shape: dict, tokens: int, topk_ids: torch.Tensor) -> float:
    """Lower bound on DRAM traffic: every touched expert's weights once,
    plus activations in and out (fp16)."""
    touched = topk_ids.unique().numel()
    w = touched * 3 * shape["hidden"] * shape["inter"] * 2
    act = tokens * shape["hidden"] * 2 * 2 + tokens * shape["top_k"] * shape["inter"] * 2 * 3 * 2
    return w + act


def bench(fn, iters: int = 20) -> float:
    return triton.testing.do_bench(fn, warmup=5, rep=iters * 10) / 1e3


def main() -> None:
    bw = None
    try:
        bw = json.loads((RESULTS / "hw.json").read_text())["meas_read_bw_gbs"]
    except (OSError, KeyError, ValueError):
        pass
    rows = []
    for name, shape in SHAPES.items():
        for tokens in (1, 8, 32, 128, 512):
            x, w1, w2, tw, ti = make(shape, tokens)
            ref = reference_moe(x, w1, w2, tw, ti)
            out = fused_moe(x, w1, w2, tw, ti)
            err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
            t_f, t_n = bench(lambda: fused_moe(x, w1, w2, tw, ti)), bench(lambda: naive_moe(x, w1, w2, tw, ti))
            mb = bytes_moved(shape, tokens, ti)
            row = {"model": name, "tokens": tokens, "experts_touched": ti.unique().numel(), "fused_ms": round(t_f * 1e3, 3),
                   "naive_ms": round(t_n * 1e3, 3), "speedup": round(t_n / t_f, 2), "max_rel_err": float(f"{err:.2e}"),
                   "fused_gbs": round(mb / t_f / 1e9, 1)}
            if bw:
                row["bound_ms"] = round(mb / (bw * 1e9) * 1e3, 3)
                row["fused_vs_bound"] = f"{(mb / (bw * 1e9)) / t_f:.0%}"
            rows.append(row)
            print(json.dumps(row))
            del x, w1, w2
            torch.cuda.empty_cache()
    try:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "moe.json").write_text(json.dumps(rows, indent=2))
    except OSError:
        pass


if __name__ == "__main__":
    main()
