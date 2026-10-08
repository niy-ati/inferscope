"""First-principles performance model for transformer inference.

Given a model's shape (from its HF config + checkpoint size) and a GPU's
ceilings (spec or measured, see hw.py), predict:

  prefill time   compute bound: 2 * params * tokens + attention FLOPs
  decode step    max(bytes moved / bandwidth, FLOPs / peak), where bytes =
                 weights read once per step + every sequence's KV cache
  KV capacity    how many tokens of KV cache fit next to the weights

The model ignores kernel launch overhead, sampling, scheduling and imperfect
kernel efficiency on purpose: the gap between this bound and what an engine
measures is exactly the number worth explaining.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelShape:
    name: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tied: bool
    experts: int = 0  # 0 = dense
    experts_per_tok: int = 0
    moe_intermediate: int = 0
    weight_bytes: float = 0.0  # what one full read of the served weights costs
    bits: float = 16.0
    kv_bytes_per_elem: float = 2.0

    # ---- parameter counts
    @property
    def attn_params(self) -> int:
        q = self.hidden * self.heads * self.head_dim
        kv = 2 * self.hidden * self.kv_heads * self.head_dim
        return q + kv + self.heads * self.head_dim * self.hidden

    @property
    def expert_params(self) -> int:
        return 3 * self.hidden * (self.moe_intermediate if self.experts else self.intermediate)

    @property
    def layer_params(self) -> int:
        mlp = self.experts * self.expert_params + self.hidden * self.experts if self.experts else self.expert_params
        return self.attn_params + mlp

    @property
    def active_layer_params(self) -> int:
        mlp = self.experts_per_tok * self.expert_params if self.experts else self.expert_params
        return self.attn_params + mlp

    @property
    def params(self) -> int:
        return self.layers * self.layer_params + self.vocab * self.hidden * (1 if self.tied else 2)

    @property
    def active_params(self) -> int:
        """Weights one token touches in its forward pass (incl. the LM head)."""
        return self.layers * self.active_layer_params + self.vocab * self.hidden

    @property
    def kv_bytes_per_token(self) -> float:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.kv_bytes_per_elem

    # ---- bytes read per decode step
    def decode_weight_bytes(self, batch: int) -> float:
        """Bytes of weights read by one decode step for `batch` sequences.
        Dense: every layer weight + the LM head, once per step, whatever the
        batch (that is why batching is nearly free until compute binds).
        MoE: only experts some token in the batch was routed to."""
        bpp = self.weight_bytes / self.params if self.weight_bytes else self.bits / 8
        if not self.experts:
            return (self.layers * self.layer_params + self.vocab * self.hidden) * bpp
        k, e = self.experts_per_tok, self.experts
        touched = e * (1 - (1 - k / e) ** batch)  # expected distinct experts per layer
        per_layer = self.attn_params + touched * self.expert_params
        return (self.layers * per_layer + self.vocab * self.hidden) * bpp


def load_shape(model_dir: str | os.PathLike, name: str | None = None, kv_bytes: float = 2.0) -> ModelShape:
    d = Path(model_dir)
    c = json.loads((d / "config.json").read_text())
    c = c.get("text_config", c)
    q = c.get("quantization_config") or {}
    bits = float(q.get("bits") or q.get("w_bit") or 16)
    files = list(d.glob("*.safetensors")) or list(d.glob("*.bin"))
    return ModelShape(
        name=name or d.name,
        layers=c["num_hidden_layers"],
        hidden=c["hidden_size"],
        heads=c["num_attention_heads"],
        kv_heads=c.get("num_key_value_heads", c["num_attention_heads"]),
        head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
        intermediate=c.get("intermediate_size", 0),
        vocab=c["vocab_size"],
        tied=bool(c.get("tie_word_embeddings", False)),
        experts=c.get("num_experts") or c.get("num_local_experts") or 0,
        experts_per_tok=c.get("num_experts_per_tok", 0),
        moe_intermediate=c.get("moe_intermediate_size", 0),
        weight_bytes=float(sum(f.stat().st_size for f in files)),
        bits=bits,
        kv_bytes_per_elem=kv_bytes,
    )


@dataclass
class Ceilings:
    bw_gbs: float
    tflops: float
    vram_gb: float


def prefill_s(m: ModelShape, hw: Ceilings, tokens: int) -> float:
    flops = 2 * m.active_params * tokens + 2 * m.layers * tokens * tokens * m.heads * m.head_dim
    return max(flops / (hw.tflops * 1e12), m.decode_weight_bytes(1) / (hw.bw_gbs * 1e9))


def decode_step_s(m: ModelShape, hw: Ceilings, batch: int, context: int) -> dict[str, float]:
    """One decode step for `batch` sequences that each hold `context` tokens."""
    weight = m.decode_weight_bytes(batch)
    kv = batch * context * m.kv_bytes_per_token
    flops = 2 * m.active_params * batch + 4 * m.layers * context * m.heads * m.head_dim * batch
    t_mem = (weight + kv) / (hw.bw_gbs * 1e9)
    t_cmp = flops / (hw.tflops * 1e12)
    return {
        "step_s": max(t_mem, t_cmp),
        "bound": "memory" if t_mem >= t_cmp else "compute",
        "weight_gb": weight / 1e9,
        "kv_gb": kv / 1e9,
        "tok_s": batch / max(t_mem, t_cmp),
        "per_seq_tok_s": 1 / max(t_mem, t_cmp),
    }


def kv_capacity_tokens(m: ModelShape, hw: Ceilings, gpu_util: float, overhead_gb: float) -> int:
    """KV-cache tokens that fit; `overhead_gb` = activations, CUDA context,
    graphs (measured from the engine's own startup log when available)."""
    free = gpu_util * hw.vram_gb * 2**30 - m.weight_bytes - overhead_gb * 2**30
    return max(0, int(free // m.kv_bytes_per_token))


def crossover_batch(m: ModelShape, hw: Ceilings, context: int, max_batch: int = 512) -> int | None:
    """Smallest batch at which decode turns compute bound."""
    for b in range(1, max_batch + 1):
        if decode_step_s(m, hw, b, context)["bound"] == "compute":
            return b
    return None
