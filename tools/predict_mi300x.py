"""Predict Qwen3-8B serving on AMD MI300X *before* measuring it.

This is a prediction, not a measurement. It uses the roofline model in
inferscope/roofline.py with two efficiency factors taken from runs that did
happen, and is back-tested on the 2x T4 runs first. When MI300X time is
available, scripts/cloud_multi_gpu.sh measures the same quantities and the
README reports predicted vs measured.

Only quantities a roofline can predict honestly are predicted:
  - single-request decode time per token (TPOT at concurrency 1)
  - KV-cache capacity
Throughput under load is not: prefill interleaving and queueing dominate it
(see the laptop results), and they are outside this model.

    python tools/predict_mi300x.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferscope.roofline import ModelShape  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

# Efficiency factors, each a (low, high) band measured on hardware we ran:
#   bw_frac   sustained read bandwidth / datasheet bandwidth
#             RTX 3050 Laptop 116.6/168 = 0.69, Tesla T4 275.9/320 = 0.86
#   dec_eff   bandwidth floor / measured single-request TPOT
#             T4 Qwen3-4B TP=1 0.65 ... RTX 3050 Qwen3-1.7B on AC 0.97
BW_FRAC = (0.69, 0.86)
DEC_EFF = (0.65, 0.97)
# Two all-reduces per layer under tensor parallelism. Per-call latency is an
# assumption (no MI300X measurement yet): a few microseconds on xGMI.
ALLREDUCE_US = (5.0, 20.0)

GPUS = {
    "Tesla T4": {"bw_gbs": 320.0, "vram_gb": 15.0},              # datasheet
    "AMD MI300X": {"bw_gbs": 5300.0, "vram_gb": 192.0},          # datasheet, HBM3
}


def shape(config: str, name: str) -> ModelShape:
    c = json.loads((HERE / "shapes" / config).read_text())
    m = ModelShape(name=name, layers=c["num_hidden_layers"], hidden=c["hidden_size"], heads=c["num_attention_heads"],
                   kv_heads=c["num_key_value_heads"], head_dim=c["head_dim"], intermediate=c["intermediate_size"],
                   vocab=c["vocab_size"], tied=bool(c["tie_word_embeddings"]), bits=16.0)
    m.weight_bytes = m.params * 2.0  # bf16 / fp16
    return m


def tpot_ms(m: ModelShape, gpu: str, tp: int) -> tuple[float, float]:
    """(fast, slow) single-request decode time per token."""
    weight = m.decode_weight_bytes(1) / tp
    bw = GPUS[gpu]["bw_gbs"] * 1e9
    fast = weight / (bw * BW_FRAC[1]) / DEC_EFF[1]
    slow = weight / (bw * BW_FRAC[0]) / DEC_EFF[0]
    if tp > 1:
        fast += 2 * m.layers * ALLREDUCE_US[0] * 1e-6
        slow += 2 * m.layers * ALLREDUCE_US[1] * 1e-6
    return round(fast * 1e3, 2), round(slow * 1e3, 2)


def kv_tokens(m: ModelShape, gpu: str, tp: int, util: float = 0.90, overhead_gb: float = 4.0) -> int:
    """KV heads are sharded across TP ranks, so each GPU stores 1/tp of every token."""
    free = util * GPUS[gpu]["vram_gb"] * 1e9 - m.weight_bytes / tp - overhead_gb * 1e9
    return int(free // (m.kv_bytes_per_token / tp))


def main() -> None:
    q4, q8 = shape("qwen3-4b.json", "Qwen3-4B"), shape("qwen3-8b.json", "Qwen3-8B")
    measured_t4 = {("Qwen3-4B", 1): 45.37, ("Qwen3-4B", 2): 22.6, ("Qwen3-8B", 2): 37.59}  # results/t4/
    backtest = []
    for (name, tp), meas in measured_t4.items():
        m = q4 if name == "Qwen3-4B" else q8
        lo, hi = tpot_ms(m, "Tesla T4", tp)
        backtest.append({"gpu": "Tesla T4", "model": name, "tp": tp, "predicted_tpot_ms": [lo, hi], "measured_tpot_ms": meas,
                         "inside_band": lo <= meas <= hi})
    worst_request = 1971 + 1751  # longest prompt + longest output in the trace
    pred = []
    for tp in (1, 2, 4, 8):
        lo, hi = tpot_ms(q8, "AMD MI300X", tp)
        kv = kv_tokens(q8, "AMD MI300X", tp)
        pred.append({"gpu": "AMD MI300X", "model": "Qwen3-8B bf16", "tp": tp, "predicted_tpot_ms": [lo, hi],
                     "predicted_kv_tokens": kv, "worst_case_requests_in_kv": kv // worst_request})
    out = {"status": "PREDICTION - not measured", "made_on": date.today().isoformat(),
           "assumptions": {"bw_frac": BW_FRAC, "decode_efficiency": DEC_EFF, "allreduce_us": ALLREDUCE_US,
                           "datasheet": GPUS, "kv_util": 0.90, "kv_overhead_gb_per_gpu": 4.0},
           "backtest_on_measured_t4": backtest, "mi300x": pred}
    (RESULTS / "predictions").mkdir(exist_ok=True)
    (RESULTS / "predictions" / "mi300x_qwen3-8b.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
